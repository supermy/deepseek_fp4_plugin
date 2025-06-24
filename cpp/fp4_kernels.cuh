#pragma once

#include "common/assert.h"
#include "common/cudaTypeUtils.cuh"
#include "common/cudaUtils.h"
#include "common/quantTypeUtils.cuh"
#include "common/reduceKernelUtils.cuh"
#include <cuda_fp16.h>
#include <float.h>
#include <optional>
#include <cstdint> // for uint8_t, int32_t, int64_t

using namespace tensorrt_llm::common;

namespace deepseek_fp4_plugin
{
namespace kernels
{

enum class FP4QuantizationSFLayout
{
    SWIZZLED,
    LINEAR
};

#define PadUpFn(X, Y) ((X + Y - 1) / (Y) * (Y))

template <int SF_VEC_SIZE>
inline __device__ int64_t get_sf_out_offset_128x4(
    std::optional<int> batchIdx, int mIdx, int kIdx, std::optional<int> numRows, int numCols)
{
    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    // batched tensor
    // SF layout [numBTiles, numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [bTileIdx, mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4 * innerKStride; // 4

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * innerMStride; // 16

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * outerMStride; // 512

    // SF vector size 16. We round the "numCols" up to a multiple of 64.
    int factor = SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int32_t mTileIdx = mIdx / (32 * 4);
    int64_t mTileStride = numKTiles * kTileStride;

    // Each SF block has 128 rows so pad rows to the multiple of 128.
    int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
    int64_t bTileStride = numMTiles * mTileStride;

    // Compute the global offset.
    int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride + kTileIdx * kTileStride
        + outerMIdx * outerMStride + innerMIdx * innerMStride + innerKIdx * innerKStride;

    return SFOffset;
}

// 性能优化的核函数实现
namespace optimized_kernels {

// 向量化加载常量
constexpr int vec_size = 4;
constexpr int warp_size = 32;
constexpr int max_threads = 256;
constexpr int min_blocks = 8;

// 优化的矢量数据类型
template<typename T>
struct alignas(16) vec4_t {
    T x, y, z, w;
    
    __device__ vec4_t(): x(0), y(0), z(0), w(0) {}
    
    __device__ vec4_t(const T* ptr) {
        x = ptr[0];
        y = ptr[1];
        z = ptr[2];
        w = ptr[3];
    }
    
    __device__ void store(T* ptr) {
        ptr[0] = x;
        ptr[1] = y;
        ptr[2] = z;
        ptr[3] = w;
    }
};

// 优化的矢量乘法
template<typename T>
__device__ __forceinline__ float4 vec_mul(vec4_t<T> a, vec4_t<T> b) {
    return make_float4(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z,
        a.w * b.w
    );
}

// 使用向量化和共享内存的优化核函数
template<typename T>
__global__ void optimized_expert_compute_kernel(
    const T* __restrict__ inputs,
    const T* __restrict__ weights,
    T* __restrict__ outputs,
    const int batch_size,
    const int hidden_size,
    const int expert_size
) {
    // 分配共享内存
    extern __shared__ float smem[];
    
    // 计算线程和块索引
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int lane_id = tidx % warp_size;
    const int warp_id = tidx / warp_size;
    
    // 加载输入到共享内存
    for(int i = tidx; i < hidden_size; i += blockDim.x) {
        const int vec_idx = i / vec_size;
        if(vec_idx * vec_size < hidden_size) {
            // 使用向量化加载
            vec4_t<T> vec_input(inputs + bidx * hidden_size + i);
            reinterpret_cast<vec4_t<T>*>(smem)[vec_idx] = vec_input;
        }
    }
    __syncthreads();
    
    // 计算专家输出
    for(int i = tidx; i < expert_size; i += blockDim.x) {
        float sum = 0.0f;
        
        #pragma unroll 4
        for(int j = 0; j < hidden_size; j += vec_size) {
            // 向量化乘累加
            vec4_t<T> vec_input = reinterpret_cast<vec4_t<T>*>(smem)[j/vec_size];
            vec4_t<T> vec_weight(weights + i * hidden_size + j);
            
            float4 prod = vec_mul(vec_input, vec_weight);
            sum += prod.x + prod.y + prod.z + prod.w;
        }
        
        // 写回结果
        outputs[bidx * expert_size + i] = sum;
    }
}

// 优化的核函数启动包装器
void launch_expert_compute(
    torch::Tensor& inputs,
    torch::Tensor& weights,
    torch::Tensor& outputs,
    cudaStream_t stream
) {
    const int batch_size = inputs.size(0);
    const int hidden_size = inputs.size(1);
    const int expert_size = outputs.size(1);
    
    // 获取设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 计算最优启动配置
    const int warps_per_block = 8;
    const int threads = warp_size * warps_per_block;
    const int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / threads;
    const int num_sms = prop.multiProcessorCount;
    const int blocks = std::min(
        (batch_size + threads - 1) / threads,
        num_sms * max_blocks_per_sm
    );
    
    // 计算共享内存需求
    const int smem_size = hidden_size * sizeof(float);
    
    // 检查并设置最大共享内存配置
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, optimized_expert_compute_kernel<float>);
    if(smem_size > attr.maxDynamicSharedSizeBytes) {
        cudaFuncSetAttribute(
            optimized_expert_compute_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }
    
    // 启动优化核函数
    optimized_expert_compute_kernel<float>
        <<<blocks, threads, smem_size, stream>>>(
            inputs.data_ptr<float>(),
            weights.data_ptr<float>(),
            outputs.data_ptr<float>(),
            batch_size,
            hidden_size,
            expert_size
        );
}

// 优化的块尺度交错核函数
__global__ void nvfp4_block_scale_interleave_kernel(
    int numbatches,
    int numRows,
    int numCols,
    const uint8_t* __restrict__ SFIn,
    uint8_t* __restrict__ SFOutput
) {
    constexpr int SF_VEC_SIZE = 16;
    constexpr int TILE_DIM = 32;
    
    // 使用共享内存进行块内交换
    __shared__ uint8_t tile[TILE_DIM][TILE_DIM + 1]; // +1 避免bank冲突
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // 计算全局索引
    const int row = by * TILE_DIM + ty;
    const int col = bx * TILE_DIM + tx;
    
    // 加载到共享内存
    if(row < numRows && col < numCols) {
        tile[ty][tx] = SFIn[row * numCols + col];
    }
    __syncthreads();
    
    // 计算目标位置
    const int out_row = bx * TILE_DIM + ty;
    const int out_col = by * TILE_DIM + tx;
    
    // 写回全局内存,实现交错访问
    if(out_row < numCols && out_col < numRows) {
        const int out_idx = get_sf_out_offset_128x4<SF_VEC_SIZE>(
            0, out_col, out_row, numRows, numCols
        );
        SFOutput[out_idx] = tile[tx][ty];
    }
}

// 优化的块尺度交错启动包装器
void invokeNVFP4BlockScaleInterleave(
    int b,
    int m,
    int n,
    const uint8_t* SFIn,
    uint8_t* SFOutput,
    int multiProcessorCount,
    cudaStream_t stream
) {
    constexpr int TILE_DIM = 32;
    
    // 计算网格和块大小
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(
        (n + TILE_DIM - 1) / TILE_DIM,
        (m + TILE_DIM - 1) / TILE_DIM
    );
    
    // 启动优化核函数
    nvfp4_block_scale_interleave_kernel
        <<<grid, block, 0, stream>>>(
            b, m, n, SFIn, SFOutput
        );
}

} // namespace optimized_kernels

} // namespace kernels
} // namespace deepseek_fp4_plugin