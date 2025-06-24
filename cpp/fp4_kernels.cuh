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

__global__ void nvfp4_block_scale_interleave_kernel(
    int numbatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput)
{
    constexpr int SF_VEC_SIZE = 16;
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x)
            {
                int64_t inOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
                auto sf = SFIn[inOffset];

                std::optional<int> batchIdxOpt = batchIdx;
                std::optional<int> numRowsOpt = numRows;

                auto dstIdx
                    = get_sf_out_offset_128x4<SF_VEC_SIZE>(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols * 16);
                SFOutput[dstIdx] = sf;
            }
        }
    }
}

void invokeNVFP4BlockScaleInterleave(
    int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput, int multiProcessorCount, cudaStream_t stream);
} // namespace kernels
} // namespace deepseek_fp4_plugin 