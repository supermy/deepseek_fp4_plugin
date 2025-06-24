#include <torch/extension.h>
#include "fp4_kernels.cuh"

void invoke_nvfp4_block_scale_interleave_cuda(
    int b, int m, int n, torch::Tensor sf_in, torch::Tensor sf_output, int multi_processor_count) {
    CHECK_CUDA_INPUT(sf_in);
    CHECK_CUDA_INPUT(sf_output);

    AT_ASSERTM(sf_in.dim() == 1, "sf_in must be a 1D tensor");
    AT_ASSERTM(sf_output.dim() == 1, "sf_output must be a 1D tensor");

    // Ensure tensors are contiguous
    sf_in = sf_in.contiguous();
    sf_output = sf_output.contiguous();

    deepseek_fp4_plugin::kernels::invokeNVFP4BlockScaleInterleave(
        b, m, n,
        reinterpret_cast<uint8_t const*>(sf_in.data_ptr<uint8_t>()),
        reinterpret_cast<uint8_t*>(sf_output.data_ptr<uint8_t>()),
        multi_processor_count,
        at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invoke_nvfp4_block_scale_interleave", &invoke_nvfp4_block_scale_interleave_cuda, "Invoke NVFP4 Block Scale Interleave CUDA (CUDA)");
} 