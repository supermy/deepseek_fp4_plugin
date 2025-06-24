import torch
import deepseek_fp4_plugin_cuda as _C

# The declarations must be aligned with thUtils.h
SF_DTYPE = torch.uint8
FLOAT4_E2M1X2 = torch.uint8

# Taken from https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime//modelConfig.h#L38
# TODO: move to model config, tune for blackwell hardware
FP4_BUCKETS = [64, 128, 256, 512, 1024]

# Export
float4_e2m1x2 = FLOAT4_E2M1X2
float4_sf_dtype = SF_DTYPE
fp4_buckets = FP4_BUCKETS

__all__ = ['float4_e2m1x2', 'float4_sf_dtype', 'pad_up', 'fp4_buckets']


def get_fp4_shape(input_shape, sf_vec_size):
    m = 1
    for i in range(len(input_shape) - 1):
        m *= input_shape[i]

    output_shape = [i for i in input_shape]
    output_shape[-1] //= 2

    scale_shape = pad_up(m, 128) * pad_up(input_shape[-1] // sf_vec_size, 4)
    return output_shape, scale_shape


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    """
    Reorders rows in the gemm/MOE_gemm weight matrix for min-latency
    [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    to
    [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    """
    assert x.dim() == 2, f"x should be a 2D tensor, not {x.dim()}"
    M, K = x.shape
    assert M % 2 == 0, f"x.shape[0] must be even, not {M}"

    row_indices = torch.arange(M, dtype=torch.long)

    # We split into top half and bottom half, but if M is odd,
    # the bottom half is one row larger.
    top = row_indices[:(M + 1) // 2]  # round up
    bot = row_indices[(M + 1) // 2:]  # remainder

    # Create the output
    permuted_row_indices = torch.empty_like(row_indices)

    # We'll place rows of `top` and `bot` in alternation
    permuted_row_indices[0::2] = top
    permuted_row_indices[1::2] = bot

    return permuted_row_indices


def reorder_rows_for_gated_act_gemm(x):
    """
    PyTorch implementation of trt-llm gen `reorderRowsForGatedActGemm`
    """
    row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(x)

    permute = lambda x: x[row_indices]

    return permute(x)


# yapf: disable
srcToDstBlk16RowMap = [
    0,  8,
    1,  9,
    2, 10,
    3, 11,
    4, 12,
    5, 13,
    6, 14,
    7, 15
]

srcToDstBlk32RowMap = [
    0,  8, 16, 24,
    1,  9, 17, 25,
    2, 10, 18, 26,
    3, 11, 19, 27,
    4, 12, 20, 28,
    5, 13, 21, 29,
    6, 14, 22, 30,
    7, 15, 23, 31
]
# yapf: enable


def get_shuffle_block_size(epilogue_tile_m: int) -> int:
    shuffle_block_size = 16
    if epilogue_tile_m % 128 == 0:
        shuffle_block_size = 32
    return shuffle_block_size


def get_shuffle_matrix_a_row_indices(input_tensor: torch.Tensor,
                                     epilogue_tile_m: int) -> torch.Tensor:
    """
    Higher-level PyTorch approach to reorder the rows in blocks of size 16 or 32.
    - We do NOT try to handle custom e2m1 memory usage (i.e. no 'K/2' bytes).
    - Instead, we purely reorder rows in a standard PyTorch shape [M, K].
    """
    assert input_tensor.dim(
    ) == 2, f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"

    # M, K from the input
    M, K = input_tensor.shape

    # Choose block size 16 or 32
    shuffle_block_size = get_shuffle_block_size(epilogue_tile_m)
    row_map = (srcToDstBlk16RowMap
               if shuffle_block_size == 16 else srcToDstBlk32RowMap)

    assert M % shuffle_block_size == 0, f"input_tensor.shape[0] must be multiples of {shuffle_block_size}"

    # row_indices[new_row] = old_row
    # so row_indices is an array of size M telling us from which old_row
    # the new_row should be taken.
    row_indices = torch.empty(M, dtype=torch.long)

    for old_row in range(M):
        block_idx = old_row // shuffle_block_size
        row_in_block = old_row % shuffle_block_size
        mapped_row_in_block = row_map[row_in_block]

        new_row = block_idx * shuffle_block_size + mapped_row_in_block

        row_indices[new_row] = old_row

    return row_indices


def shuffle_matrix_a(input_tensor: torch.Tensor,
                     epilogue_tile_m: int) -> torch.Tensor:
    """
    PyTorch equivalent of trtllm-gen `shuffleMatrixA`
    """
    row_indices = get_shuffle_matrix_a_row_indices(input_tensor,
                                                   epilogue_tile_m)

    # Replaced torch.ops.trtllm.shuffle_matrix with direct indexing
    return input_tensor[row_indices.to(input_tensor.device)]


def get_shuffle_matrix_sf_a_row_indices(
        input_tensor: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: int = 16) -> torch.Tensor:

    assert input_tensor.dtype == float4_sf_dtype
    assert num_elts_per_sf == 16

    assert input_tensor.dim(
    ) == 2, f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"

    # M, K from the input
    M, K = input_tensor.shape
    assert M % 128 == 0
    assert K % 4 == 0

    # The row_indices for SF_A is slightly different to the matrix A. The row_indices
    # is used to shuffle the 1D input SF.
    # So we want row_indices[new_idx] = old_idx.
    row_indices = torch.arange(M, dtype=torch.long)

    shuffle_block_size = get_shuffle_block_size(epilogue_tile_m)
    row_map = (srcToDstBlk16RowMap
               if shuffle_block_size == 16 else srcToDstBlk32RowMap)

    dst_indices = torch.empty_like(row_indices)
    for old_row in range(M):
        block_idx = old_row // shuffle_block_size
        row_in_block = old_row % shuffle_block_size
        mapped_row_in_block = row_map[row_in_block]

        new_row = block_idx * shuffle_block_size + mapped_row_in_block
        dst_indices[new_row] = old_row

    return dst_indices


def shuffle_matrix_sf_a(
    input_tensor: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int = 16,
):
    """
    Cuda implementation of trtllm-gen `shuffleMatrixSfA` but with a caveat.
    `shuffleMatrixSfA` expects the input to be in 128x4 layout and then
    apply the same shuffling in `shuffleMatrixA` and writes out in 128x4
    layout.
    This function expects the input to be in linear layout. It's done this
    way because the scaling factors in the NVFP4 checkpoints are quantized
    and are in linear layout.
    This function doesn't add padding.
    """
    M, K = input_tensor.shape
    num_batches = 1
    if input_tensor.dim() == 3:
        num_batches = input_tensor.shape[0]
        M = input_tensor.shape[1]
        K = input_tensor.shape[2]

    sf_output_size = pad_up(M, 128) * pad_up(K, 4)
    if num_batches > 1:
        sf_output_size *= num_batches

    sf_output = torch.empty(sf_output_size, dtype=input_tensor.dtype,
                            device=input_tensor.device)

    _C.invoke_nvfp4_block_scale_interleave(
        num_batches, M, K, input_tensor.view(-1).contiguous(), sf_output, torch.cuda.device_count())

    return sf_output


def pack_int4_weight_col_wise(weight: torch.Tensor, weight2: torch.Tensor = None):
    """
    Simplified placeholder for packing int4 weights column-wise.
    In a real scenario, this would involve specific bit manipulation and
    potentially custom CUDA kernels for optimal performance.
    For standalone plugin, we'll assume a basic concatenation for now.
    """
    if weight2 is not None:
        # Assuming weight and weight2 are already in a pseudo-int4 range (0-15 or -8 to 7)
        # For simplicity, we'll cast to int8 (or relevant type) and then pack.
        # This requires careful consideration of the actual int4 quantization scheme.
        # A typical packing would be (high_4_bits << 4) | low_4_bits

        # First, ensure they are integer-like and in the correct range for int4
        # This step would ideally be a proper quantization to int4.
        weight_int = weight.round().to(torch.int8)
        weight2_int = weight2.round().to(torch.int8)

        # Pack column-wise: each byte contains two 4-bit values from the same row, but different columns
        # If weight is [M, K], we want to pack K/2 columns.
        # This assumes weight and weight2 are designed to be interleaved or concatenated for packing.
        # For column-wise packing, it typically means W[row, 2*col] and W[row, 2*col+1] are packed together.
        # Given the previous context, it seems like weight and weight2 might represent interleaved halves.
        # Let's assume weight is the 'even' columns and weight2 is the 'odd' columns of the original float weight.
        # So, we pack weight's elements with weight2's elements.
        # (weight2_element << 4) | (weight_element & 0xF)

        # We need to reshape for column-wise packing if the input is treated as flat for concatenation.
        # Assuming input `weight` and `weight2` are already shaped such that their elements can be paired.
        # For a linear layer, weight is typically [out_features, in_features].
        # So, we are packing `in_features` into `in_features / 2` packed bytes.

        # Example: weight = [W0, W1, W2, W3], weight2 = [W4, W5, W6, W7]
        # Packed: [(W4 << 4) | W0, (W5 << 4) | W1, ...]

        # Transpose if necessary to handle column-wise packing correctly given the default weight orientation.
        # Assuming weight is [out_features, in_features] and we want to pack along in_features (columns).
        # The original TRT-LLM code seemed to involve a transpose before packing.

        # To simplify and align with the `(qweight[:, 1::2] * 16 + qweight[:, ::2])` pattern:
        # Let `weight` be the low 4 bits (even columns) and `weight2` be the high 4 bits (odd columns).
        # Need to ensure dimensions match for element-wise operations.
        assert weight_int.shape == weight2_int.shape, "Weight and Weight2 must have the same shape for packing."

        # The packing: (high_bits << 4) | low_bits
        # Assuming weight_int contains the lower nibbles and weight2_int contains the upper nibbles.
        packed_weight = (weight2_int << 4) | (weight_int & 0xF)
        
        # The result needs to be a torch.uint8 tensor
        return packed_weight.to(torch.uint8)
    else:
        # If only one weight tensor is provided, it's assumed to be pre-packed or a dummy case.
        # For actual int4 weight only, we expect two parts (e.g., from a split float weight).
        # If this path is taken with float/bfloat16, it implies no actual int4 packing is happening.
        # For now, we'll just cast to uint8 as a placeholder for a single weight tensor.
        return weight.to(torch.uint8) # Placeholder for single weight packing/conversion

class Fp4QuantizedTensor:
    def __init__(self, fp4_tensor: torch.Tensor, scaling_factor: torch.Tensor):
        self.fp4_tensor = fp4_tensor
        self.scaling_factor = scaling_factor

    @property
    def shape(self):
        return self.fp4_tensor.shape

_disable_fp4_allgather_flag = False

def disable_fp4_allgather():
    global _disable_fp4_allgather_flag
    _disable_fp4_allgather_flag = True

def enable_fp4_allgather():
    global _disable_fp4_allgather_flag
    _disable_fp4_allgather_flag = False

def is_fp4_allgather_disabled():
    global _disable_fp4_allgather_flag
    return _disable_fp4_allgather_flag
