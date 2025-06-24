import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from ..configs.model_configs import QuantConfig
from ..quantization.fp4_utils import Fp4QuantizedTensor, shuffle_matrix_a, shuffle_matrix_sf_a
from ..modules.fused_moe import Mapping # Re-using Mapping from fused_moe.py

class WeightMode(str, enum.Enum):
    # weight of a vanilla layer
    VANILLA = 'vanilla'
    # weight of a fused QKV linear layer
    FUSED_QKV_LINEAR = 'fused_qkv_linear'
    # weight of a fused gate and up linear layer
    FUSED_GATE_UP_LINEAR = 'fused_gate_up_linear'


@dataclass(kw_only=True)
class WeightsLoadingConfig:
    weight_mode: WeightMode = WeightMode.VANILLA
    ignore_tensor_parallel: bool = False


class TensorParallelMode(str, enum.Enum):
    COLUMN = 'column'
    ROW = 'row'

    @classmethod
    def split_dim(cls, mode):
        return 1 if mode == cls.ROW else 0


def load_weight_shard(
        weight,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        device: torch.device = torch.device('cuda'), # Assuming CUDA for plugin
) -> torch.Tensor:
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor,
                                          indices: slice = None):
            if indices is None:
                # Avoid unnecessary copy
                return tensor.to(device)
            else:
                return tensor[indices].to(device)
    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to the module
    # safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice, tuple[slice]] = slice(None)):
            return tensor[indices].to(device)
    else:
        raise ValueError(f'unsupported weight type: {type(weight)}')
    if tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)

    split_dim = TensorParallelMode.split_dim(tensor_parallel_mode)

    if len(tensor_shape) == 1 and split_dim == 1:
        return maybe_convert_to_torch_tensor(weight)

    width = tensor_shape[split_dim]
    if width == 1:
        return maybe_convert_to_torch_tensor(weight)

    slice_width = math.ceil(width / tensor_parallel_size)
    slice_start = tensor_parallel_rank * slice_width
    slice_end = min((tensor_parallel_rank + 1) * slice_width, width)
    slice_obj = [slice(None)] * len(tensor_shape)
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))


def copy_weight(dst: Parameter, src: torch.Tensor):
    if dst.dtype != src.dtype:
        src = src.to(dst.dtype)
    assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
    dst.data.copy_(src)


def load_weights_vanilla_helper(module: 'Linear', weights: List[Dict]):
    assert len(weights) == 1
    device = torch.device('cuda')

    weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                               module.tp_rank, module.tp_mode, device)
    copy_weight(module.weight, weight)

    if module.bias is not None:
        bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                 module.tp_rank, module.tp_mode, device)
        copy_weight(module.bias, bias)


def load_weights_fused_qkv_helper(
        module: 'Linear',
        weights: List[Dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(weights) == 3
    device = torch.device('cuda')

    q_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                                 module.tp_rank, module.tp_mode, device)
    k_weight = load_weight_shard(weights[1]['weight'], module.tp_size,
                                 module.tp_rank, module.tp_mode, device)
    v_weight = load_weight_shard(weights[2]['weight'], module.tp_size,
                                 module.tp_rank, module.tp_mode, device)

    if module.bias is not None:
        q_bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                   module.tp_rank, module.tp_mode, device)
        k_bias = load_weight_shard(weights[1]['bias'], module.tp_size,
                                   module.tp_rank, module.tp_mode, device)
        v_bias = load_weight_shard(weights[2]['bias'], module.tp_size,
                                   module.tp_rank, module.tp_mode, device)
        copy_weight(module.bias, torch.cat((q_bias, k_bias, v_bias)))

    return (q_weight, k_weight, v_weight)


def load_weights_fused_gate_up_helper(
        module: 'Linear',
        weights: List[Dict]) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(weights) == 2
    device = torch.device('cuda')

    gate_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                                    module.tp_rank, module.tp_mode, device)
    up_weight = load_weight_shard(weights[1]['weight'], module.tp_size,
                                  module.tp_rank, module.tp_mode, device)
    if module.bias is not None:
        gate_bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                      module.tp_rank, module.tp_mode, device)
        up_bias = load_weight_shard(weights[1]['bias'], module.tp_size,
                                    module.tp_rank, module.tp_mode, device)
        copy_weight(module.bias, torch.cat((up_bias, gate_bias)))
    return (gate_weight, up_weight)


class LinearMethodBase(ABC):
    \"\"\"
    Base class for all linear methods.
    \"\"\"

    @abstractmethod
    def create_weights(self, module: 'Linear', in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype, *args,
                       **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply(self, module: 'Linear', input: torch.Tensor,
              bias: Optional[torch.Tensor], *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, module: 'Linear', weights: List[Dict],
                     weight_mode: WeightMode):
        \"\"\"
        Load weights from the checkpoint.
        \"\"\"
        if weight_mode == WeightMode.VANILLA:
            self.load_weights_vanilla(module, weights)
        elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
            self.load_weights_fused_qkv_linear(module, weights)
        elif weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
            self.load_weights_fused_gate_up_linear(module, weights)
        else:
            raise ValueError(f'unsupported weight mode: {weight_mode}')

    def load_weight_scales(self, weights: List[Dict], *args, **kwargs):
        \"\"\"
        Load quantized weight scales from the checkpoint.
        \"\"\"

    @abstractmethod
    def load_weights_vanilla(self, module: 'Linear', weights: List[Dict]):
        \"\"\"
        Load weights for the VANILLA weight mode.
        \"\"\"\
        raise NotImplementedError

    @abstractmethod
    def load_weights_fused_qkv_linear(self, module: 'Linear',
                                      weights: List[Dict]):
        \"\"\"
        Load weights for the FUSED_QKV_LINEAR weight mode.\n        \"\"\"\
        raise NotImplementedError

    @abstractmethod
    def load_weights_fused_gate_up_linear(self, module: 'Linear',
                                          weights: List[Dict]):
        \"\"\"
        Load weights for the FUSED_GATE_UP_LINEAR weight mode.\n        \"\"\"\
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):

    def create_weights(self, module: 'Linear', in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        weight_shape = (out_features, in_features)
        module.weight = Parameter(torch.empty(weight_shape, dtype=dtype),
                                  requires_grad=False)

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: 'Linear', input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        # Removed custom cublas_mm for standalone
        output = F.linear(input, module.weight, bias)
        return output

    def load_weights_vanilla(self, module: 'Linear', weights: List[Dict]):
        load_weights_vanilla_helper(module, weights)

    def load_weights_fused_qkv_linear(self, module: 'Linear',
                                      weights: List[Dict]):
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights)
        weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        copy_weight(module.weight, weight)

    def load_weights_fused_gate_up_linear(self, module: 'Linear',
                                          weights: List[Dict]):
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights)
        weight = torch.cat((up_weight, gate_weight), dim=0)
        copy_weight(module.weight, weight)


class NVFP4LinearMethod(LinearMethodBase):

    def create_weights(self, module: 'Linear', in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        # packed fp4 weights (16 x nvfp4 values)
        weight_shape = (out_features // 16 * 4, in_features)
        module.weight = Parameter(torch.empty(
            weight_shape, dtype=torch.int64),
                                  requires_grad=False)

        # per-block scaling factors (4 x fp8 values)
        scales_shape = (out_features // 4, in_features // 16)
        module.weight_scales = Parameter(torch.empty(
            scales_shape, dtype=torch.int32),
                                         requires_grad=False)

        if bias:
            module.bias = Parameter(torch.empty((out_features), dtype=dtype),
                                    requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: 'Linear', input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        # Simplified: Assuming input is not Fp4QuantizedTensor for now, and skipping custom op.
        # This part likely needs custom CUDA kernel for optimal FP4 performance.
        # For a basic functional plugin, we'll dequantize and use F.linear.

        # Placeholder: This will require a proper FP4 GEMM kernel.
        # For now, we dequantize and perform standard linear.
        # This will be slow for FP4.

        # Dequantize weights and scales to float for `F.linear`
        # This is a simplification and not the intended efficient FP4 path.
        # Proper FP4 inference requires custom kernel operations.

        # This part requires dedicated NVFP4 GEMM kernel
        # For a standalone plugin without TRT-LLM's custom kernels,
        # we would need to implement an equivalent CUDA kernel or fall back to dequantization.
        # For now, we will assume dequantization is performed elsewhere or
        # the model will be loaded with already dequantized weights if no custom kernel is available.

        # If Fp4QuantizedTensor input is supported, this logic needs to be enhanced
        # with actual FP4 GEMM or a dequantization followed by linear.

        raise NotImplementedError(
            "NVFP4 apply method requires custom CUDA kernels for efficient FP4 GEMM."
            " Currently, it's not implemented for standalone plugin."
            " You might need to add a custom kernel or convert weights to FP16/BF16"
            " before passing them to this linear layer for unquantized operation."
        )

    def load_weight_scales(self,
                           weights: List[Dict],
                           tp_size: int = 1,
                           tp_rank: int = 0,
                           tp_mode: Optional[TensorParallelMode] = None):
        assert len(weights) == 2
        device = torch.device('cuda')

        weight_scales = load_weight_shard(weights[0]['per_block_scaling_factor'], tp_size,
                                          tp_rank, tp_mode, device)
        weight_scales = shuffle_matrix_sf_a(weight_scales)

        input_scales = load_weight_shard(weights[1]['act_scales'], tp_size,
                                         tp_rank, tp_mode, device)
        input_scales = shuffle_matrix_sf_a(input_scales)
        # Assuming only global input scale is available for now.
        # More complex input scaling (e.g., per-token) would require more logic.
        return (weight_scales, input_scales)


    def load_weights_vanilla(self, module: 'Linear', weights: List[Dict]):
        # Vanilla path for NVFP4 means loading packed FP4 weights and scales
        assert len(weights) == 2 # packed_weights, per_block_scaling_factor
        device = torch.device('cuda')

        packed_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                                          module.tp_rank, module.tp_mode, device)
        copy_weight(module.weight, packed_weight)

        per_block_scaling_factor = load_weight_shard(weights[1]['per_block_scaling_factor'], module.tp_size,
                                                    module.tp_rank, module.tp_mode, device)
        copy_weight(module.weight_scales, per_block_scaling_factor)

        if module.bias is not None:
            bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                     module.tp_rank, module.tp_mode, device)
            copy_weight(module.bias, bias)


    def load_weights_fused_qkv_linear(self, module: 'Linear',
                                      weights: List[Dict]):
        assert len(weights) == 6 # 3 packed_weights, 3 per_block_scaling_factors
        device = torch.device('cuda')

        q_packed_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                                            module.tp_rank, module.tp_mode, device)
        k_packed_weight = load_weight_shard(weights[1]['weight'], module.tp_size,
                                            module.tp_rank, module.tp_mode, device)
        v_packed_weight = load_weight_shard(weights[2]['weight'], module.tp_size,
                                            module.tp_rank, module.tp_mode, device)
        packed_weight = torch.cat((q_packed_weight, k_packed_weight, v_packed_weight), dim=0)
        copy_weight(module.weight, packed_weight)

        q_per_block_scaling_factor = load_weight_shard(weights[3]['per_block_scaling_factor'], module.tp_size,
                                                       module.tp_rank, module.tp_mode, device)
        k_per_block_scaling_factor = load_weight_shard(weights[4]['per_block_scaling_factor'], module.tp_size,
                                                       module.tp_rank, module.tp_mode, device)
        v_per_block_scaling_factor = load_weight_shard(weights[5]['per_block_scaling_factor'], module.tp_size,
                                                       module.tp_rank, module.tp_mode, device)
        per_block_scaling_factor = torch.cat((q_per_block_scaling_factor, k_per_block_scaling_factor, v_per_block_scaling_factor), dim=0)
        copy_weight(module.weight_scales, per_block_scaling_factor)


        if module.bias is not None:
            q_bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                       module.tp_rank, module.tp_mode, device)
            k_bias = load_weight_shard(weights[1]['bias'], module.tp_size,
                                       module.tp_rank, module.tp_mode, device)
            v_bias = load_weight_shard(weights[2]['bias'], module.tp_size,
                                       module.tp_rank, module.tp_mode, device)
            copy_weight(module.bias, torch.cat((q_bias, k_bias, v_bias)))


    def load_weights_fused_gate_up_linear(self, module: 'Linear',
                                          weights: List[Dict]):
        assert len(weights) == 4 # 2 packed_weights, 2 per_block_scaling_factors
        device = torch.device('cuda')

        gate_packed_weight = load_weight_shard(weights[0]['weight'], module.tp_size,
                                               module.tp_rank, module.tp_mode, device)
        up_packed_weight = load_weight_shard(weights[1]['weight'], module.tp_size,
                                             module.tp_rank, module.tp_mode, device)
        packed_weight = torch.cat((up_packed_weight, gate_packed_weight), dim=0) # Note the order is up, then gate
        copy_weight(module.weight, packed_weight)

        gate_per_block_scaling_factor = load_weight_shard(weights[2]['per_block_scaling_factor'], module.tp_size,
                                                          module.tp_rank, module.tp_mode, device)
        up_per_block_scaling_factor = load_weight_shard(weights[3]['per_block_scaling_factor'], module.tp_size,
                                                        module.tp_rank, module.tp_mode, device)
        per_block_scaling_factor = torch.cat((up_per_block_scaling_factor, gate_per_block_scaling_factor), dim=0) # Note the order is up, then gate
        copy_weight(module.weight_scales, per_block_scaling_factor)

        if module.bias is not None:
            gate_bias = load_weight_shard(weights[0]['bias'], module.tp_size,
                                          module.tp_rank, module.tp_mode, device)
            up_bias = load_weight_shard(weights[1]['bias'], module.tp_size,
                                        module.tp_rank, module.tp_mode, device)
            copy_weight(module.bias, torch.cat((up_bias, gate_bias))) # Note the order is up, then gate


def get_quant_method(quant_config: Optional[QuantConfig] = None):
    if quant_config is None or quant_config.quant_mode.is_float8_dq():
        return UnquantizedLinearMethod()
    elif quant_config.quant_mode.is_float8_block_wise_weight_only():
        # This method is not implemented yet
        raise NotImplementedError("FP8BlockScalesLinearMethod is not implemented for standalone plugin.")
    elif quant_config.quant_mode.is_int4_weight_only_per_group():
        return NVFP4LinearMethod()
    elif quant_config.quant_mode.is_int4_weight_only_per_group_act_int8():
        # This method is not implemented yet
        raise NotImplementedError("W4A8MXFP4FP8LinearMethod is not implemented for standalone plugin.")
    else:
        raise ValueError(f"Unsupported quant_config: {quant_config}")


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,  # COLUMN parallel only
        quant_config: Optional[QuantConfig] = None,
        weights_loading_config: Optional[WeightsLoadingConfig] = None,
        reduce_output: bool = True,  # ROW parallel only
        skip_create_weights_in_init: bool = False,
        # Removed use_custom_cublas_mm for standalone plugin
        # Removed lora for standalone plugin
        # Removed allreduce_strategy for standalone plugin
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = mapping.tp_size if mapping is not None else 1
        self.tp_rank = mapping.tp_rank if mapping is not None else 0
        self.tp_mode = tensor_parallel_mode
        self.gather_output = gather_output
        self.reduce_output = reduce_output
        self.quant_config = quant_config
        self.weights_loading_config = weights_loading_config or WeightsLoadingConfig()
        self.linear_method = get_quant_method(self.quant_config)

        if not skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        self.linear_method.create_weights(self, self.in_features,
                                          self.out_features, self.bias is not None,
                                          self.dtype if hasattr(self, 'dtype') else None) # Use self.dtype if it exists

    @property
    def has_any_quant(self):
        return self.quant_config is not None

    @property
    def has_fp8_qdq(self):
        return self.quant_config is not None and self.quant_config.quant_mode.is_float8_dq()

    @property
    def has_fp8_block_scales(self):
        return self.quant_config is not None and self.quant_config.quant_mode.is_float8_block_wise_weight_only()

    @property
    def has_nvfp4(self):
        return self.quant_config is not None and self.quant_config.quant_mode.is_int4_weight_only_per_group()

    def apply_linear(self,
                     input,
                     bias,
                     lora_params: Optional[dict] | None = None, # Removed lora for standalone plugin
                     layer_idx: Optional[int] | None = None):
        # Removed lora handling for standalone plugin
        return self.linear_method.apply(self, input, bias)

    # Removed _maybe_fuse_bias_into_allreduce for standalone plugin

    def forward(
        self,
        input: Union[torch.Tensor, Fp4QuantizedTensor],
        *,
        all_reduce_params: Optional[None] = None, # Simplified, all_reduce removed
        lora_params: Optional[dict] = None, # Removed lora for standalone plugin
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if self.tp_mode == TensorParallelMode.ROW:
            # Removed AllReduce/reduce_output logic for standalone plugin
            pass # No-op for standalone, as distributed functions are removed

        # Apply linear
        bias = self.bias if hasattr(self, 'bias') else None # Check if bias exists
        output = self.apply_linear(input, bias, lora_params, layer_idx)

        # Removed distributed logic for COLUMN parallel

        return output

    def load_weights(self, weights: List[Dict]):
        self.linear_method.load_weights(self, weights, self.weights_loading_config.weight_mode) 