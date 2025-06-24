from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..configs.model_configs import ModelConfig # Use local ModelConfig
from ..quantization.fp4_utils import Fp4QuantizedTensor
from ..modules.linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig # Use local Linear
from ..modules.fused_moe import Mapping # Use local Mapping

def swiglu(x):
    # Simplified: Removed flashinfer dependency for standalone plugin
    gate, x = x.chunk(2, dim=-1)
    return F.silu(gate) * x


class GatedMLP(nn.Module):

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        dtype: Optional[torch.dtype] = None,
        config: Optional[ModelConfig] = None,
        overridden_tp_size: Optional[int] = None, # Simplified, will not use for standalone
        reduce_output: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation

        config = config or ModelConfig() # Use local ModelConfig
        self.mapping = config.mapping # Simplified: Mapping should be provided or default to non-parallel
        
        # Simplified: Removed complex mapping logic related to overridden_tp_size as distributed is removed
        # For standalone, mapping should be simple (tp_size=1, etc.)
        if self.mapping is None: # Ensure mapping is not None if not provided by config
            self.mapping = Mapping()

        self.gate_up_proj = Linear(
            self.hidden_size,
            self.intermediate_size * 2,
            bias=bias,
            dtype=dtype,
            mapping=self.mapping, # Use local mapping
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
            quant_config=config.get_quant_config(),
            reduce_output=False,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            # allreduce_strategy removed
        )

        # Removed down_lora and related LoraLayer for standalone plugin
        # Removed splitted_gate_up_lora and fused_gate_up_lora for standalone plugin

        self.down_proj = Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=bias,
            dtype=dtype,
            mapping=self.mapping, # Use local mapping
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=config.get_quant_config(),
            reduce_output=reduce_output,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            # lora removed
            # allreduce_strategy removed
        )

    def _apply_activation(self, x):
        if self.activation == F.silu:
            return swiglu(x)
        elif self.activation == None:
            return x
        else:
            raise NotImplementedError(
                f"Activation {self.activation} not yet implemented for fused GatedMLP"
            )

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens=None, # Removed for standalone plugin
        final_all_reduce_params: Optional[None] = None, # Simplified for standalone plugin
        lora_params: Optional[dict] = None, # Removed for standalone plugin
        **kwargs,
    ) -> torch.Tensor:
        # Removed lora handling for standalone plugin
        if bool(lora_params): # Just in case lora_params is still passed, raise an error.
            raise NotImplementedError("LoRA is not supported in standalone DeepSeek FP4 plugin.")

        h1 = self.gate_up_proj(x)
        h2 = self._apply_activation(h1)
        output = self.down_proj(h2,
                                # all_reduce_params removed
                                layer_idx=self.layer_idx)
        return output

    # Removed forward_lora for standalone plugin 