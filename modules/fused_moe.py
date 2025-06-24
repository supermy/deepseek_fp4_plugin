import copy
import math
import os
import threading
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

# Local imports
from ..configs.model_configs import ModelConfig, MoeLoadBalancerConfig
from ..quantization.fp4_utils import (
    Fp4QuantizedTensor,
    get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
# Functionality to be reimplemented or simplified.


# The declarations aligns with moe_kernels.h
# pack inputs into int64, e.g. 4 x bf16 input values
FUSED_MOE_NVFP4_INPUT_DTYPE = torch.int64
# pack weights into int64, e.g. 16 x nvfp4 weight values
FUSED_MOE_NVFP4_WEIGHT_DTYPE = torch.int64
# pack weight block scales into int32, e.g. 4 x fp8 weight values
FUSED_MOE_NVFP4_WEIGHT_BLOCK_SCALE_DTYPE = torch.int32


# The type of method in top-K routing, for use in torch custom op
class RoutingMethodType(IntEnum):
    """
    Enum for different types of expert routing methods in MoE.
    MoE中不同专家路由方法的枚举类型。
    """
    Default = 0         # Default: Softmax -> TopK
                       # 默认方式：Softmax后取TopK
    Renormalize = 1    # Renormalize: TopK -> Softmax
                       # 重归一化：先TopK后Softmax
    DeepSeekV3 = 2     # DeepSeekV3: Sigmoid -> RoutingBiasAdd -> Top2 in group -> Top4 groups -> Top8 experts
                       # DeepSeekV3方式：Sigmoid->路由偏置加法->组内Top2->Top4组->Top8专家
    Llama4 = 3         # Llama4: Top1 -> Sigmoid
                       # Llama4方式：Top1后接Sigmoid
    Qwen3 = 4          # Qwen3: Softmax -> TopK -> Renormalize
                       # Qwen3方式：Softmax->TopK->重归一化
    Unspecified = 5    # Unspecified routing method
                       # 未指定的路由方法

class BaseMoeRoutingMethod(nn.Module):
    """
    Base class for MoE routing methods.
    MoE路由方法的基类。
    """
    def apply(self, router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Apply routing logic to compute expert assignments.
        应用路由逻辑计算专家分配。

        Args:
            router_logits: Raw routing logits
                         原始路由logits

        Returns:
            Tuple[Tensor, Tensor]: (expert_indices, routing_weights)
                                 (专家索引，路由权重)
        """
        raise NotImplementedError

    def get_experts_per_token(self) -> int:
        """
        Get number of experts assigned per token.
        获取每个token分配的专家数量。
        """
        raise NotImplementedError

@dataclass
class MoEWeightLoadingMode(Enum):
    """
    Weight loading modes for MoE.
    MoE的权重加载模式。
    """
    VANILLA = 0               # Standard loading mode 标准加载模式
    FUSED_GATE_UP_PROJ = 1   # Fused gate and up projection 融合门控和上投影


class DefaultMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(torch.nn.functional.softmax(
            router_logits.float(), dim=-1),
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), topk_values

    @property
    def routing_method_type(self):
        return RoutingMethodType.Default


class DeepSeekV3MoeRoutingMethod(BaseMoeRoutingMethod):

    # Intentionally leave apply() unimplemented.
    # See comments in DeepseekV3Gate on why routing is done by DeepseekV3Gate.
    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    @property
    def routing_method_type(self):
        return RoutingMethodType.DeepSeekV3


class RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(
        self,
        top_k: int,
    ):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.nn.functional.softmax(
            topk_values.float(), dim=-1)

    @property
    def routing_method_type(self):
        return RoutingMethodType.Renormalize


class Llama4RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(router_logits,
                                               k=self.top_k,
                                               dim=-1)
        return topk_indices.to(torch.int32), torch.sigmoid(topk_values.float())

    @property
    def routing_method_type(self):
        return RoutingMethodType.Llama4


# TODO: re-enable this once the custom op is working.
# class Llama4RenormalizeMoeRoutingMethod(BaseMoeRoutingMethod):

#     def __init__(self, top_k: int, num_experts_total: int, ep_size: int,
#                  ep_rank: int):
#         super().__init__()
#         self.top_k = top_k
#         self.num_experts_total = num_experts_total
#         self.num_experts_per_node = self.num_experts_total // ep_size
#         self.start_expert = self.num_experts_per_node * ep_rank
#         self.end_expert = self.start_expert + self.num_experts_per_node

#     def apply(self,
#               router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         unpermuted_scales, indices = torch.ops.trtllm.fused_topk_softmax(
#             router_logits, self.top_k, self.num_experts_total,
#             self.start_expert, self.end_expert)
#         return indices, unpermuted_scales


# TODO Test this for Phi models
class SparseMixerMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int, eps: float):
        super().__init__()
        self.top_k = top_k
        self.eps = eps

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        router_logits = router_logits.float()
        topk_values = torch.empty(router_logits.shape[0],
                                  self.top_k,
                                  device=router_logits.device,
                                  dtype=torch.float32)
        topk_indices = torch.empty(router_logits.shape[0],
                                   self.top_k,
                                   device=router_logits.device,
                                   dtype=torch.int32)
        for i in range(self.top_k):
            if i > 0:
                max_elem = torch.argmax(router_logits, dim=-1)
                # Mask out the previously selected indices to negative infinity
                router_logits.scatter_(-1, max_elem.unsqueeze(-1),
                                       -float('inf'))
            # Get the max value of the remaining indices
            max_values, max_indices = torch.max(router_logits,
                                                dim=-1,
                                                keepdim=True)
            assert torch.all(max_values != -float('inf'))

            topk_indices[:, i] = max_indices.squeeze(-1)

            # Mask out any values that fail the condition '(max - value) / std::max(abs(value), max) > 2 * epsilon'
            mask = (
                (max_values - router_logits) /
                torch.max(torch.abs(router_logits), max_values)) > 2 * self.eps
            masked_logits = torch.where(mask, -float('inf'), router_logits)
            softmax_masked_logits = torch.nn.functional.softmax(masked_logits,
                                                                dim=-1)
            selected_values = torch.gather(softmax_masked_logits, -1,
                                           max_indices)
            topk_values[:, i] = selected_values.squeeze(-1)

        return topk_indices.to(torch.int32), topk_values


class StaticMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(
        self,
                 routing_tensor: torch.Tensor,
        routing_scales: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert routing_tensor.dtype == torch.int32
        if routing_scales is not None:
            assert routing_tensor.shape[0] == routing_scales.shape[0]
            assert routing_tensor.shape[1] == routing_scales.shape[1]
            assert routing_scales.dtype == torch.float32
        self.routing_tensor = routing_tensor
        self.routing_scales = routing_scales

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self.routing_tensor, self.routing_scales

    def get_experts_per_token(self):
        return self.routing_tensor.shape[1]


class LoadBalancedMoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k


class Qwen3MoeRoutingMethod(BaseMoeRoutingMethod):

    def __init__(self, top_k: int):
        super().__init__()
        self.top_k = top_k

    def apply(self,
              router_logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        topk_values, topk_indices = torch.topk(torch.nn.functional.softmax(
            router_logits.float(), dim=-1),
                                               k=self.top_k,
                                               dim=-1)
        # Note: We do not renormalize in Qwen3, instead we leave the softmax output as it is.
        return topk_indices.to(torch.int32), topk_values

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Qwen3


class StreamManager:
    """优化的CUDA流管理器"""
    
    compute: torch.cuda.Stream  # 计算流
    prefetch: torch.cuda.Stream # 预取流
    transfer: torch.cuda.Stream # 传输流
    
    @staticmethod
    def create(device: torch.device):
        return StreamManager(
            compute=torch.cuda.Stream(device=device, priority=0),    # 高优先级
            prefetch=torch.cuda.Stream(device=device, priority=-1),  # 中优先级  
            transfer=torch.cuda.Stream(device=device, priority=-2)   # 低优先级
        )
    
    def synchronize(self):
        """同步所有流"""
        for stream in [self.compute, self.prefetch, self.transfer]:
            stream.synchronize()

class OptimizedFusedMoE(torch.nn.Module):
    """优化的MoE实现"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda")
        self.stream_mgr = StreamManager.create(self.device)
        
        # 性能优化配置
        self.perf_config = {
            'max_active_experts': 32,
            'prefetch_lookahead': 3,
            'min_batch_size': 1,
            'max_batch_size': 32,
            'shared_memory_size': 48*1024
        }
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        
        with torch.cuda.stream(self.stream_mgr.prefetch):
            # 预取下一批专家权重
            next_experts = self._predict_next_experts(hidden_states)
            self._prefetch_expert_weights(next_experts)
        
        with torch.cuda.stream(self.stream_mgr.compute):
            # 优化的专家计算
            expert_outputs = []
            for i in range(0, batch_size, self.perf_config['max_batch_size']):
                # 获取当前批次
                batch = hidden_states[i:i + self.perf_config['max_batch_size']]
                
                # 计算专家路由
                router_logits = self.compute_router_logits(batch)
                expert_indices = self.route_to_experts(router_logits)
                
                # 执行专家计算
                output = self._parallel_expert_compute(
                    batch,
                    expert_indices,
                    shared_memory_size=self.perf_config['shared_memory_size']
                )
                expert_outputs.append(output)
                
        with torch.cuda.stream(self.stream_mgr.transfer):
            # 传输结果
            output = torch.cat(expert_outputs, dim=0)
            
        # 在返回前同步所有流
        self.stream_mgr.synchronize()
        
        return output
        
    def _predict_next_experts(self, hidden_states: torch.Tensor) -> List[int]:
        """预测下一批次可能需要的专家"""
        with torch.no_grad():
            logits = self.compute_router_logits(hidden_states)
            probs = torch.sigmoid(logits)
            # 选择概率大于阈值的专家
            active_experts = (probs > self.config.expert_threshold).nonzero()
            return active_experts.tolist()
    
    def _prefetch_expert_weights(self, expert_indices: List[int]):
        """预取专家权重到GPU"""
        for expert_id in expert_indices[:self.perf_config['max_active_experts']]:
            if expert_id not in self.experts:
                self.load_expert(expert_id, non_blocking=True)
                
    def _parallel_expert_compute(
        self,
        inputs: torch.Tensor,
        expert_indices: torch.Tensor,
        shared_memory_size: int
    ) -> torch.Tensor:
        """并行执行专家计算"""
        num_experts = len(expert_indices)
        
        # 获取设备信息用于优化
        device_props = torch.cuda.get_device_properties(self.device)
        
        # 计算最优的并行度
        threads_per_block = min(256, device_props.max_threads_per_multi_processor // 2)
        blocks_per_sm = max(4, device_props.max_threads_per_multi_processor // threads_per_block)
        num_blocks = device_props.multi_processor_count * blocks_per_sm
        
        # 分配共享内存
        shared_memory = torch.cuda.SharedMemory(size=shared_memory_size)
        
        try:
            # 并行执行专家计算
            outputs = []
            for i in range(0, num_experts, blocks_per_sm):
                expert_batch = expert_indices[i:i + blocks_per_sm]
                
                # 启动优化后的CUDA核函数
                output = torch.ops.deepseek_fp4.parallel_expert_compute(
                    inputs,
                    expert_batch,
                    self.experts,
                    blocks=num_blocks,
                    threads=threads_per_block,
                    shared_memory=shared_memory
                )
                outputs.append(output)
            
            # 合并所有输出
            return torch.cat(outputs, dim=0)
            
        finally:
            # 释放共享内存
            shared_memory.close()