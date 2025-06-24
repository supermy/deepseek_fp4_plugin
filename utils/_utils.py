import contextlib
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch

is_torch_compiling_flag = False

aux_stream_name_list = ['Attention', 'MoeShared', 'MoeChunkingOverlap']
AuxStreamType = Enum(
    'AuxStreamType',
    aux_stream_name_list,
)
EventType = Enum(
    'EventType',
    ['Main', *aux_stream_name_list],
    start=0,
)


def set_torch_compiling(enable: bool):
    global is_torch_compiling_flag
    is_torch_compiling_flag = enable


def is_torch_compiling() -> bool:
    global is_torch_compiling_flag
    return is_torch_compiling_flag


_global_attrs = threading.local()


def get_global_attrs():
    return _global_attrs


_model_extra_attrs = threading.local()


def get_model_extra_attrs():
    return getattr(_model_extra_attrs, 'attrs', None)


@contextlib.contextmanager
def model_extra_attrs(attrs: Dict):
    old_attrs = getattr(_model_extra_attrs, 'attrs', None)
    _model_extra_attrs.attrs = attrs
    try:
        yield
    finally:
        _model_extra_attrs.attrs = old_attrs


def with_model_extra_attrs(get_attrs):

    def decorator(func):

        def wrapper(self, *args, **kwargs):
            with model_extra_attrs(get_attrs(self)):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


def make_weak_ref(x):

    if isinstance(x, torch.Tensor):
        # For standalone plugin, direct tensor reference is sufficient
        # if x.is_cuda is not strictly necessary for weak ref behavior.
        # Original TensorRT-LLM used TensorWrapper for complex memory management.
        return x
    elif isinstance(x, tuple):
        return tuple(make_weak_ref(i) for i in x)
    elif isinstance(x, list):
        return [make_weak_ref(i) for i in x]
    elif isinstance(x, dict):
        return {k: make_weak_ref(v) for k, v in x.items()}
    elif isinstance(x, (int, float, bool)):
        return x
    else:
        raise TypeError(f"Invalid type {type(x)} to make weak ref")


@dataclass
class Fp4QuantizedTensor:
    fp4_tensor: torch.Tensor
    scaling_factor: torch.Tensor

    @property
    def shape(self):
        return self.fp4_tensor.shape


_disable_fp4_allgather = os.getenv("TLLM_DISABLE_FP4_ALLGATHER", "0") == "1"


def disable_fp4_allgather():
    return _disable_fp4_allgather


def swizzle_sf(sf: torch.Tensor,
               row: int,
               col: int,
               scaling_vector_size: int = 16):
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_full = torch.zeros(num_m_tiles * 32 * 4,
                          num_k_tiles * 4,
                          dtype=sf.dtype,
                          device=sf.device)
    sf_full[:row, :(col //
                    scaling_vector_size)] = sf[:row, :(col //
                                                       scaling_vector_size)]
    sf_full_reshaped = sf_full.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_full_swizzle = sf_full_reshaped.transpose(1, 3)
    sf_swizzle = sf_full_swizzle.reshape(-1)
    return sf_swizzle


def unswizzle_sf(sf: torch.Tensor,
                 row: int,
                 col: int,
                 scaling_vector_size: int = 16):
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    sf_unswizzle_sliced = sf_unswizzle[:row, :(col // scaling_vector_size)]
    return sf_unswizzle_sliced.contiguous()


def reswizzle_sf(sf: torch.Tensor,
                 row: int,
                 col: int,
                 scaling_vector_size: int = 16):
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    partition_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
    num_partitions = sf.numel() // partition_size
    sf_reshaped = sf.view(num_partitions, num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(2, 4)
    sf_unswizzle = sf_unswizzle.reshape(num_partitions, num_m_tiles * 32 * 4,
                                        num_k_tiles * 4)
    total_rows = num_partitions * row
    num_m_tiles_out = (total_rows + 128 - 1) // 128
    sf_out = torch.zeros(
        num_m_tiles_out,
        4,
        32,
        num_k_tiles,
        4,
        dtype=sf.dtype,
        device=sf.device,
    )
    sf_out_reshaped = sf_out.view(num_m_tiles_out * 32 * 4, num_k_tiles * 4)
    sf_out_reshaped[:total_rows] = sf_unswizzle[:, :row].reshape(total_rows, -1)
    sf_out_swizzle = sf_out.transpose(1, 3).reshape(-1)
    return sf_out_swizzle


def next_positive_power_of_2(x: int) -> int:
    if x < 1:
        return 1

    return 1 << (x - 1).bit_length()


def last_positive_power_of_2(x: int) -> int:
    next = next_positive_power_of_2(x)
    if next == x:
        return next

    return next // 2


def nearest_in_buckets(x: int, buckets: List[int]) -> int:
    return min(max(next_positive_power_of_2(x), buckets[0]), buckets[-1])


def get_power_of_2_num_tokens_buckets(max_num_tokens) -> List[int]:
    max_num_tokens = next_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= 1:
        num_token_buckets.append(m)
        m //= 2

    return tuple(num_token_buckets)


def get_last_power_of_2_num_tokens_buckets(max_num_tokens) -> List[int]:
    max_num_tokens = last_positive_power_of_2(max_num_tokens)
    num_token_buckets = []
    m = max_num_tokens
    while m >= 1:
        num_token_buckets.append(m)
        m //= 2
    return num_token_buckets


_enable_piecewise_cuda_graph = True


def set_piecewise_cuda_graph_flag(enable: bool):
    global _enable_piecewise_cuda_graph
    _enable_piecewise_cuda_graph = enable


def get_piecewise_cuda_graph_flag() -> bool:
    global _enable_piecewise_cuda_graph
    return _enable_piecewise_cuda_graph

def get_sm_version() -> int:
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
    return 0

import torch
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PerfMetrics:
    """性能指标收集器"""
    gpu_util: float = 0.0
    mem_util: float = 0.0
    compute_time: float = 0.0
    transfer_time: float = 0.0
    prefetch_hits: int = 0
    total_requests: int = 0
    
class PerfMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = PerfMetrics()
        self.start_time = time.time()
        
    def update_metrics(self, new_metrics: Dict[str, float]):
        """更新性能指标"""
        for k, v in new_metrics.items():
            if hasattr(self.metrics, k):
                setattr(self.metrics, k, v)
                
    def get_utilization(self) -> Dict[str, float]:
        """获取资源利用率"""
        return {
            'gpu': self.metrics.gpu_util,
            'memory': self.metrics.mem_util,
            'prefetch_hit_rate': self.metrics.prefetch_hits / max(1, self.metrics.total_requests)
        }

class AutoTuner:
    """自动性能调优器"""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.perf_monitor = PerfMonitor()
        self.tuning_ranges = {
            'max_gpu_layers': (2, 8),
            'prefetch_layers': (1, 4),
            'expert_threshold': (0.05, 0.2),
            'batch_size': (1, 32)
        }
        
    def optimize(self, target_metric: str = 'gpu') -> Dict[str, any]:
        """执行自动调优"""
        best_config = self.config.copy()
        best_util = 0.0
        
        # 网格搜索最优配置
        for max_layers in range(*self.tuning_ranges['max_gpu_layers']):
            for prefetch in range(*self.tuning_ranges['prefetch_layers']):
                for threshold in [0.05, 0.1, 0.15, 0.2]:
                    for batch_size in [1, 2, 4, 8, 16, 32]:
                        # 尝试新配置
                        test_config = {
                            'max_gpu_layers': max_layers,
                            'prefetch_layers': prefetch,
                            'expert_threshold': threshold,
                            'batch_size': batch_size
                        }
                        
                        # 运行一次推理收集指标
                        self._run_inference(test_config)
                        util = self.perf_monitor.get_utilization()[target_metric]
                        
                        # 更新最优配置
                        if util > best_util:
                            best_util = util
                            best_config.update(test_config)
                            
        return best_config
    
    def _run_inference(self, config: Dict[str, any]):
        """使用给定配置运行推理并收集指标"""
        # 设置当前配置
        self.config.update(config)
        
        # 运行推理并收集指标
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        # 收集GPU利用率
        gpu_util = torch.cuda.utilization()
        mem_util = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        
        # 更新性能指标
        self.perf_monitor.update_metrics({
            'gpu_util': gpu_util,
            'mem_util': mem_util,
            'compute_time': end.elapsed_time(start),
            'total_requests': self.perf_monitor.metrics.total_requests + 1
        })
        
class CudaProfiler:
    """CUDA性能分析器"""
    
    def __init__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        )
    
    def __enter__(self):
        self.profiler.start()
        return self
        
    def __exit__(self, *args):
        self.profiler.stop()
        
    def step(self):
        """记录一个性能分析步骤"""
        self.profiler.step()
        
def optimize_single_task():
    """优化单任务性能"""
    # 创建调优器
    tuner = AutoTuner({
        'max_gpu_layers': 4,
        'prefetch_layers': 2,
        'expert_threshold': 0.1,
        'batch_size': 16
    })
    
    # 运行自动调优
    with CudaProfiler() as profiler:
        optimal_config = tuner.optimize(target_metric='gpu')
        profiler.step()
        
    # 获取性能指标
    metrics = tuner.perf_monitor.get_utilization()
    
    return optimal_config, metrics
