import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class MemoryConfig:
    """内存配置"""
    gpu_memory: int = 16 * 1024 * 1024 * 1024  # 16GB显存
    cpu_memory: int = 96 * 1024 * 1024 * 1024  # 96GB系统内存
    page_size: int = 2 * 1024 * 1024  # 2MB页大小
    prefetch_size: int = 64 * 1024 * 1024  # 64MB预取大小
    
class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        self.config = MemoryConfig()
        self.gpu_allocated = 0  # 已分配显存
        self.cpu_allocated = 0  # 已分配内存
        self.page_table = {}    # 页表
        self.access_history = deque(maxlen=1000)  # 访问历史
        
    def get_memory_info(self) -> Dict[str, float]:
        """获取内存使用情况"""
        gpu_free = self.config.gpu_memory - self.gpu_allocated
        cpu_free = self.config.cpu_memory - self.cpu_allocated
        
        return {
            'gpu_total': self.config.gpu_memory,
            'gpu_used': self.gpu_allocated,
            'gpu_free': gpu_free,
            'gpu_util': self.gpu_allocated / self.config.gpu_memory,
            'cpu_total': self.config.cpu_memory,
            'cpu_used': self.cpu_allocated,
            'cpu_free': cpu_free,
            'cpu_util': self.cpu_allocated / self.config.cpu_memory
        }
        
    def should_offload(self, tensor_size: int) -> bool:
        """判断是否应该卸载到CPU"""
        gpu_free = self.config.gpu_memory - self.gpu_allocated
        return tensor_size > gpu_free * 0.1  # 如果张量大小超过剩余显存10%
        
    def should_prefetch(self, tensor_id: str) -> bool:
        """判断是否应该预取到GPU"""
        if tensor_id not in self.access_history:
            return False
            
        # 分析访问模式
        access_count = sum(1 for x in self.access_history if x == tensor_id)
        return access_count >= 3  # 如果最近访问次数>=3
        
    def allocate(self, size: int, gpu: bool = True) -> Optional[int]:
        """分配内存"""
        if gpu:
            if size > self.config.gpu_memory - self.gpu_allocated:
                return None
            self.gpu_allocated += size
            return size
        else:
            if size > self.config.cpu_memory - self.cpu_allocated:
                return None
            self.cpu_allocated += size
            return size
            
    def free(self, size: int, gpu: bool = True):
        """释放内存"""
        if gpu:
            self.gpu_allocated = max(0, self.gpu_allocated - size)
        else:
            self.cpu_allocated = max(0, self.cpu_allocated - size)
            
    def optimize_memory_usage(self, 
                            model_size: int,
                            batch_size: int,
                            seq_length: int) -> Dict:
        """优化内存使用"""
        # 计算关键尺寸
        hidden_size = 4096  # 假设hidden_size=4096
        activation_size = batch_size * seq_length * hidden_size * 4  # float32
        weight_size = model_size
        
        # 计算最优分配
        gpu_weights = min(weight_size, self.config.gpu_memory * 0.6)  # 最多使用60%显存放权重
        gpu_activations = min(activation_size, self.config.gpu_memory * 0.3)  # 30%放激活值
        gpu_workspace = self.config.gpu_memory * 0.1  # 10%作为工作空间
        
        # 剩余放在CPU内存
        cpu_weights = weight_size - gpu_weights
        cpu_activations = activation_size - gpu_activations
        
        return {
            'gpu': {
                'weights': gpu_weights,
                'activations': gpu_activations,
                'workspace': gpu_workspace
            },
            'cpu': {
                'weights': cpu_weights,
                'activations': cpu_activations,
                'prefetch_buffer': self.config.prefetch_size
            }
        }
        
    def create_memory_plan(self, 
                          total_size: int,
                          batch_size: int,
                          seq_length: int) -> Dict:
        """创建内存规划"""
        # 优化内存使用
        memory_usage = self.optimize_memory_usage(
            total_size, batch_size, seq_length
        )
        
        # 计算分页
        pages_gpu = memory_usage['gpu']['weights'] // self.config.page_size
        pages_cpu = memory_usage['cpu']['weights'] // self.config.page_size
        
        # 创建页表
        page_table = {
            'gpu': list(range(int(pages_gpu))),
            'cpu': list(range(int(pages_gpu), int(pages_gpu + pages_cpu)))
        }
        
        # 预取策略
        prefetch_strategy = {
            'window_size': 3,  # 预取窗口大小
            'threshold': 0.8,  # 预取阈值
            'batch_size': memory_usage['cpu']['prefetch_buffer']  # 预取批大小
        }
        
        return {
            'memory_usage': memory_usage,
            'page_table': page_table,
            'prefetch_strategy': prefetch_strategy
        }
        
    def monitor_memory_pressure(self) -> Tuple[float, float]:
        """监控内存压力"""
        gpu_pressure = self.gpu_allocated / self.config.gpu_memory
        cpu_pressure = self.cpu_allocated / self.config.cpu_memory
        return gpu_pressure, cpu_pressure
        
    def suggest_batch_size(self, 
                          base_batch_size: int,
                          model_size: int) -> int:
        """建议批大小"""
        gpu_mem_info = self.get_memory_info()
        gpu_free = gpu_mem_info['gpu_free']
        
        # 估算每个样本的显存使用
        mem_per_sample = model_size / base_batch_size
        
        # 计算安全的批大小
        safe_batch_size = int(gpu_free * 0.8 / mem_per_sample)  # 留20%余量
        
        # 向下取整到2的幂
        return 2 ** int(np.log2(safe_batch_size))