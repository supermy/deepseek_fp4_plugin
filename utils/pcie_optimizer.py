import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class PCIeConfig:
    """PCIe传输配置"""
    buffer_size: int = 16 * 1024 * 1024  # 16MB传输缓冲区
    compression_enabled: bool = True      # 启用压缩传输
    overlap_compute: bool = True         # 计算传输重叠
    num_channels: int = 2               # 传输通道数
    max_bandwidth: float = 14.0         # 最大带宽(GB/s)
    
class PCIeOptimizer:
    """PCIe传输优化器"""
    
    def __init__(self):
        self.config = PCIeConfig()
        self.transfer_history = []  # 记录传输历史
        self.compute_overlap = []   # 记录计算重叠
        
    def optimize_buffer_size(self, data_size: int) -> int:
        """优化传输缓冲区大小"""
        # 根据数据大小调整缓冲区
        if data_size < 1024 * 1024:  # 1MB以下
            return 1024 * 1024  # 使用1MB缓冲区
        elif data_size < 16 * 1024 * 1024:  # 16MB以下
            return 4 * 1024 * 1024  # 使用4MB缓冲区
        else:
            return 16 * 1024 * 1024  # 使用16MB缓冲区
            
    def estimate_transfer_time(self, data_size: int) -> float:
        """估算传输时间(ms)"""
        compression_ratio = 0.6 if self.config.compression_enabled else 1.0
        effective_size = data_size * compression_ratio
        return (effective_size / (self.config.max_bandwidth * 1e9)) * 1000
        
    def should_compress(self, data_size: int, compute_time: float) -> bool:
        """判断是否应该启用压缩"""
        # 如果计算时间足够长,值得压缩
        transfer_time = self.estimate_transfer_time(data_size)
        compression_overhead = data_size * 0.0001  # 假设压缩开销
        return transfer_time > compression_overhead
        
    def optimize_channels(self, active_transfers: int) -> int:
        """优化传输通道数"""
        # 根据活跃传输数调整通道数
        if active_transfers <= 2:
            return 1
        elif active_transfers <= 4:
            return 2
        else:
            return 3  # 最多使用3个通道
            
    def schedule_transfer(self, 
                         data_size: int,
                         compute_time: Optional[float] = None) -> Dict:
        """调度数据传输"""
        # 优化传输参数
        buffer_size = self.optimize_buffer_size(data_size)
        num_channels = self.optimize_channels(len(self.transfer_history))
        use_compression = self.should_compress(data_size, compute_time or 0)
        
        # 构建传输配置
        transfer_config = {
            'buffer_size': buffer_size,
            'num_channels': num_channels,
            'use_compression': use_compression,
            'estimated_time': self.estimate_transfer_time(data_size)
        }
        
        # 记录传输
        self.transfer_history.append({
            'size': data_size,
            'config': transfer_config
        })
        
        return transfer_config
        
    def start_compute_overlap(self):
        """开始计算重叠区间"""
        self.compute_overlap.append({
            'start_time': torch.cuda.Event(enable_timing=True),
            'end_time': None
        })
        self.compute_overlap[-1]['start_time'].record()
        
    def end_compute_overlap(self):
        """结束计算重叠区间"""
        if self.compute_overlap:
            self.compute_overlap[-1]['end_time'] = torch.cuda.Event(enable_timing=True)
            self.compute_overlap[-1]['end_time'].record()
            
    def get_overlap_ratio(self) -> float:
        """获取计算传输重叠比例"""
        if not self.compute_overlap:
            return 0.0
            
        total_time = 0
        overlap_time = 0
        
        for interval in self.compute_overlap:
            if interval['end_time']:
                interval['start_time'].synchronize()
                interval['end_time'].synchronize()
                duration = interval['start_time'].elapsed_time(interval['end_time'])
                total_time += duration
                
                # 检查是否有传输重叠
                for transfer in self.transfer_history:
                    if transfer['config']['estimated_time'] < duration:
                        overlap_time += transfer['config']['estimated_time']
                        
        return overlap_time / max(total_time, 1e-6)
        
    def create_events(self) -> Dict:
        """创建CUDA事件用于同步"""
        return {
            'start': torch.cuda.Event(enable_timing=True),
            'end': torch.cuda.Event(enable_timing=True)
        }
        
    def optimize_pipeline(self, 
                         batch_size: int,
                         hidden_size: int,
                         num_layers: int) -> Dict:
        """优化整个传输流水线"""
        # 计算每层参数大小
        params_per_layer = hidden_size * hidden_size * 4  # 假设4倍膨胀
        total_params = params_per_layer * num_layers
        
        # 估算计算时间
        compute_time_per_layer = (hidden_size * hidden_size * batch_size) * 1e-9  # 估算
        
        # 优化流水线配置
        pipeline_config = {
            'num_stages': min(3, num_layers),  # 最多3级流水线
            'batch_size': batch_size,
            'params_per_transfer': self.optimize_buffer_size(params_per_layer),
            'use_compression': self.should_compress(total_params, compute_time_per_layer),
            'num_channels': self.optimize_channels(num_layers),
            'overlap_enabled': True
        }
        
        return pipeline_config