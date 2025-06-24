import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.cuda as cuda

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    gpu_util: float = 0.0
    memory_util: float = 0.0
    pcie_util: float = 0.0
    total_latency: float = 0.0
    compute_latency: float = 0.0
    transfer_latency: float = 0.0
    memory_allocated: float = 0.0
    memory_reserved: float = 0.0
    throughput: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'gpu_util': self.gpu_util,
            'memory_util': self.memory_util,
            'pcie_util': self.pcie_util,
            'total_latency': self.total_latency,
            'compute_latency': self.compute_latency,
            'transfer_latency': self.transfer_latency,
            'memory_allocated': self.memory_allocated,
            'memory_reserved': self.memory_reserved,
            'throughput': self.throughput
        }

class MetricsAnalyzer:
    """性能指标分析器"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.bottleneck_thresholds = {
            'gpu_util': 0.85,
            'memory_util': 0.90,
            'pcie_util': 0.80,
            'latency': 100.0  # ms
        }
        
    def add_metrics(self, metrics: PerformanceMetrics):
        """添加新的性能指标"""
        self.metrics_history.append(metrics)
        
    def get_average_metrics(self, window: int = 10) -> PerformanceMetrics:
        """计算平均性能指标"""
        if not self.metrics_history:
            return PerformanceMetrics()
            
        # 获取最近window个样本
        recent = self.metrics_history[-window:]
        
        avg_metrics = PerformanceMetrics()
        for key in avg_metrics.to_dict().keys():
            values = [getattr(m, key) for m in recent]
            setattr(avg_metrics, key, sum(values) / len(values))
            
        return avg_metrics
        
    def detect_bottlenecks(self) -> List[Dict]:
        """检测性能瓶颈"""
        if not self.metrics_history:
            return []
            
        bottlenecks = []
        latest = self.metrics_history[-1]
        
        # 检查GPU利用率
        if latest.gpu_util < self.bottleneck_thresholds['gpu_util']:
            bottlenecks.append({
                'type': 'GPU Utilization',
                'current': f'{latest.gpu_util*100:.1f}%',
                'target': f'{self.bottleneck_thresholds["gpu_util"]*100:.1f}%'
            })
            
        # 检查内存利用率
        if latest.memory_util > self.bottleneck_thresholds['memory_util']:
            bottlenecks.append({
                'type': 'Memory Utilization',
                'current': f'{latest.memory_util*100:.1f}%',
                'target': f'<{self.bottleneck_thresholds["memory_util"]*100:.1f}%'
            })
            
        # 检查PCIe带宽利用率
        if latest.pcie_util < self.bottleneck_thresholds['pcie_util']:
            bottlenecks.append({
                'type': 'PCIe Bandwidth',
                'current': f'{latest.pcie_util*100:.1f}%',
                'target': f'{self.bottleneck_thresholds["pcie_util"]*100:.1f}%'
            })
            
        # 检查延迟
        if latest.total_latency > self.bottleneck_thresholds['latency']:
            bottlenecks.append({
                'type': 'Latency',
                'current': f'{latest.total_latency:.1f}ms',
                'target': f'<{self.bottleneck_thresholds["latency"]:.1f}ms'
            })
            
        return bottlenecks
        
    def get_performance_summary(self) -> Dict:
        """生成性能总结"""
        if not self.metrics_history:
            return {}
            
        metrics = np.array([m.to_dict() for m in self.metrics_history])
        
        summary = {
            'averages': {},
            'min_max': {},
            'percentiles': {},
            'stability': {}
        }
        
        # 计算平均值
        for key in metrics[0].keys():
            values = metrics[key]
            summary['averages'][key] = float(np.mean(values))
            summary['min_max'][key] = {
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            summary['percentiles'][key] = {
                'p50': float(np.percentile(values, 50)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }
            summary['stability'][key] = float(np.std(values))
            
        return summary
        
    def suggest_optimizations(self) -> List[str]:
        """提供优化建议"""
        if not self.metrics_history:
            return []
            
        suggestions = []
        avg_metrics = self.get_average_metrics(window=10)
        
        # GPU利用率优化建议
        if avg_metrics.gpu_util < 0.7:
            suggestions.append(
                "GPU利用率较低,建议:\n"
                "1. 增加批处理大小\n"
                "2. 增加GPU常驻层数\n"
                "3. 启用TF32计算"
            )
            
        # 内存优化建议
        if avg_metrics.memory_util > 0.95:
            suggestions.append(
                "内存压力较大,建议:\n"
                "1. 减少批处理大小\n"
                "2. 启用内存池管理\n"
                "3. 优化显存分配策略"
            )
            
        # PCIe优化建议
        if avg_metrics.pcie_util < 0.6:
            suggestions.append(
                "PCIe带宽利用率较低,建议:\n"
                "1. 增加传输缓冲区大小\n"
                "2. 启用数据压缩\n"
                "3. 优化预取策略"
            )
            
        # 延迟优化建议
        if avg_metrics.total_latency > 150:
            suggestions.append(
                "延迟较高,建议:\n"
                "1. 减小批处理大小\n"
                "2. 使用固定内存\n"
                "3. 启用CUDA Graph"
            )
            
        return suggestions
        
    def export_metrics(self, file_path: str):
        """导出性能指标数据"""
        data = {
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'summary': self.get_performance_summary(),
            'bottlenecks': self.detect_bottlenecks(),
            'suggestions': self.suggest_optimizations()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
class PerformanceMonitor:
    """实时性能监控器"""
    
    def __init__(self):
        self.metrics = []
        self.start_time = time.time()
        self.last_update = self.start_time
        
    def update(self):
        """更新性能指标"""
        current_metrics = PerformanceMetrics()
        
        # 获取GPU利用率
        current_metrics.gpu_util = self._get_gpu_utilization()
        
        # 获取内存利用率
        current_metrics.memory_util = self._get_memory_utilization()
        
        # 获取延迟指标
        current_metrics.total_latency = time.time() - self.last_update
        
        # 更新时间戳
        self.last_update = time.time()
        
        # 保存指标
        self.metrics.append(current_metrics)
        
    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率"""
        try:
            if not cuda.is_available():
                return 0.0
            # 使用nvml获取GPU利用率
            return cuda.utilization() / 100.0
        except:
            return 0.0
            
    def _get_memory_utilization(self) -> float:
        """获取显存利用率"""
        try:
            if not cuda.is_available():
                return 0.0
            # 获取当前显存使用量
            allocated = cuda.memory_allocated()
            reserved = cuda.memory_reserved()
            total = cuda.get_device_properties(0).total_memory
            return allocated / total
        except:
            return 0.0
            
    def get_average_metrics(self, window: int = 10) -> PerformanceMetrics:
        """获取平均性能指标"""
        if not self.metrics:
            return PerformanceMetrics()
            
        recent = self.metrics[-window:]
        avg_metrics = PerformanceMetrics()
        
        for key in avg_metrics.to_dict().keys():
            values = [getattr(m, key) for m in recent]
            setattr(avg_metrics, key, sum(values) / len(values))
            
        return avg_metrics
        
    def reset(self):
        """重置监控器"""
        self.metrics = []
        self.start_time = time.time()
        self.last_update = self.start_time