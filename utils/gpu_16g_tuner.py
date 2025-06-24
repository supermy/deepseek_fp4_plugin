import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from .perf_monitor import PerformanceMonitor
from ..configs.gpu_16g_perf import GPU16GConfig

@dataclass
class TuningResult:
    """调优结果"""
    config: Dict
    metrics: Dict
    score: float
    
class GPU16GAutoTuner:
    """16GB GPU自动调优器"""
    
    def __init__(self):
        self.config = GPU16GConfig()
        self.monitor = PerformanceMonitor()
        self.best_result = None
        
        # 调优参数范围
        self.tuning_ranges = {
            'batch_size': [1, 2, 4, 8, 16],
            'max_gpu_layers': [2, 3, 4, 5],
            'prefetch_layers': [1, 2, 3],
            'num_compute_streams': [1, 2]
        }
        
        # 参数权重
        self.weights = {
            'gpu_util': 0.5,      # GPU利用率权重
            'memory_util': 0.3,    # 显存利用率权重
            'latency': 0.2        # 延迟权重
        }
        
    def _evaluate_config(self, config: Dict) -> TuningResult:
        """评估一个配置的性能"""
        # 应用配置
        self.config = GPU16GConfig()
        for k, v in config.items():
            setattr(self.config, k, v)
        self.config.enable_optimization()
        
        # 收集性能指标
        metrics = {}
        for _ in range(10):  # 运行10次取平均
            self.monitor.update()
            current_metrics = self.monitor.get_average_metrics(window=5)
            for k, v in current_metrics.__dict__.items():
                metrics[k] = metrics.get(k, 0.0) + v / 10
                
        # 计算综合得分
        score = (
            self.weights['gpu_util'] * metrics['compute_util'] +
            self.weights['memory_util'] * metrics['memory_util'] -
            self.weights['latency'] * (1.0 - metrics['compute_util'])  # 利用率越高延迟分数越好
        )
        
        return TuningResult(config=config, metrics=metrics, score=score)
        
    def _search_best_config(self, num_trials: int = 30) -> TuningResult:
        """搜索最优配置"""
        results = []
        
        # 网格搜索
        for batch_size in self.tuning_ranges['batch_size']:
            for max_layers in self.tuning_ranges['max_gpu_layers']:
                for prefetch in self.tuning_ranges['prefetch_layers']:
                    for compute_streams in self.tuning_ranges['num_compute_streams']:
                        if len(results) >= num_trials:
                            break
                            
                        # 构建配置
                        config = {
                            'batch_size': batch_size,
                            'max_gpu_layers': max_layers,
                            'prefetch_layers': prefetch,
                            'num_compute_streams': compute_streams
                        }
                        
                        # 评估配置
                        result = self._evaluate_config(config)
                        results.append(result)
                        
                        # 更新最优结果
                        if (not self.best_result or 
                            result.score > self.best_result.score):
                            self.best_result = result
                            
        return self.best_result
        
    def optimize(self, 
                time_budget: Optional[float] = 120.0,
                target_gpu_util: float = 0.85) -> Tuple[Dict, Dict]:
        """执行自动调优
        
        Args:
            time_budget: 调优时间预算(秒)
            target_gpu_util: 目标GPU利用率
            
        Returns:
            Tuple[最优配置, 性能指标]
        """
        print(f"\n开始16GB GPU自动调优 (目标利用率: {target_gpu_util*100:.1f}%)")
        
        # 设置目标
        self.monitor.targets['gpu_util'] = target_gpu_util
        
        # 执行搜索
        best_result = self._search_best_config()
        
        print("\n调优完成!")
        print(f"最优配置:")
        for k, v in best_result.config.items():
            print(f"  {k}: {v}")
        print(f"\n性能指标:")
        for k, v in best_result.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v*100:.1f}%")
                
        return best_result.config, best_result.metrics
        
    def apply_best_config(self):
        """应用最优配置"""
        if self.best_result:
            config = GPU16GConfig()
            for k, v in self.best_result.config.items():
                setattr(config, k, v)
            config.enable_optimization()
            return config
        return None