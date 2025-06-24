import torch
import time
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
from .gpu_16g_tuner import GPU16GAutoTuner
from .memory_optimizer import MemoryOptimizer
from .pcie_optimizer import PCIeOptimizer
from .perf_monitor import PerformanceMonitor, MetricsAnalyzer

@dataclass
class IterationConfig:
    batch_size: int
    num_layers: int
    memory_fraction: float
    pcie_buffer_size: int
    optimization_flags: Dict[str, bool]

class PerformanceCoordinator:
    """性能优化协调器"""
    
    def __init__(self, config_path: str):
        self.gpu_tuner = GPU16GAutoTuner()
        self.memory_optimizer = MemoryOptimizer()
        self.pcie_optimizer = PCIeOptimizer()
        self.monitor = PerformanceMonitor()
        self.analyzer = MetricsAnalyzer()
        
        self.current_config = self._load_config(config_path)
        self.best_config = None
        self.best_throughput = 0.0
        self.iteration_count = 0
        self.max_iterations = 50
        
        # 协调目标
        self.targets = {
            'gpu_util': 0.85,       # GPU利用率目标
            'memory_util': 0.90,    # 显存利用率目标
            'pcie_util': 0.70,      # PCIe带宽利用率目标
            'latency_ms': 100.0     # 延迟目标(ms)
        }
        
        # 优化状态
        self.current_state = {
            'batch_size': 1,
            'gpu_layers': 4,
            'prefetch_size': 64 * 1024 * 1024,  # 64MB
            'compression_enabled': True
        }
        
    def _load_config(self, config_path: str) -> IterationConfig:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return IterationConfig(**config_data)
    
    def optimize_all(self, model_config: Dict) -> Dict:
        """执行全面优化"""
        print("\n=== 开始全面性能优化 ===")
        
        # 1. 优化GPU配置
        gpu_config, gpu_metrics = self.gpu_tuner.optimize(
            target_gpu_util=self.targets['gpu_util']
        )
        print(f"\nGPU优化完成: {gpu_metrics['gpu_util']*100:.1f}% 利用率")
        
        # 2. 优化内存分配
        memory_plan = self.memory_optimizer.create_memory_plan(
            total_size=model_config.get('model_size', 10 * 1024 * 1024 * 1024),
            batch_size=gpu_config['batch_size'],
            seq_length=model_config.get('seq_length', 2048)
        )
        print("\n内存规划完成:")
        print(f"- GPU权重: {memory_plan['memory_usage']['gpu']['weights']/1024/1024/1024:.1f}GB")
        print(f"- GPU激活: {memory_plan['memory_usage']['gpu']['activations']/1024/1024/1024:.1f}GB")
        
        # 3. 优化PCIe传输
        pcie_config = self.pcie_optimizer.optimize_pipeline(
            batch_size=gpu_config['batch_size'],
            hidden_size=model_config.get('hidden_size', 4096),
            num_layers=model_config.get('num_layers', 32)
        )
        print("\nPCIe优化完成:")
        print(f"- 流水线级数: {pcie_config['num_stages']}")
        print(f"- 传输通道数: {pcie_config['num_channels']}")
        
        # 4. 协调三个优化器
        final_config = self._coordinate_optimizers(
            gpu_config, memory_plan, pcie_config
        )
        
        # 5. 验证最终配置
        validation_results = self._validate_config(final_config)
        if not validation_results['valid']:
            print("\n警告: 配置验证未通过,正在调整...")
            final_config = self._adjust_config(
                final_config, 
                validation_results['issues']
            )
            
        return final_config
        
    def _coordinate_optimizers(self,
                             gpu_config: Dict,
                             memory_plan: Dict,
                             pcie_config: Dict) -> Dict:
        """协调三个优化器的配置"""
        # 基础配置
        config = {
            'batch_size': gpu_config['batch_size'],
            'gpu_layers': gpu_config['max_gpu_layers'],
            'num_compute_streams': gpu_config['num_compute_streams']
        }
        
        # 内存配置
        config.update({
            'gpu_memory': {
                'weights': memory_plan['memory_usage']['gpu']['weights'],
                'activations': memory_plan['memory_usage']['gpu']['activations'],
                'workspace': memory_plan['memory_usage']['gpu']['workspace']
            },
            'cpu_memory': {
                'weights': memory_plan['memory_usage']['cpu']['weights'],
                'activations': memory_plan['memory_usage']['cpu']['activations']
            }
        })
        
        # PCIe传输配置
        config.update({
            'pcie': {
                'num_channels': pcie_config['num_channels'],
                'buffer_size': pcie_config['params_per_transfer'],
                'compression': pcie_config['use_compression']
            }
        })
        
        # 协调可能的冲突
        if config['gpu_layers'] > memory_plan['page_table']['gpu'][-1] + 1:
            # GPU层数超过页表容量,调整
            config['gpu_layers'] = memory_plan['page_table']['gpu'][-1] + 1
            
        if config['batch_size'] * config['gpu_layers'] > 32:
            # 总并行度过高,调整批大小
            config['batch_size'] = 32 // config['gpu_layers']
            
        return config
        
    def _validate_config(self, config: Dict) -> Dict:
        """验证配置的可行性"""
        issues = []
        
        # 检查显存使用
        total_gpu_mem = sum(config['gpu_memory'].values())
        if total_gpu_mem > 15 * 1024 * 1024 * 1024:  # 15GB上限
            issues.append('gpu_memory_exceeded')
            
        # 检查PCIe带宽
        pcie_bandwidth = (
            config['batch_size'] * 
            config['gpu_layers'] * 
            config['pcie']['buffer_size']
        ) / 1e9  # GB/s
        if pcie_bandwidth > 14.0:  # 14GB/s上限
            issues.append('pcie_bandwidth_exceeded')
            
        # 检查批大小
        if config['batch_size'] < 1:
            issues.append('batch_size_too_small')
            
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
        
    def _adjust_config(self, config: Dict, issues: list) -> Dict:
        """根据验证问题调整配置"""
        if 'gpu_memory_exceeded' in issues:
            # 减少GPU层数
            config['gpu_layers'] = max(2, config['gpu_layers'] - 1)
            
        if 'pcie_bandwidth_exceeded' in issues:
            # 启用压缩或减少并行度
            config['pcie']['compression'] = True
            config['pcie']['num_channels'] = max(1, config['pcie']['num_channels'] - 1)
            
        if 'batch_size_too_small' in issues:
            config['batch_size'] = 1
            
        return config
        
    def monitor_performance(self, interval: float = 1.0):
        """持续监控性能"""
        while True:
            # 更新监控指标
            self.monitor.update()
            
            # 获取当前指标
            metrics = self.monitor.get_average_metrics(window=5)
            
            # 检查是否需要调整
            if metrics.compute_util < self.targets['gpu_util']:
                self._adjust_for_low_gpu_util()
            elif metrics.memory_util > self.targets['memory_util']:
                self._adjust_for_high_memory()
            elif metrics.pcie_util < self.targets['pcie_util']:
                self._adjust_for_low_pcie()
                
            time.sleep(interval)
            
    def _adjust_for_low_gpu_util(self):
        """处理GPU利用率过低"""
        if self.current_state['batch_size'] < 32:
            # 尝试增加批大小
            self.current_state['batch_size'] *= 2
        elif self.current_state['gpu_layers'] < 5:
            # 尝试增加GPU层数
            self.current_state['gpu_layers'] += 1
            
    def _adjust_for_high_memory(self):
        """处理显存使用过高"""
        if self.current_state['gpu_layers'] > 2:
            # 减少GPU层数
            self.current_state['gpu_layers'] -= 1
        elif self.current_state['batch_size'] > 1:
            # 减小批大小
            self.current_state['batch_size'] //= 2
            
    def _adjust_for_low_pcie(self):
        """处理PCIe带宽利用率过低"""
        if not self.current_state['compression_enabled']:
            # 启用压缩
            self.current_state['compression_enabled'] = True
        elif self.current_state['prefetch_size'] < 256 * 1024 * 1024:
            # 增加预取大小
            self.current_state['prefetch_size'] *= 2
            
    def should_continue(self) -> bool:
        """Determine if we should continue iteration"""
        if self.iteration_count >= self.max_iterations:
            return False
            
        # Check if performance has converged
        if self.iteration_count > 10:
            recent_metrics = self.analyzer.get_average_metrics(window=5)
            if recent_metrics.throughput < self.best_throughput * 1.01:
                return False
                
        return True
        
    def update_config(self) -> IterationConfig:
        """Update configuration based on performance analysis"""
        bottlenecks = self.analyzer.detect_bottlenecks()
        new_config = self.current_config
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'GPU Utilization':
                new_config.batch_size = min(new_config.batch_size * 2, 32)
                new_config.optimization_flags['tf32_enabled'] = True
            elif bottleneck['type'] == 'Memory Utilization':
                new_config.batch_size = max(new_config.batch_size // 2, 1)
                new_config.memory_fraction -= 0.1
            elif bottleneck['type'] == 'PCIe Bandwidth':
                new_config.pcie_buffer_size *= 2
                
        return new_config
        
    def run_iteration(self) -> Dict:
        """运行单次性能优化迭代"""
        self.iteration_count += 1
        
        # 开始监控
        self.monitor.reset()
        self.monitor.update()
        
        # 运行测试负载
        time.sleep(1)  # 实际中替换为模型推理
        
        # 更新监控数据
        self.monitor.update()
        
        # 分析性能指标
        metrics = self.monitor.get_average_metrics()
        self.analyzer.add_metrics(metrics)
        
        # 更新最佳配置
        if metrics.throughput > self.best_throughput:
            self.best_throughput = metrics.throughput
            self.best_config = self.current_config
            
        # 获取优化建议
        suggestions = self.analyzer.suggest_optimizations()
        
        # 更新配置
        self.current_config = self.update_config()
        
        return {
            'iteration': self.iteration_count,
            'metrics': metrics.to_dict(),
            'bottlenecks': self.analyzer.detect_bottlenecks(),
            'suggestions': suggestions,
            'config': vars(self.current_config)
        }
        
    def get_final_report(self) -> Dict:
        """生成最终优化报告"""
        return {
            'total_iterations': self.iteration_count,
            'best_config': vars(self.best_config),
            'best_throughput': self.best_throughput,
            'performance_summary': self.analyzer.get_performance_summary(),
            'optimization_history': [m.to_dict() for m in self.analyzer.metrics_history]
        }
        
    def save_report(self, file_path: str):
        """保存优化报告到文件"""
        report = self.get_final_report()
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)