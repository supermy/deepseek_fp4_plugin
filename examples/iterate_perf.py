import os
import time
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from ..configs.gpu_16g_perf import GPU16GConfig
from ..utils.perf_coordinator import PerformanceCoordinator
from ..utils.perf_monitor import MetricsAnalyzer

class PerformanceIterator:
    """性能迭代优化器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = GPU16GConfig()
        self.coordinator = PerformanceCoordinator()
        self.analyzer = MetricsAnalyzer()
        self.history = []
        
    def iterate(self, 
        num_iterations: int = 10,
        optimization_target: str = 'balanced',
        plot_progress: bool = True
    ) -> Dict:
        """运行性能迭代优化"""
        
        print(f"\n=== 开始性能迭代优化 (目标: {optimization_target}) ===")
        
        best_metrics = None
        best_config = None
        
        for i in range(num_iterations):
            print(f"\n迭代 {i+1}/{num_iterations}")
            
            # 1. 根据历史调整配置
            if i > 0:
                self._adjust_config_from_history()
                
            # 2. 运行测试
            metrics = self._run_test()
            
            # 3. 评估性能
            score = self._evaluate_performance(metrics, optimization_target)
            
            # 4. 记录结果
            iteration_result = {
                'iteration': i + 1,
                'config': self.config.to_dict(),
                'metrics': metrics.to_dict(),
                'score': score
            }
            self.history.append(iteration_result)
            
            # 5. 更新最佳结果
            if not best_metrics or score > best_metrics['score']:
                best_metrics = iteration_result
                best_config = self.config.to_dict()
                print(f"发现新的最佳配置! 得分: {score:.2f}")
                
            # 6. 打印当前状态
            self._print_status(metrics)
            
        # 保存优化历史
        self._save_history()
        
        # 绘制进度图表
        if plot_progress:
            self._plot_progress()
            
        return {
            'best_config': best_config,
            'best_metrics': best_metrics,
            'history': self.history
        }
        
    def _run_test(self) -> Dict:
        """运行单次测试"""
        metrics_samples = []
        
        # 预热
        for _ in range(5):
            self.coordinator.monitor.update()
            time.sleep(0.1)
            
        # 收集样本
        for _ in range(10):
            self.coordinator.monitor.update()
            metrics = self.coordinator.monitor.get_average_metrics(window=5)
            metrics_samples.append(metrics)
            time.sleep(0.1)
            
        # 计算平均指标
        return self._average_metrics(metrics_samples)
        
    def _evaluate_performance(self, 
        metrics: Dict,
        optimization_target: str
    ) -> float:
        """评估性能得分"""
        
        if optimization_target == 'throughput':
            # 吞吐量优先
            score = (
                0.6 * metrics['gpu_util'] +
                0.2 * (1 - metrics['total_latency']/1000) +
                0.2 * metrics['pcie_util']
            )
        elif optimization_target == 'latency':
            # 延迟优先
            score = (
                0.6 * (1 - metrics['total_latency']/1000) +
                0.2 * metrics['gpu_util'] +
                0.2 * (1 - metrics['memory_util'])
            )
        else:  # balanced
            # 平衡模式
            score = (
                0.4 * metrics['gpu_util'] +
                0.3 * (1 - metrics['total_latency']/1000) +
                0.2 * (1 - metrics['memory_util']) +
                0.1 * metrics['pcie_util']
            )
            
        return score
        
    def _adjust_config_from_history(self):
        """根据历史调整配置"""
        if len(self.history) < 2:
            return
            
        # 获取最近两次迭代
        current = self.history[-1]
        previous = self.history[-2]
        
        # 性能改善
        if current['score'] > previous['score']:
            # 继续当前方向的调整
            if current['config']['batch_size'] > previous['config']['batch_size']:
                self.config.max_batch_size = min(32, self.config.max_batch_size + 1)
            if current['config']['gpu_resident_layers'] > previous['config']['gpu_resident_layers']:
                self.config.gpu_resident_layers = min(5, self.config.gpu_resident_layers + 1)
        else:
            # 性能下降,回退并尝试其他方向
            self.config.max_batch_size = max(1, self.config.max_batch_size - 1)
            self.config.prefetch_layers = max(1, self.config.prefetch_layers - 1)
            
    def _average_metrics(self, metrics_list: List[Dict]) -> Dict:
        """计算平均指标"""
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = sum(values) / len(values)
        return avg_metrics
        
    def _print_status(self, metrics: Dict):
        """打印当前状态"""
        print("\n当前性能指标:")
        print(f"- GPU利用率: {metrics['gpu_util']*100:.1f}%")
        print(f"- 显存利用率: {metrics['memory_util']*100:.1f}%")
        print(f"- PCIe带宽利用率: {metrics['pcie_util']*100:.1f}%")
        print(f"- 总延迟: {metrics['total_latency']:.1f}ms")
        
        # 打印瓶颈分析
        bottlenecks = self.analyzer.detect_bottlenecks()
        if bottlenecks:
            print("\n检测到性能瓶颈:")
            for bottleneck in bottlenecks:
                print(f"- {bottleneck['type']}: {bottleneck['current']} "
                      f"(目标: {bottleneck['target']})")
                      
    def _save_history(self):
        """保存优化历史"""
        history_file = 'optimization_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n优化历史已保存到: {history_file}")
        
    def _plot_progress(self):
        """绘制优化进度图表"""
        iterations = [h['iteration'] for h in self.history]
        scores = [h['score'] for h in self.history]
        gpu_utils = [h['metrics']['gpu_util'] * 100 for h in self.history]
        latencies = [h['metrics']['total_latency'] for h in self.history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 性能得分
        ax1.plot(iterations, scores, 'b-', marker='o')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('性能得分')
        ax1.set_title('性能优化进度')
        ax1.grid(True)
        
        # GPU利用率和延迟
        ax2.plot(iterations, gpu_utils, 'g-', marker='o', label='GPU利用率 (%)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(iterations, latencies, 'r-', marker='s', label='延迟 (ms)')
        
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('GPU利用率 (%)')
        ax2_twin.set_ylabel('延迟 (ms)')
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('optimization_progress.png')
        print("\n优化进度图表已保存到: optimization_progress.png")
        
def main():
    # 运行示例
    model_path = "deepseek-r1-fp4"
    iterator = PerformanceIterator(model_path)
    
    # 运行10轮优化迭代
    results = iterator.iterate(
        num_iterations=10,
        optimization_target='balanced',
        plot_progress=True
    )
    
    # 打印最终结果
    print("\n=== 优化完成 ===")
    print(f"最佳性能得分: {results['best_metrics']['score']:.2f}")
    print("\n最佳配置:")
    for key, value in results['best_config'].items():
        print(f"- {key}: {value}")
        
if __name__ == "__main__":
    main()