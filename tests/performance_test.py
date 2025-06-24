import os
import time
import json
import torch
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from ..utils.perf_coordinator import PerformanceCoordinator
from ..utils.perf_monitor import MetricsAnalyzer
from ..configs.gpu_16g_perf import GPU16GConfig

def run_performance_test(
    model_path: str,
    batch_sizes: List[int] = [1, 2, 4, 8],
    seq_lengths: List[int] = [512, 1024, 2048],
    test_duration: int = 300,  # 5分钟测试
    warmup_steps: int = 50
) -> Dict:
    """运行完整的性能测试"""
    
    print("\n=== 开始性能测试 ===")
    results = {}
    
    # 初始化组件
    coordinator = PerformanceCoordinator()
    analyzer = MetricsAnalyzer()
    config = GPU16GConfig()
    
    # 创建结果目录
    os.makedirs('test_results', exist_ok=True)
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            print(f"\n测试配置: batch_size={batch_size}, seq_length={seq_length}")
            
            # 更新配置
            test_config = {
                'model_size': 10 * 1024 * 1024 * 1024,  # 10GB模型
                'batch_size': batch_size,
                'seq_length': seq_length,
                'hidden_size': 4096,
                'num_layers': 32
            }
            
            # 优化配置
            optimal_config = coordinator.optimize_all(test_config)
            
            # 预热
            print("预热中...")
            for _ in range(warmup_steps):
                # 模拟推理
                time.sleep(0.1)  # 简化示例,实际应运行真实推理
                
            # 运行测试
            print("开始测试...")
            start_time = time.time()
            step = 0
            
            while time.time() - start_time < test_duration:
                step += 1
                
                # 模拟一次推理
                time.sleep(0.1)  # 简化示例
                
                # 收集性能指标
                coordinator.monitor.update()
                metrics = coordinator.monitor.get_average_metrics(window=5)
                analyzer.add_metrics(metrics)
                
                # 每50步打印进度
                if step % 50 == 0:
                    print(f"已完成 {step} 步...")
                    
            # 分析结果
            avg_metrics = analyzer.get_average_metrics()
            bottlenecks = analyzer.detect_bottlenecks()
            trends = analyzer.analyze_trends()
            
            # 保存结果
            result_key = f"b{batch_size}_s{seq_length}"
            results[result_key] = {
                'config': optimal_config,
                'metrics': avg_metrics.to_dict(),
                'bottlenecks': bottlenecks,
                'trends': trends
            }
            
            # 生成图表
            plot_path = f"test_results/perf_plot_{result_key}.png"
            analyzer.plot_metrics(save_path=plot_path)
            
            # 导出详细指标
            metrics_path = f"test_results/metrics_{result_key}.json"
            analyzer.export_metrics(metrics_path)
            
    # 生成总结报告
    generate_summary_report(results)
    
    return results

def generate_summary_report(results: Dict):
    """生成性能测试总结报告"""
    print("\n=== 生成性能测试报告 ===")
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    batch_sizes = sorted(set(int(k.split('_')[0][1:]) for k in results.keys()))
    seq_lengths = sorted(set(int(k.split('_')[1][1:]) for k in results.keys()))
    
    # 1. GPU利用率热力图
    gpu_utils = np.zeros((len(batch_sizes), len(seq_lengths)))
    for i, b in enumerate(batch_sizes):
        for j, s in enumerate(seq_lengths):
            key = f"b{b}_s{s}"
            if key in results:
                gpu_utils[i, j] = results[key]['metrics']['gpu_util'] * 100
                
    im1 = ax1.imshow(gpu_utils)
    ax1.set_xticks(range(len(seq_lengths)))
    ax1.set_yticks(range(len(batch_sizes)))
    ax1.set_xticklabels(seq_lengths)
    ax1.set_yticklabels(batch_sizes)
    ax1.set_xlabel('序列长度')
    ax1.set_ylabel('批大小')
    ax1.set_title('GPU利用率 (%)')
    plt.colorbar(im1, ax=ax1)
    
    # 2. 延迟随批大小变化
    for s in seq_lengths:
        latencies = [results[f"b{b}_s{s}"]['metrics']['total_latency'] 
                    for b in batch_sizes if f"b{b}_s{s}" in results]
        ax2.plot(batch_sizes[:len(latencies)], latencies, 
                marker='o', label=f'seq_len={s}')
    ax2.set_xlabel('批大小')
    ax2.set_ylabel('延迟 (ms)')
    ax2.set_title('延迟随批大小变化')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 吞吐量分析
    throughputs = []
    configs = []
    for key, data in results.items():
        b = int(key.split('_')[0][1:])
        s = int(key.split('_')[1][1:])
        latency = data['metrics']['total_latency']
        throughput = (b * s) / (latency / 1000)  # tokens/s
        throughputs.append(throughput)
        configs.append(f"b{b}_s{s}")
        
    ax3.bar(range(len(configs)), throughputs)
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(configs, rotation=45)
    ax3.set_ylabel('吞吐量 (tokens/s)')
    ax3.set_title('各配置吞吐量对比')
    
    # 4. 资源利用率对比
    for key, data in results.items():
        metrics = data['metrics']
        ax4.plot([1, 2, 3], 
                [metrics['gpu_util']*100,
                 metrics['memory_util']*100,
                 metrics['pcie_util']*100],
                marker='o', label=key)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['GPU', '显存', 'PCIe'])
    ax4.set_ylabel('利用率 (%)')
    ax4.set_title('资源利用率对比')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('test_results/summary_report.png')
    
    # 生成文本报告
    with open('test_results/performance_report.md', 'w') as f:
        f.write("# 性能测试报告\n\n")
        
        # 最佳配置
        best_throughput = max(throughputs)
        best_config = configs[throughputs.index(best_throughput)]
        f.write(f"## 最佳配置\n")
        f.write(f"- 配置: {best_config}\n")
        f.write(f"- 吞吐量: {best_throughput:.1f} tokens/s\n")
        f.write(f"- GPU利用率: {results[best_config]['metrics']['gpu_util']*100:.1f}%\n\n")
        
        # 性能瓶颈分析
        f.write("## 性能瓶颈\n")
        for bottleneck in results[best_config]['bottlenecks']:
            f.write(f"- {bottleneck['type']}: {bottleneck['current']} "
                   f"(目标: {bottleneck['target']})\n")
            for suggestion in bottleneck['suggestions']:
                f.write(f"  * {suggestion}\n")
        f.write("\n")
        
        # 优化建议
        f.write("## 优化建议\n")
        seen_suggestions = set()
        for data in results.values():
            for bottleneck in data['bottlenecks']:
                for suggestion in bottleneck['suggestions']:
                    if suggestion not in seen_suggestions:
                        f.write(f"- {suggestion}\n")
                        seen_suggestions.add(suggestion)
                        
    print("\n性能测试报告已生成:")
    print("- 图表总结: test_results/summary_report.png")
    print("- 详细报告: test_results/performance_report.md")
    
if __name__ == "__main__":
    model_path = "deepseek-r1-fp4"
    results = run_performance_test(model_path)