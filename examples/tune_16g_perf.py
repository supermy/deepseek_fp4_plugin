import os
import time
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional
from ..configs.gpu_16g_perf import GPU16GConfig
from ..utils.perf_coordinator import PerformanceCoordinator
from ..utils.perf_monitor import MetricsAnalyzer
from ..utils.gpu_16g_tuner import GPU16GAutoTuner
from ..utils.perf_monitor import PerformanceMonitor

def run_performance_tuning(model_path: str):
    """运行16GB GPU性能调优"""
    print("\n=== 开始16GB GPU性能调优 ===")
    
    # 初始化调优器和监控器
    tuner = GPU16GAutoTuner()
    monitor = PerformanceMonitor()
    
    # 运行自动调优
    optimal_config, metrics = tuner.optimize(
        time_budget=120.0,      # 2分钟调优时间
        target_gpu_util=0.85    # 目标85%GPU利用率
    )
    
    # 应用最优配置
    config = tuner.apply_best_config()
    if not config:
        print("调优失败!")
        return
        
    # 运行性能测试
    print("\n=== 运行性能测试 ===")
    test_iterations = 100
    gpu_utils = []
    mem_utils = []
    latencies = []
    
    # 预热
    for _ in range(10):
        monitor.update()
        
    # 收集性能数据
    start_time = time.time()
    for i in range(test_iterations):
        # 更新监控指标
        monitor.update()
        metrics = monitor.get_average_metrics(window=5)
        
        # 记录指标
        gpu_utils.append(metrics.compute_util * 100)
        mem_utils.append(metrics.memory_util * 100)
        latencies.append(time.time() - start_time)
        
        # 打印进度
        if (i + 1) % 10 == 0:
            print(f"进度: {i+1}/{test_iterations}")
            
    # 绘制性能图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # GPU利用率随时间变化
    ax1.plot(latencies, gpu_utils, 'b-', label='GPU利用率')
    ax1.plot(latencies, mem_utils, 'r-', label='显存利用率')
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('利用率 (%)')
    ax1.set_title('GPU资源利用率')
    ax1.grid(True)
    ax1.legend()
    
    # 性能分布直方图
    ax2.hist(gpu_utils, bins=20, alpha=0.5, label='GPU利用率分布')
    ax2.set_xlabel('GPU利用率 (%)')
    ax2.set_ylabel('样本数')
    ax2.set_title('GPU利用率分布')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('gpu_16g_perf.png')
    print("\n性能测试结果已保存到 gpu_16g_perf.png")
    
    # 打印性能总结
    monitor.print_summary()
    
def tune_for_throughput(
    model_path: str,
    max_iterations: int = 100,
    target_throughput: float = 100.0  # tokens/s
) -> Dict:
    """优化吞吐量配置"""
    
    print("\n=== 开始吞吐量优化 ===")
    
    # 初始化组件
    config = GPU16GConfig()
    coordinator = PerformanceCoordinator()
    analyzer = MetricsAnalyzer()
    
    # 记录最佳配置
    best_config = None
    best_throughput = 0.0
    
    for iteration in range(max_iterations):
        print(f"\n迭代 {iteration + 1}/{max_iterations}")
        
        # 1. 应用当前配置
        config.optimize_for_throughput()
        test_config = {
            'model_size': config.gpu_model_mem,
            'batch_size': config.max_batch_size,
            'seq_length': config.seq_length,
            'hidden_size': 4096,
            'num_layers': 32
        }
        
        # 2. 运行优化器
        optimal_config = coordinator.optimize_all(test_config)
        
        # 3. 收集性能指标
        for _ in range(10):  # 采样10次
            coordinator.monitor.update()
            metrics = coordinator.monitor.get_average_metrics(window=5)
            analyzer.add_metrics(metrics)
            time.sleep(0.1)
            
        # 4. 计算吞吐量
        avg_metrics = analyzer.get_average_metrics()
        current_throughput = (
            config.batch_size * config.seq_length / 
            (avg_metrics.total_latency / 1000)  # 转换为秒
        )
        
        print(f"当前吞吐量: {current_throughput:.1f} tokens/s")
        print(f"GPU利用率: {avg_metrics.gpu_util*100:.1f}%")
        
        # 5. 更新最佳配置
        if current_throughput > best_throughput:
            best_throughput = current_throughput
            best_config = optimal_config.copy()
            print(f"发现新的最佳配置! 吞吐量: {best_throughput:.1f} tokens/s")
            
        # 6. 检查是否达到目标
        if current_throughput >= target_throughput:
            print(f"\n已达到目标吞吐量!")
            break
            
        # 7. 分析瓶颈并调整
        bottlenecks = analyzer.detect_bottlenecks()
        if bottlenecks:
            print("\n检测到性能瓶颈:")
            for bottleneck in bottlenecks:
                print(f"- {bottleneck['type']}: {bottleneck['current']} "
                      f"(目标: {bottleneck['target']})")
                      
    return {
        'best_config': best_config,
        'best_throughput': best_throughput,
        'iterations': iteration + 1
    }

def tune_for_latency(
    model_path: str,
    max_iterations: int = 100,
    target_latency: float = 100.0  # ms
) -> Dict:
    """优化延迟配置"""
    
    print("\n=== 开始延迟优化 ===")
    
    # 初始化组件
    config = GPU16GConfig()
    coordinator = PerformanceCoordinator()
    analyzer = MetricsAnalyzer()
    
    # 记录最佳配置
    best_config = None
    best_latency = float('inf')
    
    for iteration in range(max_iterations):
        print(f"\n迭代 {iteration + 1}/{max_iterations}")
        
        # 1. 应用当前配置
        config.optimize_for_latency()
        test_config = {
            'model_size': config.gpu_model_mem,
            'batch_size': config.max_batch_size,
            'seq_length': config.seq_length,
            'hidden_size': 4096,
            'num_layers': 32
        }
        
        # 2. 运行优化器
        optimal_config = coordinator.optimize_all(test_config)
        
        # 3. 收集性能指标
        for _ in range(10):  # 采样10次
            coordinator.monitor.update()
            metrics = coordinator.monitor.get_average_metrics(window=5)
            analyzer.add_metrics(metrics)
            time.sleep(0.1)
            
        # 4. 评估延迟
        avg_metrics = analyzer.get_average_metrics()
        current_latency = avg_metrics.total_latency
        
        print(f"当前延迟: {current_latency:.1f} ms")
        print(f"GPU利用率: {avg_metrics.gpu_util*100:.1f}%")
        
        # 5. 更新最佳配置
        if current_latency < best_latency:
            best_latency = current_latency
            best_config = optimal_config.copy()
            print(f"发现新的最佳配置! 延迟: {best_latency:.1f} ms")
            
        # 6. 检查是否达到目标
        if current_latency <= target_latency:
            print(f"\n已达到目标延迟!")
            break
            
        # 7. 分析瓶颈并调整
        bottlenecks = analyzer.detect_bottlenecks()
        if bottlenecks:
            print("\n检测到性能瓶颈:")
            for bottleneck in bottlenecks:
                print(f"- {bottleneck['type']}: {bottleneck['current']} "
                      f"(目标: {bottleneck['target']})")
                      
    return {
        'best_config': best_config,
        'best_latency': best_latency,
        'iterations': iteration + 1
    }

def auto_tune(
    model_path: str,
    optimization_target: str = 'balanced',
    max_iterations: int = 100
) -> Dict:
    """自动调优入口函数"""
    
    print(f"\n=== 开始自动调优 (目标: {optimization_target}) ===")
    
    if optimization_target == 'throughput':
        return tune_for_throughput(
            model_path,
            max_iterations=max_iterations,
            target_throughput=100.0
        )
    elif optimization_target == 'latency':
        return tune_for_latency(
            model_path,
            max_iterations=max_iterations,
            target_latency=100.0
        )
    else:  # balanced
        # 先优化延迟到可接受水平
        latency_results = tune_for_latency(
            model_path,
            max_iterations=max_iterations//2,
            target_latency=150.0  # 放宽延迟要求
        )
        
        # 在延迟约束下优化吞吐量
        throughput_results = tune_for_throughput(
            model_path,
            max_iterations=max_iterations//2,
            target_throughput=80.0  # 适度的吞吐量目标
        )
        
        # 返回综合结果
        return {
            'latency_results': latency_results,
            'throughput_results': throughput_results,
            'final_config': throughput_results['best_config']
        }

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行性能调优
    model_path = "deepseek-r1-fp4"
    run_performance_tuning(model_path)
    
    # 运行自动调优
    results = auto_tune(
        model_path,
        optimization_target='balanced',
        max_iterations=200
    )
    
    # 打印结果
    print("\n=== 调优结果 ===")
    if 'latency_results' in results:
        print(f"最佳延迟: {results['latency_results']['best_latency']:.1f} ms")
        print(f"最佳吞吐量: {results['throughput_results']['best_throughput']:.1f} tokens/s")
    else:
        print(f"最佳性能: {results['best_throughput' if 'best_throughput' in results else 'best_latency']:.1f}")
    print(f"总迭代次数: {results.get('iterations', 200)}")
    
    # 保存最佳配置
    import json
    with open('best_tuned_config.json', 'w') as f:
        json.dump(results['final_config'], f, indent=2)
    
if __name__ == "__main__":
    main()