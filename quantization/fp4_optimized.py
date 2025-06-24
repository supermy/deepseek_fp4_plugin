import torch
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class CudaConfig:
    """CUDA配置"""
    threads_per_block: int = 256
    blocks_per_sm: int = 4
    shared_memory_size: int = 48 * 1024  # 48KB
    stream_priority: int = 0  # 高优先级
    
class OptimizedFP4:
    """优化的FP4计算实现"""
    
    def __init__(self):
        self.device = torch.device("cuda")
        self._init_cuda_config()
        
    def _init_cuda_config(self):
        """初始化CUDA配置"""
        props = torch.cuda.get_device_properties(self.device)
        
        self.cuda_config = CudaConfig(
            threads_per_block=min(256, props.maxThreadsPerBlock // 2),
            blocks_per_sm=max(4, props.maxThreadsPerMultiProcessor // 256),
            shared_memory_size=min(48*1024, props.maxSharedMemoryPerBlock),
            stream_priority=0
        )
        
        # 创建CUDA流
        self.compute_stream = torch.cuda.Stream(
            device=self.device,
            priority=self.cuda_config.stream_priority
        )
    
    def optimize_launch_config(self, 
                             input_size: Tuple[int, ...],
                             perf_monitor: Optional[Dict] = None) -> Dict:
        """优化内核启动配置"""
        props = torch.cuda.get_device_properties(self.device)
        
        # 计算理论最大占用率
        theoretical_occupancy = min(1.0, (
            self.cuda_config.threads_per_block * 
            self.cuda_config.blocks_per_sm /
            props.maxThreadsPerMultiProcessor
        ))
        
        # 调整配置以最大化GPU利用率
        if perf_monitor and 'gpu_util' in perf_monitor:
            current_util = perf_monitor['gpu_util']
            if current_util < 0.8:  # GPU利用率低于80%
                # 增加并行度
                self.cuda_config.blocks_per_sm = min(
                    self.cuda_config.blocks_per_sm + 1,
                    props.maxThreadsPerMultiProcessor // 
                    self.cuda_config.threads_per_block
                )
        
        return {
            'grid_size': (
                props.multiProcessorCount * 
                self.cuda_config.blocks_per_sm
            ),
            'block_size': self.cuda_config.threads_per_block,
            'shared_memory': self.cuda_config.shared_memory_size,
            'theoretical_occupancy': theoretical_occupancy
        }
    
    @torch.no_grad()
    def compute(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """执行优化的FP4计算"""
        batch_size, hidden_size = inputs.shape
        expert_size = weights.size(0)
        
        # 分配输出tensor
        outputs = torch.empty(
            (batch_size, expert_size),
            dtype=inputs.dtype,
            device=self.device
        )
        
        # 优化启动配置
        launch_config = self.optimize_launch_config(inputs.shape)
        
        with torch.cuda.stream(self.compute_stream):
            # 调用优化的CUDA核函数
            torch.ops.deepseek_fp4.optimized_expert_compute(
                inputs,
                weights,
                outputs,
                grid=launch_config['grid_size'],
                block=launch_config['block_size'],
                shared_memory=launch_config['shared_memory']
            )
            
        return outputs
    
    @torch.no_grad()    
    def pack_weights(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """优化的权重打包"""
        # 分配输出tensors
        packed = torch.empty(
            weights.shape[0] // 2,
            dtype=torch.uint8,
            device=self.device
        )
        scales = torch.empty(
            (weights.shape[0] + 127) // 128,
            dtype=torch.float32,
            device=self.device  
        )
        
        # 优化启动配置
        launch_config = self.optimize_launch_config(weights.shape)
        
        with torch.cuda.stream(self.compute_stream):
            # 调用优化的打包核函数
            torch.ops.deepseek_fp4.optimized_pack_weights(
                weights,
                packed,
                scales,
                grid=launch_config['grid_size'],
                block=launch_config['block_size'],
                shared_memory=launch_config['shared_memory']
            )
            
        return packed, scales
        
    def __call__(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """便捷调用接口"""
        return self.compute(inputs, weights)