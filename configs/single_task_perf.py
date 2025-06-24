from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class SingleTaskConfig:
    """单任务性能优化配置"""
    
    # GPU配置
    gpu_memory_fraction: float = 0.95  # 预留5%显存
    max_workspace_size: int = 1024*1024*1024  # 1GB workspace
    prefer_larger_batch: bool = True  # 倾向于使用更大batch以提高利用率
    
    # 计算配置
    enable_tf32: bool = True  # 启用TF32以提升性能
    use_cuda_graph: bool = True  # 启用CUDA图
    overlap_compute: bool = True  # 启用计算重叠
    
    # 内存配置
    pinned_memory: bool = True  # 使用固定内存
    max_persistent_cache: int = 2*1024*1024*1024  # 2GB持久缓存
    enable_async_copy: bool = True  # 启用异步拷贝
    
    # 调度配置
    compute_priority: int = 0  # 高优先级计算流
    transfer_priority: int = -1  # 低优先级传输流
    max_active_blocks: int = 32  # 每个SM最大活跃块数
    
    # 专家配置
    expert_parallelism: int = 4  # 并行处理的专家数
    expert_threshold: float = 0.1  # 专家激活阈值
    prefetch_experts: int = 2  # 预取专家数量
    
    @classmethod
    def optimize_for_gpu(cls, gpu_name: str) -> "SingleTaskConfig":
        """根据GPU型号优化配置"""
        config = cls()
        
        if "A100" in gpu_name:
            # A100优化配置
            config.max_workspace_size = 4*1024*1024*1024  # 4GB
            config.max_active_blocks = 48
            config.expert_parallelism = 8
        elif "H100" in gpu_name:
            # H100优化配置  
            config.max_workspace_size = 8*1024*1024*1024  # 8GB
            config.max_active_blocks = 64
            config.expert_parallelism = 16
            
        return config
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'compute': {
                'tf32': self.enable_tf32,
                'cuda_graph': self.use_cuda_graph,
                'overlap': self.overlap_compute,
                'priority': self.compute_priority,
                'max_blocks': self.max_active_blocks
            },
            'memory': {
                'gpu_fraction': self.gpu_memory_fraction,
                'workspace': self.max_workspace_size,
                'pinned': self.pinned_memory,
                'persistent_cache': self.max_persistent_cache,
                'async_copy': self.enable_async_copy
            },
            'expert': {
                'parallelism': self.expert_parallelism,
                'threshold': self.expert_threshold,
                'prefetch': self.prefetch_experts
            }
        }