from dataclasses import dataclass
from typing import Dict, Optional
import torch
import os
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class GPU16GConfig:
    """16GB GPU显存优化配置"""
    
    # GPU显存分配 (总共16GB)
    model_weights: int = 10 * 1024 * 1024 * 1024    # 10GB用于模型权重
    activation_cache: int = 3 * 1024 * 1024 * 1024  # 3GB用于激活缓存
    workspace: int = 2 * 1024 * 1024 * 1024         # 2GB工作空间
    system_reserved: int = 1 * 1024 * 1024 * 1024   # 1GB系统预留
    
    # 计算优化
    max_gpu_layers: int = 4        # GPU常驻4层(约10GB)
    prefetch_layers: int = 2       # 预取2层到CPU内存
    batch_size: int = 8           # 最优批处理大小
    seq_length: int = 2048        # 最大序列长度
    
    # 内存管理
    pinned_memory: bool = True     # 使用固定内存
    enable_async_copy: bool = True # 启用异步拷贝
    use_static_shapes: bool = True # 使用静态形状优化
    
    # 计算流配置
    num_compute_streams: int = 2   # 使用2个计算流
    num_copy_streams: int = 1      # 使用1个复制流
    stream_priority: Dict = {
        'compute': 0,   # 计算流最高优先级
        'prefetch': -1, # 预取流中等优先级
        'copy': -2      # 复制流最低优先级
    }
    
    # CUDA优化
    threads_per_block: int = 256   # 每块线程数
    blocks_per_sm: int = 8         # 每个SM的块数
    shared_memory_size: int = 48 * 1024  # 48KB共享内存
    use_tensor_cores: bool = True  # 使用Tensor Cores
    enable_tf32: bool = True       # 启用TF32
    
    # IO优化
    pcie_buffer_size: int = 16 * 1024 * 1024  # 16MB PCIe传输缓冲区
    use_compressed_transfer: bool = True       # 启用压缩传输
    prefetch_distance: int = 2               # 预取距离
    
    def __init__(self):
        # 加载基础配置
        config_path = os.path.join(os.path.dirname(__file__), 'gpu_16g_best_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 硬件规格
        self.gpu_memory = self._parse_size(config['hardware_specs']['gpu_memory'])
        self.system_memory = self._parse_size(config['hardware_specs']['system_memory'])
        self.pcie_bandwidth = self._parse_size(config['hardware_specs']['pcie_bandwidth'])
        
        # 内存配置
        self.gpu_model_mem = self._parse_size(config['memory_config']['gpu']['model_weights'])
        self.gpu_activation_mem = self._parse_size(config['memory_config']['gpu']['activation_cache'])
        self.gpu_workspace_mem = self._parse_size(config['memory_config']['gpu']['workspace'])
        
        self.cpu_model_cache = self._parse_size(config['memory_config']['cpu']['model_cache'])
        self.cpu_prefetch_buffer = self._parse_size(config['memory_config']['cpu']['prefetch_buffer'])
        self.cpu_workspace = self._parse_size(config['memory_config']['cpu']['workspace'])
        
        # 计算配置
        self.max_batch_size = config['compute_config']['max_batch_size']
        self.gpu_resident_layers = config['compute_config']['gpu_resident_layers']
        self.prefetch_layers = config['compute_config']['prefetch_layers']
        self.compute_streams = config['compute_config']['compute_streams']
        self.enable_tf32 = config['compute_config']['enable_tf32']
        self.cuda_graph_mode = config['compute_config']['cuda_graph_mode']
        self.threads_per_block = config['compute_config']['threads_per_block']
        
        # PCIe配置
        self.transfer_buffer = self._parse_size(config['pcie_config']['transfer_buffer'])
        self.num_channels = config['pcie_config']['num_channels']
        self.compression_enabled = config['pcie_config']['compression_enabled']
        self.compression_ratio = config['pcie_config']['compression_ratio']
        
        # 优化目标
        self.target_gpu_util = self._parse_percentage(config['optimization_targets']['gpu_utilization'])
        self.target_memory_util = self._parse_percentage(config['optimization_targets']['memory_utilization'])
        self.target_pcie_util = self._parse_percentage(config['optimization_targets']['pcie_utilization'])
        self.target_latency = self._parse_time(config['optimization_targets']['target_latency'])
        
        # 内存管理
        self.page_size = self._parse_size(config['memory_management']['page_size'])
        self.pinned_memory = config['memory_management']['pinned_memory']
        self.memory_pool = config['memory_management']['memory_pool']
        self.pool_size = self._parse_size(config['memory_management']['pool_size'])
        
        # 性能监控
        self.metrics_window = config['monitoring']['metrics_window']
        self.update_interval = self._parse_time(config['monitoring']['update_interval'])
        self.log_level = config['monitoring']['log_level']
        
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串(如 "16GB")为字节数"""
        unit_map = {
            'KB': 1024,
            'MB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024,
            'TB': 1024 * 1024 * 1024 * 1024
        }
        
        for unit, multiplier in unit_map.items():
            if size_str.upper().endswith(unit):
                value = float(size_str[:-len(unit)])
                return int(value * multiplier)
        return int(size_str)  # 无单位,假设为字节
        
    def _parse_percentage(self, percentage_str: str) -> float:
        """解析百分比字符串(如 "85%")为小数"""
        return float(percentage_str.strip('%')) / 100.0
        
    def _parse_time(self, time_str: str) -> float:
        """解析时间字符串(如 "100ms")为秒数"""
        unit_map = {
            'us': 1e-6,
            'ms': 1e-3,
            's': 1.0
        }
        
        for unit, multiplier in unit_map.items():
            if time_str.endswith(unit):
                value = float(time_str[:-len(unit)])
                return value * multiplier
        return float(time_str)  # 无单位,假设为秒
        
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)
        
    def validate(self) -> bool:
        """验证配置的合法性"""
        try:
            # 验证内存分配
            total_gpu_mem = (
                self.gpu_model_mem +
                self.gpu_activation_mem +
                self.gpu_workspace_mem
            )
            if total_gpu_mem > self.gpu_memory:
                raise ValueError("GPU内存分配超过总容量")
                
            # 验证PCIe配置
            max_transfer = (
                self.transfer_buffer *
                self.num_channels *
                (1.0 if not self.compression_enabled else self.compression_ratio)
            )
            if max_transfer > self.pcie_bandwidth:
                raise ValueError("PCIe传输配置超过带宽上限")
                
            # 验证计算配置
            if self.max_batch_size * self.gpu_resident_layers > 32:
                raise ValueError("并行度配置过高")
                
            return True
            
        except Exception as e:
            print(f"配置验证失败: {str(e)}")
            return False
            
    def optimize_for_throughput(self):
        """优化吞吐量配置"""
        # 增加批处理大小
        self.max_batch_size = min(32, self.max_batch_size * 2)
        
        # 增加GPU常驻层数
        self.gpu_resident_layers = min(5, self.gpu_resident_layers + 1)
        
        # 启用所有优化选项
        self.enable_tf32 = True
        self.compression_enabled = True
        self.memory_pool = True
        
    def optimize_for_latency(self):
        """优化延迟配置"""
        # 减小批处理大小
        self.max_batch_size = max(1, self.max_batch_size // 2)
        
        # 减少预取层数
        self.prefetch_layers = max(1, self.prefetch_layers - 1)
        
        # 调整传输配置
        self.transfer_buffer = min(self.transfer_buffer, 8 * 1024 * 1024)  # 最大8MB
        self.num_channels = 1
        
    def enable_optimization(self):
        """启用所有优化选项"""
        # 启用GPU优化
        self.enable_tf32 = True
        self.cuda_graph_mode = 'static'
        
        # 优化内存访问
        self.pinned_memory = True
        self.memory_pool = True
        
        # 启用PCIe优化
        self.compression_enabled = True
        self.num_channels = 2
        
        # 调整预取策略
        self.prefetch_layers = 2
        
    def optimize_for_hardware(self):
        """针对硬件特性优化配置"""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            # 优化CUDA参数
            self.threads_per_block = min(256, props.maxThreadsPerBlock // 2)
            self.blocks_per_sm = min(8, props.maxThreadsPerMultiProcessor // self.threads_per_block)
            self.shared_memory_size = min(48 * 1024, props.maxSharedMemoryPerBlock)
            
            # 检查硬件能力
            if props.major >= 8:  # Ampere或更新架构
                self.use_tensor_cores = True
                self.enable_tf32 = True
            
            # 根据SM数量调整流的数量
            self.num_compute_streams = min(2, props.multiProcessorCount // 4)
            
        return self
    
    def get_launch_config(self) -> Dict:
        """获取CUDA核函数启动配置"""
        return {
            'block_size': self.threads_per_block,
            'grid_size': self.blocks_per_sm,
            'shared_mem_size': self.shared_memory_size,
            'stream_config': {
                'compute': self.num_compute_streams,
                'copy': self.num_copy_streams
            }
        }