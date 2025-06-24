import os
import json
import torch
from typing import Optional, Dict, Any
from deepseek_fp4_plugin import DeepseekFP4Plugin
from models.deepseek_v3 import DeepseekV3
from configs.model_configs import QuantMode, QuantConfig
from transformers import PreTrainedTokenizer, AutoTokenizer

def load_config(config_path: str) -> Dict[str, Any]:
    """加载模型配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_quant_config(config_path: str) -> Dict[str, Any]:
    """加载量化配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

class DeepseekInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 FP4 plugin
        self.plugin = DeepseekFP4Plugin()
        
        # 加载模型配置
        config_path = os.path.join(model_path, "config.json")
        self.model_config = load_config(config_path)
        
        # 加载量化配置
        quant_config_path = os.path.join(model_path, "hf_quant_config.json")
        self.quant_config_json = load_quant_config(quant_config_path)
        
        # 设置量化模式和精度
        quant_mode = QuantMode.from_description(use_nvfp4=True)
        self.quant_config = QuantConfig(quant_mode)
        
        # 设置默认精度为 float16
        torch.set_default_dtype(torch.float16)
        
        # 初始化模型
        self.model = DeepseekV3(self.model_config, quant_config=self.quant_config)
        self.model = self.model.half()  # 转换模型为 float16
        
        # 加载预训练权重
        self._load_model_weights()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 将模型移到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()

    def _load_model_weights(self):
        """加载模型权重"""
        checkpoint_path = os.path.join(self.model_path, "model.safetensors")
        try:
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                # 将权重转换为 float16
                state_dict = {k: v.half() if v.dtype == torch.float32 else v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                print("模型权重加载成功 (float16)")
            else:
                raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"加载模型权重失败: {e}")

    def generate(self, 
                prompt: str,
                max_length: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 0.95,
                top_k: int = 50,
                num_beams: int = 1,
                do_sample: bool = True) -> str:
        """
        生成文本
        """
        if max_length is None:
            max_length = self.model_config.get('max_position_embeddings', 2048)
            
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # 解码输出
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
                
        except Exception as e:
            raise RuntimeError(f"生成过程中出错: {e}")

    def optimize_single_task_performance(self):
        """优化单任务 GPU 利用率配置"""
        
        # 计算流配置
        stream_config = {
            'compute': torch.cuda.Stream(),
            'prefetch': torch.cuda.Stream(),
            'transfer': torch.cuda.Stream()
        }

        # 获取 GPU 信息
        device_props = torch.cuda.get_device_properties(self.device)
        num_sms = device_props.multi_processor_count
        max_threads_per_sm = device_props.max_threads_per_multi_processor
        
        # 计算最优线程配置
        threads_per_block = min(1024, max_threads_per_sm // 2)  # 每个block使用一半SM的线程
        blocks_per_grid = num_sms * 4  # 每个SM分配4个block以最大化占用率

        # 更新内存配置
        self.memory_config.update({
            'max_gpu_layers': 6,          # 增加常驻GPU层数
            'prefetch_layers': 3,         # 预取3层以平衡内存
            'expert_threshold': 0.1,      # 降低专家启用阈值提高并行度
            'max_active_experts': 32,     # 最大激活专家数
            'shared_memory_size': 48*1024 # 48KB共享内存/block
        })

        # 更新计算配置
        self.compute_config.update({
            'use_cuda_graph': True,      # 启用CUDA图
            'enable_fusion': True,       # 启用算子融合
            'threads_per_block': threads_per_block,
            'blocks_per_grid': blocks_per_grid,
            'shared_memory_size': self.memory_config['shared_memory_size'],
            'stream_config': stream_config
        })

        # 设置流水线调度
        def schedule_pipeline(self):
            with torch.cuda.stream(stream_config['prefetch']):
                self._prefetch_next_batch()
                
            with torch.cuda.stream(stream_config['compute']):
                # 主计算流 - 使用优化后的并行度配置
                self.process_current_batch(
                    threads_per_block=self.compute_config['threads_per_block'],
                    blocks_per_grid=self.compute_config['blocks_per_grid'],
                    shared_memory=self.compute_config['shared_memory_size']
                )
                
            with torch.cuda.stream(stream_config['transfer']):
                self._transfer_results()

            # 同步所有流
            torch.cuda.synchronize()
            
            # 收集性能指标
            self.metrics.update({
                'gpu_util': self.get_gpu_utilization(),
                'memory_util': self.get_memory_utilization(),
                'pcie_util': self.get_pcie_utilization()
            })
            
        # 注册优化后的调度器
        self.pipeline_scheduler = schedule_pipeline

    def optimize_for_single_task(self):
        """优化单任务性能"""
        # 获取GPU信息
        device_props = torch.cuda.get_device_properties(self.device)
        gpu_name = device_props.name
        
        # 加载优化配置
        from configs.single_task_perf import SingleTaskConfig
        config = SingleTaskConfig.optimize_for_gpu(gpu_name)
        
        # 应用TF32优化
        if config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 设置CUDA图
        if config.use_cuda_graph:
            self.use_cuda_graph = True
            # 预热以捕获计算图
            dummy_input = torch.randn(1, self.model_config.get('max_position_embeddings', 2048), device=self.device)
            for _ in range(3):
                _ = self.generate(dummy_input)
        
        # 配置流优先级
        self.compute_stream = torch.cuda.Stream(priority=config.compute_priority)
        self.transfer_stream = torch.cuda.Stream(priority=config.transfer_priority)
        
        # 设置内存策略
        if config.pinned_memory:
            torch.cuda.set_pinned_memory_allocator()
        
        # 更新模型配置
        self.memory_config.update({
            'max_gpu_layers': 6,          # 增加GPU常驻层数
            'prefetch_layers': 3,         # 预取3层
            'expert_threshold': config.expert_threshold,
            'max_active_experts': config.expert_parallelism
        })
        
        # 使用优化的FP4实现
        from quantization.fp4_optimized import OptimizedFP4
        self.fp4_engine = OptimizedFP4()
        
        # 设置性能监控
        from utils._utils import PerfMonitor
        self.perf_monitor = PerfMonitor()
        
        def _monitor_performance(self, **metrics):
            """监控性能指标"""
            self.perf_monitor.update_metrics(metrics)
            current_util = self.perf_monitor.get_utilization()
            
            # 如果GPU利用率低于阈值,自动调整配置
            if current_util['gpu'] < 0.8:  # 低于80%
                self.memory_config['max_gpu_layers'] = min(
                    self.memory_config['max_gpu_layers'] + 1,
                    8  # 最大8层
                )
        
        def generate(self, prompt: str, **kwargs):
            """优化后的生成函数"""
            with torch.cuda.stream(self.compute_stream):
                # 记录开始时间
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                
                # 执行生成
                output = super().generate(prompt, **kwargs)
                
                # 记录结束时间和性能指标
                end.record()
                torch.cuda.synchronize()
                
                self._monitor_performance(
                    compute_time=start.elapsed_time(end),
                    gpu_util=torch.cuda.utilization(),
                    mem_util=torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                )
                
                return output

def main():
    # 初始化推理类
    model_path = "deepseek-r1-fp4"
    inferencer = DeepseekInference(model_path)
    
    # 示例输入
    prompt = "写一个快速排序的Python实现。"
    
    try:
        # 生成响应
        response = inferencer.generate(
            prompt=prompt,
            max_length=2048,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        print(f"\n输入: {prompt}")
        print(f"输出: {response}")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()