# DeepSeek FP4 推理插件

这是一个从 TensorRT-LLM 项目中提取的独立插件，专门用于 DeepSeek 模型的 FP4 量化推理。本插件针对大规模内存受限场景进行了优化。

## 功能特性

- **高效的内存管理**
  - 动态层卸载：GPU(16GB) ↔ CPU(96GB) ↔ SSD(400GB)
  - 异步预取和流水线并行 
  - 智能专家(MoE)缓存
  - PCIe 5.0 高带宽利用(14GB/s)

- **FP4 量化支持**
  - 支持 fp4/fp8 混合精度
  - 自动量化配置
  - MoE 动态专家路由
  
- **优化算法**
  - 双缓冲预取机制
  - 计算与IO重叠
  - 智能层缓存策略
  - 动态 Batch 处理

## 快速开始

### 环境要求

- Python 3.8+ 
- PyTorch 2.0+
- CUDA 11.8+
- 16GB+ GPU 内存
- 96GB+ 系统内存
- PCIe 5.0 SSD (推荐)

### 安装步骤

1. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

2. 安装依赖
```bash 
pip install -r requirements.txt
```

3. 安装插件
```bash
pip install -e .
```

### 基础使用

```python
from deepseek_fp4_plugin import DeepseekInference

# 初始化推理实例
model = DeepseekInference(
    model_path="deepseek-r1-fp4",
    device="cuda",
    memory_limit=14 # GPU显存限制(GB)
)

# 设置内存配置
model.set_memory_config(
    prefetch_layers=2,    # 预取层数
    max_gpu_layers=4,     # GPU最大层数
    expert_threshold=0.1  # 专家加载阈值
)

# 运行推理
response = model.generate(
    "请介绍一下 NVIDIA 公司",
    max_tokens=1024,
    temperature=0.7
)
print(response)
```

### 高级配置

```python
# 自定义内存策略
model.set_memory_strategy({
    'gpu_reserved': 2,      # GPU保留内存(GB) 
    'cpu_cache_size': 32,   # CPU缓存大小(GB)
    'prefetch_size': 4,     # 预取缓存大小(GB)
    'min_layers': 1,        # 最小保留层数
    'max_experts': 32       # 最大激活专家数
})

# 设置计算流
model.set_compute_streams({
    'prefetch': True,      # 启用预取流
    'compute': True,       # 启用计算流
    'transfer': True       # 启用传输流
})
```

## 性能优化指南 

1. **显存优化**
   - 调整 `max_gpu_layers` 平衡吞吐和延迟
   - 使用 `expert_threshold` 控制专家加载
   - 启用 `prefetch` 减少等待时间

2. **吞吐优化** 
   - 增大 `batch_size` 提高 GPU 利用率
   - 调整 `prefetch_layers` 平衡内存使用
   - 开启多流并行计算

3. **延迟优化**
   - 减小 `max_gpu_layers` 降低切换开销
   - 使用 SSD 预取减少 CPU 内存占用
   - 调整 `expert_threshold` 降低路由开销

## 限制说明

- 需要 PCIe 5.0 SSD 获得最佳性能
- 建议系统内存≥96GB 以支持大模型
- 仅支持 NVIDIA GPU (需 Ampere 及以上架构)
- 目前仅支持 DeepSeek 系列模型

## 常见问题

1. **内存不足:**
   - 减小 `max_gpu_layers` 和 `prefetch_layers`
   - 增加 SSD 缓存使用
   - 降低 batch size

2. **性能问题:**
   - 检查 PCIe 带宽是否满足要求
   - 优化预取策略和缓存配置
   - 调整计算流和内存策略

3. **精度损失:**
   - 适当增加保留在 GPU 中的层数
   - 调整专家路由阈值
   - 使用混合精度训练

## 协议说明

本项目采用 MIT 协议开源。详见 LICENSE 文件。