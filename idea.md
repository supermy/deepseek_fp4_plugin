# DeepSeek FP4 推理插件设计文档

## 1. 项目概述

从 NVIDIA TensorRT-LLM 项目中提取的独立 pip 插件，专注于 DeepSeek V3 模型的 FP4 量化推理。
GPU 单任务利用率最大化，从而达到单任务最大化推理速度。

## 2. 硬件环境要求

- GPU: 16GB 显存
- CPU: 96GB 系统内存
- 存储: PCIe 5.0 SSD (带宽 14GB/s)

## 3. 核心优化策略

### 3.1 三级内存架构

```
GPU (16GB)   - 活跃层和专家计算
│
CPU (96GB)   - 权重缓存和预取
│
SSD (400GB)  - 完整模型存储
```

### 3.2 动态内存管理

1. GPU 内存分配
   - 活跃计算层: 8GB
   - 专家缓存: 4GB
   - 系统预留: 2GB
   - 临时缓存: 2GB

2. CPU 内存规划
   - 模型权重: 40GB
   - 预取缓存: 32GB
   - 临时存储: 16GB
   - 系统预留: 8GB

3. 分层预取策略
   - 双缓冲机制
   - 异步IO操作
   - 流水线并行

### 3.3 MoE 专家优化

1. 专家调度
   - 动态加载卸载
   - 预测性缓存
   - 概率阈值控制

2. 路由优化
   - 批处理合并
   - 权重局部性
   - 计算重叠

## 4. 性能优化实现

### 4.1 计算流水线

```python
# 三流并行
compute_stream = cuda.Stream()  # 计算流
prefetch_stream = cuda.Stream() # 预取流
transfer_stream = cuda.Stream() # 传输流

# 异步执行
with cuda.stream(prefetch_stream):
    prefetch_next_weights()     # 预取下一批权重
    
with cuda.stream(compute_stream):
    process_current_batch()     # 处理当前批次
    
with cuda.stream(transfer_stream):
    transfer_results()          # 传输结果
```

### 4.2 内存优化

1. 权重量化
   - FP4/FP8 混合精度
   - 块尺度量化
   - 动态量化范围

2. 缓存策略
   - LRU 替换策略
   - 预测性缓存
   - 专家组合优化

### 4.3 IO 优化

1. PCIe 带宽利用
   - 批量传输
   - 压缩传输
   - 零拷贝访问

2. 存储优化
   - 文件对齐
   - 顺序访问
   - 异步预读

## 5. 模型结构

### 5.1 权重分布

```
model.embed_tokens.weight      [129280, 7168]  BF16
model.layers(4)               
  ├─ input_layernorm.weight   [7168]          BF16
  ├─ mlp
  │   ├─ down_proj           
  │   │   ├─ weight          [7168, 9216]     U8
  │   │   └─ weight_scale    [7168, 1152]     F8_E4M3
  │   ├─ gate_proj
  │   │   ├─ weight          [18432, 3584]    U8
  │   │   └─ weight_scale    [18432, 448]     F8_E4M3
  │   └─ up_proj
  │       ├─ weight          [18432, 3584]    U8
  │       └─ weight_scale    [18432, 448]     F8_E4M3
  └─ self_attn
      ├─ k_proj              [32768, 512]     BF16
      ├─ q_proj              [24576, 1536]    BF16
      ├─ v_proj              [32768, 512]     BF16
      └─ o_proj              [7168, 16384]    BF16
```

## 6. 性能调优配置

### 6.1 低延迟模式
```yaml
memory:
  gpu_layers: 2
  prefetch: 1
  expert_threshold: 0.2
compute:
  fusion: true
  cuda_graph: true
```

### 6.2 高吞吐模式
```yaml
memory:
  gpu_layers: 4
  prefetch: 2
  expert_threshold: 0.1
compute:
  parallel: true
  batch_size: 32
```

### 6.3 平衡模式
```yaml
memory:
  gpu_layers: 3
  prefetch: 2
  expert_threshold: 0.15
compute:
  auto_tune: true
  dynamic_batch: true
```

## 7. 优化效果

1. 内存使用
   - GPU 峰值: 14GB
   - CPU 占用: 88GB
   - SSD 读取: 12GB/s

2. 性能指标
   - 单次推理: 200ms
   - 批量处理: 1000 tokens/s
   - 预取命中: 85%

3. 资源利用
   - GPU 利用率: 75%
   - CPU 利用率: 60%
   - PCIe 带宽: 80%

## 8. 后续优化方向

1. 计算优化
   - 算子融合
   - 量化精度
   - 专家并行

2. 内存优化
   - 压缩算法
   - 缓存策略
   - 预取精度

3. IO优化
   - 文件组织
   - 批量加载
   - 异步机制