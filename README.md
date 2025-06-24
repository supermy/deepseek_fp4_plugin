# DeepSeek FP4 推理插件

这是一个从 TensorRT-LLM 项目中提取并封装的独立 DeepSeek FP4 推理 PyTorch 插件。它旨在提供 DeepSeek 模型 FP4 量化推理所需的核心功能，包括自定义的 CUDA 内核操作。

## 功能

- **DeepSeekV3 模型支持**：包含 DeepSeekV3 模型结构及其 FP4 推理相关的修改。
- **FP4 量化**：支持 FP4 权重的处理和相关的量化实用程序。
- **CUDA 内核集成**：集成了用于 NVFP4 块尺度交错（block scale interleaving）的 CUDA 内核，以加速推理性能。

## 安装

请按照以下步骤安装此插件：

1.  **克隆或下载本插件**：

    ```bash
    git clone https://github.com/yourusername/deepseek_fp4_plugin # 请将 'yourusername' 和 'deepseek_fp4_plugin' 替换为您的实际仓库信息
    cd deepseek_fp4_plugin
    ```

    如果您是从现有项目中提取此目录，请直接导航到 `deepseek_fp4_plugin` 目录。

2.  **创建并激活虚拟环境**（推荐）：

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **安装依赖**：

    ```bash
    pip install -r requirements.txt
    ```

4.  **构建并安装插件**：

    ```bash
    python setup.py install
    # 或者以可编辑模式安装（方便开发）
    # pip install -e .
    ```

## 使用示例

一旦插件安装成功，您就可以在您的 PyTorch 项目中导入并使用它：

```python
import torch
from deepseek_fp4_plugin.models.deepseek_v3 import DeepseekV3ForCausalLM
from deepseek_fp4_plugin.quantization.fp4_utils import shuffle_matrix_sf_a

# 示例：加载模型
# model = DeepseekV3ForCausalLM.from_pretrained("your_model_path")

# 示例：调用 shuffle_matrix_sf_a
# 假设有一个输入张量 input_tensor 和 epilogue_tile_m
input_tensor = torch.randn(1, 1024, dtype=torch.float16, device="cuda") # 示例输入
epilogue_tile_m = 128 # 示例值
shuffled_output = shuffle_matrix_sf_a(input_tensor, epilogue_tile_m)
print("Shuffled Output Shape:", shuffled_output.shape)

print("DeepSeek FP4 Plugin 已成功安装和导入！")

```

## 运行测试

您可以通过以下命令运行单元测试和性能测试，以验证插件的功能和性能：

1.  **运行单元测试**：

    ```bash
    python3 deepseek_fp4_plugin/tests/test_plugin.py
    ```

    这将执行 `test_plugin.py` 中定义的所有单元测试，确保各个模块的功能正确性。

2.  **运行性能测试**：

    ```bash
    python3 deepseek_fp4_plugin/tests/performance_test.py
    ```

    这将执行 `performance_test.py` 中定义的性能测试。请注意，性能测试通常需要在支持 CUDA 的 GPU 环境中运行，否则 GPU 相关的测试可能会被跳过。

## 注意事项

- 本插件是一个从 TensorRT-LLM 中提取的独立模块，因此某些依赖于 TensorRT-LLM 内部复杂组件的功能可能无法完全支持。
- `deepseek_fp4_plugin/quantization/fp4_utils.py` 中的 `pack_int4_weight_col_wise` 函数目前是一个简化实现。要获得完整的 FP4 量化性能，可能需要进一步的 CUDA 内核开发。

## 贡献

如果您希望贡献或扩展此插件的功能（例如，添加自定义 CUDA 内核以支持 `NotImplementedError` 标记的功能），请参考相关的代码部分。

## 许可证

此项目在 Apache 2.0 许可下发布。 