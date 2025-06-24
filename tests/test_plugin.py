import unittest
import torch

from deepseek_fp4_plugin.quantization.fp4_utils import pad_up
from deepseek_fp4_plugin.modules.rms_norm import RMSNorm
from deepseek_fp4_plugin.modules.linear import Linear
from deepseek_fp4_plugin.configs.model_configs import QuantConfig, QuantMode, MoeConfig
from deepseek_fp4_plugin.modules.gated_mlp import GatedMLP
from deepseek_fp4_plugin.modules.fused_moe import FusedMoE
from deepseek_fp4_plugin.configs.attention_configs import AttentionInputType, AttentionMetadata, AttentionRuntimeFeatures, PositionEmbeddingType, RotaryScalingType
from deepseek_fp4_plugin.modules.embedding import Embedding, LMHead

class TestFP4Utils(unittest.TestCase):
    """
    Test utilities for FP4 operations.
    FP4 操作的测试工具。
    """
    def test_pad_up(self):
        """
        Test the pad_up function that aligns sizes to multiples.
        测试将大小对齐到倍数的 pad_up 函数。
        """
        self.assertEqual(pad_up(10, 4), 12)
        self.assertEqual(pad_up(8, 4), 8)
        self.assertEqual(pad_up(17, 8), 24)
        self.assertEqual(pad_up(1, 128), 128)

class TestRMSNorm(unittest.TestCase):
    """
    Test cases for Root Mean Square Normalization.
    均方根归一化的测试用例。
    """
    def test_rms_norm(self):
        """
        Test RMSNorm layer's basic functionality.
        测试 RMSNorm 层的基本功能。
        """
        input_tensor = torch.randn(2, 4, 10)
        rms_norm_layer = RMSNorm(dim=10)
        output = rms_norm_layer(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

class TestLinear(unittest.TestCase):
    """
    Test cases for Linear transformation with quantization support.
    支持量化的线性变换测试用例。
    """
    def test_linear_no_quant(self):
        """
        Test linear layer without quantization.
        测试不带量化的线性层。
        """
        in_features = 32
        out_features = 64
        input_tensor = torch.randn(2, 16, in_features)
        linear_layer = Linear(in_features, out_features, bias=False, quant_mode=QuantConfig(QuantMode.NONE))
        output = linear_layer(input_tensor)
        self.assertEqual(output.shape, (2, 16, out_features))

    def test_linear_fp4_quant(self):
        """
        Test linear layer with FP4 quantization.
        测试带 FP4 量化的线性层。
        """
        in_features = 32
        out_features = 64
        input_tensor = torch.randn(2, 16, in_features)
        quant_config = QuantConfig(QuantMode.from_int(QuantMode.INT4_AWQ.value | QuantMode.FP8_KV_CACHE.value))
        linear_layer = Linear(in_features, out_features, bias=False, quant_mode=quant_config)
        output = linear_layer(input_tensor)
        self.assertEqual(output.shape, (2, 16, out_features))

class TestGatedMLP(unittest.TestCase):
    """
    Test cases for Gated MLP module.
    门控 MLP 模块的测试用例。
    """
    def test_gated_mlp(self):
        """
        Test basic functionality of GatedMLP.
        测试 GatedMLP 的基本功能。
        """
        hidden_size = 1024
        ffn_hidden_size = 4096
        num_tokens = 32
        moe_config = MoeConfig(num_experts=1, top_k=1, top_k_mode=0, normalize_expert_output=False)
        quant_config = QuantConfig(QuantMode.NONE)

        mlp = GatedMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            moe_config=moe_config,
            quant_config=quant_config
        )
        input_tensor = torch.randn(num_tokens, hidden_size)
        output = mlp(input_tensor)
        self.assertEqual(output.shape, (num_tokens, hidden_size))

class TestFusedMoE(unittest.TestCase):
    """
    Test cases for FusedMoE module.
    FusedMoE 模块的测试用例。
    """

    def test_fused_moe(self):
        """
        Test the basic functionality of FusedMoE.
        测试 FusedMoE 的基本功能。

        This test:
        1. Creates a FusedMoE instance with sample configuration
           使用示例配置创建 FusedMoE 实例
        2. Runs forward pass with random input
           使用随机输入运行前向传播
        3. Verifies output shape and basic properties
           验证输出形状和基本属性
        """
        hidden_size = 1024
        ffn_hidden_size = 4096
        num_tokens = 32
        num_experts = 8
        top_k = 2

        moe_config = MoeConfig(num_experts=num_experts, top_k=top_k, top_k_mode=0, normalize_expert_output=False)
        quant_config = QuantConfig(QuantMode.NONE)

        fused_moe_layer = FusedMoE(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            moe_config=moe_config,
            quant_config=quant_config
        )
        input_tensor = torch.randn(num_tokens, hidden_size)
        output = fused_moe_layer(input_tensor)
        self.assertEqual(output.shape, (num_tokens, hidden_size))

class TestEmbedding(unittest.TestCase):
    """
    Test cases for Embedding layer.
    嵌入层的测试用例。
    """
    def test_embedding(self):
        """
        Test basic functionality of Embedding layer.
        测试嵌入层的基本功能。
        """
        vocab_size = 1000
        hidden_size = 512
        input_ids = torch.randint(0, vocab_size, (2, 10))
        embedding_layer = Embedding(vocab_size, hidden_size)
        output = embedding_layer(input_ids)
        self.assertEqual(output.shape, (2, 10, hidden_size))

class TestLMHead(unittest.TestCase):
    """
    Test cases for Language Model Head.
    语言模型头部的测试用例。
    """
    def test_lm_head(self):
        """
        Test basic functionality of Language Model Head.
        测试语言模型头部的基本功能。
        """
        hidden_size = 512
        vocab_size = 1000
        input_tensor = torch.randn(2, 10, hidden_size)
        lm_head_layer = LMHead(hidden_size, vocab_size)
        output = lm_head_layer(input_tensor)
        self.assertEqual(output.shape, (2, 10, vocab_size))

if __name__ == '__main__':
    unittest.main()