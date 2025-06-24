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
    def test_pad_up(self):
        self.assertEqual(pad_up(10, 4), 12)
        self.assertEqual(pad_up(8, 4), 8)
        self.assertEqual(pad_up(17, 8), 24)
        self.assertEqual(pad_up(1, 128), 128)

class TestRMSNorm(unittest.TestCase):
    def test_rms_norm(self):
        input_tensor = torch.randn(2, 4, 10)
        rms_norm_layer = RMSNorm(dim=10)
        output = rms_norm_layer(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)
        # Add more specific assertions if there's a reference implementation or expected values

class TestLinear(unittest.TestCase):
    def test_linear_no_quant(self):
        in_features = 32
        out_features = 64
        input_tensor = torch.randn(2, 16, in_features)
        linear_layer = Linear(in_features, out_features, bias=False, quant_mode=QuantConfig(QuantMode.NONE))
        output = linear_layer(input_tensor)
        self.assertEqual(output.shape, (2, 16, out_features))

    def test_linear_fp4_quant(self):
        # This test requires more setup for actual FP4 data and kernels
        # For now, we'll just check if it initializes without error and processes shape correctly.
        in_features = 32
        out_features = 64
        input_tensor = torch.randn(2, 16, in_features)
        # Dummy quant config for FP4
        quant_config = QuantConfig(QuantMode.from_int(QuantMode.INT4_AWQ.value | QuantMode.FP8_KV_CACHE.value))
        linear_layer = Linear(in_features, out_features, bias=False, quant_mode=quant_config)
        output = linear_layer(input_tensor)
        self.assertEqual(output.shape, (2, 16, out_features))

class TestGatedMLP(unittest.TestCase):
    def test_gated_mlp(self):
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
    def test_fused_moe(self):
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
    def test_embedding(self):
        vocab_size = 1000
        hidden_size = 512
        input_ids = torch.randint(0, vocab_size, (2, 10))
        embedding_layer = Embedding(vocab_size, hidden_size)
        output = embedding_layer(input_ids)
        self.assertEqual(output.shape, (2, 10, hidden_size))

class TestLMHead(unittest.TestCase):
    def test_lm_head(self):
        hidden_size = 512
        vocab_size = 1000
        input_tensor = torch.randn(2, 10, hidden_size)
        lm_head_layer = LMHead(hidden_size, vocab_size)
        output = lm_head_layer(input_tensor)
        self.assertEqual(output.shape, (2, 10, vocab_size))

if __name__ == '__main__':
    unittest.main() 