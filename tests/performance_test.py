import torch
import time
import unittest

from deepseek_fp4_plugin.modules.linear import Linear
from deepseek_fp4_plugin.modules.gated_mlp import GatedMLP
from deepseek_fp4_plugin.modules.fused_moe import FusedMoE
from deepseek_fp4_plugin.configs.model_configs import QuantConfig, QuantMode, MoeConfig

def benchmark_cuda_op(op, input_tensor, num_runs=100, num_warmup=10):
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU performance test.")
        return 0.0

    torch.cuda.synchronize()
    for _ in range(num_warmup):
        _ = op(input_tensor)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_runs):
        _ = op(input_tensor)
    end_event.record()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

class TestPerformance(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_linear_performance(self):
        print("\n--- Benchmarking Linear Layer ---")
        in_features = 4096
        out_features = 4096
        batch_size = 1
        seq_len = 128

        input_tensor = torch.randn(batch_size, seq_len, in_features, device='cuda')

        # Test non-quantized Linear
        linear_layer_no_quant = Linear(in_features, out_features, bias=False, quant_mode=QuantConfig(QuantMode.NONE)).cuda()
        time_no_quant = benchmark_cuda_op(linear_layer_no_quant, input_tensor)
        print(f"Linear (No Quant): {time_no_quant:.4f} ms/run")

        # Test FP4 quantized Linear
        quant_config_fp4 = QuantConfig(QuantMode.from_int(QuantMode.INT4_AWQ.value | QuantMode.FP8_KV_CACHE.value))
        linear_layer_fp4_quant = Linear(in_features, out_features, bias=False, quant_mode=quant_config_fp4).cuda()
        # For a real FP4 test, weights would need to be properly quantized
        # Here, we only test the forward pass with the initialized layer.
        time_fp4_quant = benchmark_cuda_op(linear_layer_fp4_quant, input_tensor)
        print(f"Linear (FP4 Quant): {time_fp4_quant:.4f} ms/run")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gated_mlp_performance(self):
        print("\n--- Benchmarking GatedMLP Layer ---")
        hidden_size = 4096
        ffn_hidden_size = 16384  # 4 * hidden_size
        num_tokens = 128

        moe_config = MoeConfig(num_experts=1, top_k=1, top_k_mode=0, normalize_expert_output=False)
        quant_config = QuantConfig(QuantMode.NONE)

        mlp = GatedMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            moe_config=moe_config,
            quant_config=quant_config
        ).cuda()

        input_tensor = torch.randn(num_tokens, hidden_size, device='cuda')
        time_gated_mlp = benchmark_cuda_op(mlp, input_tensor)
        print(f"GatedMLP: {time_gated_mlp:.4f} ms/run")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_fused_moe_performance(self):
        print("\n--- Benchmarking FusedMoE Layer ---")
        hidden_size = 4096
        ffn_hidden_size = 16384
        num_tokens = 128
        num_experts = 8
        top_k = 2

        moe_config = MoeConfig(num_experts=num_experts, top_k=top_k, top_k_mode=0, normalize_expert_output=False)
        quant_config = QuantConfig(QuantMode.NONE)

        fused_moe_layer = FusedMoE(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            moe_config=moe_config,
            quant_config=quant_config
        ).cuda()

        input_tensor = torch.randn(num_tokens, hidden_size, device='cuda')
        time_fused_moe = benchmark_cuda_op(fused_moe_layer, input_tensor)
        print(f"FusedMoE: {time_fused_moe:.4f} ms/run")

if __name__ == '__main__':
    unittest.main() 