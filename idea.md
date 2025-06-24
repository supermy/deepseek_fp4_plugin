从 NVIDIA TensorRT-LLM 项目中提取的独立 pip 插件，专注于 DeepSeek V3 模型的 FP4 量化推理。


deepseek-r1-0528-fp4部分权重

model.embed_tokens.weight	[129 280, 7 168]	
BF16

model.layers(4) 		

model.layers.0(4) 		

model.layers.0.input_layernorm.weight	[7 168]	
BF16

model.layers.0.mlp(3) 		

model.layers.0.mlp.down_proj(4) 		

model.layers.0.mlp.down_proj.input_scale	[]	
F32

model.layers.0.mlp.down_proj.weight	[7 168, 9 216]	
U8

model.layers.0.mlp.down_proj.weight_scale	[7 168, 1 152]	
F8_E4M3

model.layers.0.mlp.down_proj.weight_scale_2	[]	
F32

model.layers.0.mlp.gate_proj(4) 		

model.layers.0.mlp.gate_proj.input_scale	[]	
F32

model.layers.0.mlp.gate_proj.weight	[18 432, 3 584]	
U8

model.layers.0.mlp.gate_proj.weight_scale	[18 432, 448]	
F8_E4M3

model.layers.0.mlp.gate_proj.weight_scale_2	[]	
F32

model.layers.0.mlp.up_proj(4) 		

model.layers.0.mlp.up_proj.input_scale	[]	
F32

model.layers.0.mlp.up_proj.weight	[18 432, 3 584]	
U8

model.layers.0.mlp.up_proj.weight_scale	[18 432, 448]	
F8_E4M3

model.layers.0.mlp.up_proj.weight_scale_2	[]	
F32

model.layers.0.post_attention_layernorm.weight	[7 168]	
BF16

model.layers.0.self_attn(9) 		

model.layers.0.self_attn.k_proj.k_scale	[]	
F32

model.layers.0.self_attn.kv_a_layernorm.weight	[512]	
BF16

model.layers.0.self_attn.kv_a_proj_with_mqa.weight	[576, 7 168]	
BF16

model.layers.0.self_attn.kv_b_proj.weight	[32 768, 512]	
BF16

model.layers.0.self_attn.o_proj.weight	[7 168, 16 384]	
BF16

model.layers.0.self_attn.q_a_layernorm.weight	[1 536]	
BF16

model.layers.0.self_attn.q_a_proj.weight	[1 536, 7 168]	
BF16

model.layers.0.self_attn.q_b_proj.weight	[24 576, 1 536]	
BF16

model.layers.0.self_attn.v_proj.v_scale	[]	
F32