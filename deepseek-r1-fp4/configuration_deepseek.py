# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/main/configuration_deepseek.py
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

DEEPSEEK_PRETRAINED_CONFIG_ARCHIVE_MAP = {}
class DeepseekV3Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`DeepseekV3Model`].
    这是用于存储 [`DeepseekV3Model`] 配置的配置类。

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DeepseekV3Model`]
            Deep 模型的词汇表大小。定义了调用 [`DeepseekV3Model`] 时传入的 `inputs_ids` 可以表示的不同 token 的数量

        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
            隐藏表示的维度

        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
            MLP 表示的维度

        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE representations.
            MoE 表示的维度

        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
            Transformer 解码器中的隐藏层数量

        num_nextn_predict_layers (`int`, *optional*, defaults to 1):
            Number of nextn predict layers in the DeepSeekV3 Model.
            DeepSeekV3 模型中 nextn 预测层的数量

        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
            Transformer 解码器中每个注意力层的注意力头数量

        n_shared_experts (`int`, *optional*, defaults to None):
            Number of shared experts, None means dense model.
            共享专家的数量，None 表示密集模型

        n_routed_experts (`int`, *optional*, defaults to None):
            Number of routed experts, None means dense model.
            路由专家的数量，None 表示密集模型

        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts.
            路由专家的缩放因子

        topk_method (`str`, *optional*, defaults to `gready`):
            Method of selecting top-k experts.
            选择 top-k 专家的方法

        n_group (`int`, *optional*, defaults to None):
            Number of groups for routed experts.
            路由专家的分组数量

        topk_group (`int`, *optional*, defaults to None):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
            每个 token 选择的组数量(确保每个 token 选择的专家仅在 `topk_group` 组内)

        num_experts_per_tok (`int`, *optional*, defaults to None):
            Number of selected experts, None means dense model.
            选择的专家数量，None 表示密集模型

        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
            MoE 层的频率：每 `moe_layer_freq - 1` 个密集层有一个专家层

        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
            浅层中的密集层数量(embed->dense->dense->...->dense->moe->moe...->lm_head)

        norm_topk_prob (`bool`, *optional*, defaults to False):
            Whether to normalize the weights of the routed experts.
            是否对路由专家的权重进行归一化

        scoring_func (`str`, *optional*, defaults to 'softmax'):
            Method of computing expert weights.
            计算专家权重的方法

        attention_bias (`bool`, defaults to `False`, *optional*):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
            是否在自注意力期间在查询、键、值和输出投影层中使用偏置
            
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
            注意力概率的丢弃率
    """
    model_type = "deepseek_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        moe_intermediate_size = 2048,
        num_hidden_layers=61,
        num_nextn_predict_layers=1,
        num_attention_heads=128,
        num_key_value_heads=128,
        n_shared_experts = 1,
        n_routed_experts = 256,
        ep_size = 1,
        routed_scaling_factor = 2.5,
        kv_lora_rank = 512,
        q_lora_rank = 1536,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        topk_method = 'noaux_tc',
        n_group = 8,
        topk_group = 4,
        num_experts_per_tok = 8,
        moe_layer_freq = 1,
        first_k_dense_replace = 3,
        norm_topk_prob = True,
        scoring_func = 'sigmoid',
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )