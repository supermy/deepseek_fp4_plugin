from dataclasses import dataclass, field
from enum import IntFlag, auto
import json
from pathlib import Path
from typing import Dict, Generic, List, Optional, TypeVar, Union

import torch
import transformers
from strenum import StrEnum


class QuantAlgo(StrEnum):
    W8A16 = auto()
    W4A16 = auto()
    W4A16_AWQ = auto()
    W4A8_AWQ = auto()
    W8A16_GPTQ = auto()
    W4A16_GPTQ = auto()
    W8A8_SQ_PER_CHANNEL = auto()
    W8A8_SQ_PER_TENSOR_PLUGIN = auto()
    W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN = auto()
    W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN = auto()
    W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN = auto()
    W4A8_QSERVE_PER_GROUP = auto()
    W4A8_QSERVE_PER_CHANNEL = auto()
    FP8 = auto()
    FP8_PER_CHANNEL_PER_TOKEN = auto()
    FP8_BLOCK_SCALES = auto()
    INT8 = auto()
    MIXED_PRECISION = auto()
    NVFP4 = auto()
    NO_QUANT = auto()


class QuantMode(IntFlag):
    """
    Quantization mode flags for different quantization methods and options.
    不同量化方法和选项的量化模式标志。
    """
    INT4_WEIGHTS = auto()      # 4-bit integer weights quantization
                              # 4位整数权重量化
    INT8_WEIGHTS = auto()      # 8-bit integer weights quantization
                              # 8位整数权重量化
    ACTIVATIONS = auto()       # Activations quantization
                              # 激活值量化
    PER_CHANNEL = auto()       # Per-channel quantization
                              # 按通道量化
    PER_TOKEN = auto()         # Per-token quantization
                              # 按token量化
    PER_GROUP = auto()         # Per-group quantization
                              # 按组量化
    INT8_KV_CACHE = auto()     # 8-bit integer KV cache quantization
                              # 8位整数KV缓存量化
    FP8_KV_CACHE = auto()      # 8-bit floating point KV cache quantization
                              # 8位浮点KV缓存量化
    FP8_QDQ = auto()          # FP8 quantization-dequantization
                              # FP8量化反量化
    FP8_ROWWISE = auto()       # Row-wise FP8 quantization
                              # 按行FP8量化
    FP8_1x128_128x128 = auto() # FP8 block matrix multiplication
                               # FP8块矩阵乘法
    W4A8_QSERVE = auto()       # Weight 4-bit, Activation 8-bit quantization
                              # 权重4位、激活8位量化
    NVFP4 = auto()             # NVIDIA FP4 quantization
                              # NVIDIA FP4量化
    NVFP4_KV_CACHE = auto()    # NVIDIA FP4 KV cache quantization
                              # NVIDIA FP4 KV缓存量化
    COUNT = auto()             # Counter for number of modes
                              # 模式计数器
    WEIGHTS_AND_ACTIVATIONS = INT4_WEIGHTS | INT8_WEIGHTS | ACTIVATIONS  # Combined weights and activations quantization
                                                                        # 组合权重和激活值量化
    VALID_FLAGS = COUNT - 1    # Mask for valid quantization flags
                              # 有效量化标志掩码

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the quantization mode.
        创建量化模式的深拷贝。
        """
        return QuantMode(self.value)

    def _all(self, bits, mask=VALID_FLAGS):
        """
        Check if all specified bits are set in the mode.
        检查指定的所有位是否在模式中被设置。

        Args:
            bits: The bits to check
                 要检查的位
            mask: Optional mask to apply
                 可选的应用掩码
        """
        return (self & mask & bits) == bits

    def _any(self, bits):
        """
        Check if any of the specified bits are set in the mode.
        检查指定的任何位是否在模式中被设置。

        Args:
            bits: The bits to check
                 要检查的位
        """
        return (self & bits) != 0

    def is_int8_weight_only(self):
        return self._all(self.INT8_WEIGHTS, self.WEIGHTS_AND_ACTIVATIONS)

    def is_int4_weight_only(self):
        return self._all(self.INT4_WEIGHTS, self.WEIGHTS_AND_ACTIVATIONS)

    def is_weight_only(self):
        return self.is_int4_weight_only() or self.is_int8_weight_only()

    def is_int8_weight_only_per_group(self):
        return self.is_int8_weight_only() and self._any(self.PER_GROUP)

    def is_qserve_w4a8(self):
        return self._any(self.W4A8_QSERVE)

    def is_int4_weight_only_per_group(self):
        return self.is_int4_weight_only() and self._any(self.PER_GROUP)

    def has_act_and_weight_quant(self):
        return self._all(self.INT8_WEIGHTS | self.ACTIVATIONS,\
                         self.WEIGHTS_AND_ACTIVATIONS)

    def has_act_or_weight_quant(self):
        return self._any(self.INT4_WEIGHTS | self.INT8_WEIGHTS\
                         | self.ACTIVATIONS)

    def has_per_token_dynamic_scaling(self):
        return self._any(self.PER_TOKEN)

    def has_fp8_block_scales(self):
        return self._any(self.FP8_1x128_128x128)

    def has_act_static_scaling(self):
        return not self.has_per_token_dynamic_scaling() and not self.has_fp8_rowwise()

    def has_per_channel_scaling(self):
        return self._any(self.PER_CHANNEL)

    def has_per_group_scaling(self):
        return self._any(self.PER_GROUP)

    def has_int8_kv_cache(self):
        return self._any(self.INT8_KV_CACHE)

    def has_fp8_kv_cache(self):
        return self._any(self.FP8_KV_CACHE)

    def has_fp4_kv_cache(self):
        return self._any(self.NVFP4_KV_CACHE)

    def has_kv_cache_quant(self):
        return (self.has_int8_kv_cache() or self.has_fp8_kv_cache()\
                or self.has_fp4_kv_cache())

    def has_fp8_qdq(self):
        return self._any(self.FP8_QDQ)

    def has_fp8_rowwise(self):
        return self._any(self.FP8_ROWWISE)

    def has_nvfp4(self):
        return self._any(self.NVFP4)

    def has_weight_quant(self):
        return self._any(self.INT4_WEIGHTS | self.INT8_WEIGHTS)

    def has_any_quant(self, exclude_kv_cache: bool = False):
        has_quant = self._any(self.INT4_WEIGHTS\
                              | self.INT8_WEIGHTS\
                              | self.ACTIVATIONS\
                              | self.FP8_QDQ | self.FP8_ROWWISE\
                              | self.W4A8_QSERVE\
                              | self.FP8_1x128_128x128\
                              | self.NVFP4)
        if exclude_kv_cache:
            return has_quant

        return has_quant | self._any(self.INT8_KV_CACHE | self.FP8_KV_CACHE\
                                     | self.NVFP4_KV_CACHE)

    def set_int8_kv_cache(self):
        return self | self.INT8_KV_CACHE

    def set_fp8_kv_cache(self):
        return self | self.FP8_KV_CACHE

    def set_fp4_kv_cache(self):
        return self | self.NVFP4_KV_CACHE

    def set_fp8_qdq(self):
        return self | self.FP8_QDQ

    def set_fp8_rowwise(self):
        return self | self.FP8_ROWWISE | self.PER_TOKEN | self.PER_CHANNEL

    @staticmethod
    def from_description(quantize_weights=False,\
                         quantize_activations=False,\
                         per_token=False,\
                         per_channel=False,\
                         per_group=False,\
                         use_int4_weights=False,\
                         use_int8_kv_cache=False,\
                         use_fp8_kv_cache=False,\
                         use_fp8_qdq=False,\
                         use_fp8_block_scales=False,\
                         use_fp8_rowwise=False,\
                         use_nvfp4=False,\
                         use_w4a8_qserve=False):

        def raise_error():
            raise ValueError(f\"Unsupported combination of QuantMode args: \"\
                             f\"{quantize_weights=}, \"\
                             f\"{quantize_activations=}, \"\
                             f\"{per_token=}, \"\
                             f\"{per_channel=}, \"\
                             f\"{per_group=}, \"\
                             f\"{use_int4_weights=}, \"\n                             f\"{use_int8_kv_cache=}, \"\n                             f\"{use_fp8_kv_cache=}, \"\n                             f\"{use_fp8_qdq=}, \"\n                             f\"{use_fp8_block_scales=}, \"\n                             f\"{use_fp8_rowwise=}, \"\n                             f\"{use_nvfp4=}, \"\n                             f\"{use_w4a8_qserve=}\")

        # We must quantize weights when we quantize activations.
        if quantize_activations and not quantize_weights:
            raise_error()

        # If we set per_token or per_channel, we must quantize both weights and activations.
        if (per_token or per_channel) and not (quantize_weights\
                                               and quantize_activations):
            raise_error()

        mode = QuantMode(0)

        # Do we quantize the weights - if so, do we use INT4 or INT8?
        if quantize_weights and use_int4_weights:
            mode |= QuantMode.INT4_WEIGHTS
        elif quantize_weights:
            mode |= QuantMode.INT8_WEIGHTS

        if quantize_activations:
            mode |= QuantMode.ACTIVATIONS

        if per_token:
            mode |= QuantMode.PER_TOKEN

        if per_channel:
            mode |= QuantMode.PER_CHANNEL

        if per_group:
            mode |= QuantMode.PER_GROUP

        if use_int8_kv_cache:
            mode |= QuantMode.INT8_KV_CACHE

        if use_fp8_kv_cache:
            mode |= QuantMode.FP8_KV_CACHE

        if use_fp8_qdq:
            mode |= QuantMode.FP8_QDQ

        if use_fp8_block_scales:
            mode |= QuantMode.FP8_1x128_128x128

        if use_fp8_rowwise:
            mode |= QuantMode.FP8_ROWWISE

        if use_nvfp4:
            mode |= QuantMode.NVFP4

        if use_w4a8_qserve:
            mode |= QuantMode.W4A8_QSERVE

        return mode

    @staticmethod
    def use_smooth_quant(per_token=False, per_channel=False):\
        return (per_token or per_channel)

    @staticmethod
    def use_qserve(per_group):\
        return per_group

    @staticmethod
    def use_weight_only(use_int4_weights=False, per_group=False):\
        return (use_int4_weights or not per_group)

    @staticmethod
    def from_quant_algo(
        quant_algo: Optional[QuantAlgo] = None,
        kv_cache_quant_algo: Optional[QuantAlgo] = None,
    ) -> "QuantMode":
        if quant_algo == QuantAlgo.NO_QUANT:
            return QuantMode(0)

        quantize_weights = False
        quantize_activations = False
        per_token = False
        per_channel = False
        per_group = False
        use_int4_weights = False
        use_fp8_qdq = False
        use_fp8_block_scales = False
        use_fp8_rowwise = False
        use_nvfp4 = False
        use_w4a8_qserve = False

        if quant_algo == QuantAlgo.W8A16:
            quantize_weights = True
        elif quant_algo == QuantAlgo.W4A16:
            quantize_weights = True
            use_int4_weights = True
        elif quant_algo == QuantAlgo.W4A16_AWQ:
            quantize_weights = True
            use_int4_weights = True
            per_channel = True
        elif quant_algo == QuantAlgo.W4A8_AWQ:
            quantize_weights = True
            quantize_activations = True
            use_int4_weights = True
            per_channel = True
        elif quant_algo == QuantAlgo.W8A16_GPTQ:
            quantize_weights = True
            per_group = True
        elif quant_algo == QuantAlgo.W4A16_GPTQ:
            quantize_weights = True
            use_int4_weights = True
            per_group = True
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_CHANNEL:
            quantize_weights = True
            quantize_activations = True
            per_channel = True
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN:
            quantize_weights = True
            quantize_activations = True
            per_token = True
            per_channel = False  # The plugin doesn't use per-channel for activations.
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN:
            quantize_weights = True
            quantize_activations = True
            per_token = True
            per_channel = True
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN:
            quantize_weights = True
            quantize_activations = True
            per_token = False
            per_channel = True
        elif quant_algo == QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN:
            quantize_weights = True
            quantize_activations = True
            per_token = True
            per_channel = False  # The plugin doesn't use per-channel for activations.
        elif quant_algo == QuantAlgo.W4A8_QSERVE_PER_GROUP:
            quantize_weights = True
            quantize_activations = True
            use_int4_weights = True
            use_w4a8_qserve = True
            per_group = True
        elif quant_algo == QuantAlgo.W4A8_QSERVE_PER_CHANNEL:
            quantize_weights = True
            quantize_activations = True
            use_int4_weights = True
            use_w4a8_qserve = True
            per_channel = True
        elif quant_algo == QuantAlgo.FP8:
            quantize_weights = True
            quantize_activations = True
            use_fp8_qdq = True
        elif quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN:
            quantize_weights = True
            quantize_activations = True
            use_fp8_qdq = True
            per_token = True
            per_channel = True
        elif quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
            quantize_weights = True
            quantize_activations = True
            use_fp8_block_scales = True
        elif quant_algo == QuantAlgo.NVFP4:
            quantize_weights = True
            use_nvfp4 = True
        elif quant_algo is not None:
            raise ValueError(f\"Quantization algorithm not supported: {quant_algo}\")

        use_int8_kv_cache = False
        use_fp8_kv_cache = False
        use_nvfp4_kv_cache = False
        if kv_cache_quant_algo == QuantAlgo.INT8:
            use_int8_kv_cache = True
        elif kv_cache_quant_algo == QuantAlgo.FP8:
            use_fp8_kv_cache = True
        elif kv_cache_quant_algo == QuantAlgo.NVFP4:
            use_nvfp4_kv_cache = True
        elif kv_cache_quant_algo is not None and kv_cache_quant_algo != QuantAlgo.NO_QUANT:
            raise ValueError(
                f\"KV Cache Quantization algorithm not supported: {kv_cache_quant_algo}\")

        return QuantMode.from_description(
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            per_token=per_token,
            per_channel=per_channel,
            per_group=per_group,
            use_int4_weights=use_int4_weights,
            use_int8_kv_cache=use_int8_kv_cache,
            use_fp8_kv_cache=use_fp8_kv_cache,
            use_fp8_qdq=use_fp8_qdq,
            use_fp8_block_scales=use_fp8_block_scales,
            use_fp8_rowwise=use_fp8_rowwise,
            use_nvfp4=use_nvfp4,
            use_w4a8_qserve=use_w4a8_qserve,
            )

    def to_dict(self):
        return {
            \"quantize_weights\": self.has_weight_quant(),
            \"quantize_activations\": self._any(QuantMode.ACTIVATIONS),
            \"per_token\": self._any(QuantMode.PER_TOKEN),
            \"per_channel\": self._any(QuantMode.PER_CHANNEL),
            \"per_group\": self._any(QuantMode.PER_GROUP),
            \"use_int4_weights\": self._any(QuantMode.INT4_WEIGHTS),
            \"use_int8_kv_cache\": self._any(QuantMode.INT8_KV_CACHE),
            \"use_fp8_kv_cache\": self._any(QuantMode.FP8_KV_CACHE),
            \"use_fp8_qdq\": self._any(QuantMode.FP8_QDQ),
            \"use_fp8_block_scales\": self._any(QuantMode.FP8_1x128_128x128),
            \"use_fp8_rowwise\": self._any(QuantMode.FP8_ROWWISE),
            \"use_nvfp4\": self._any(QuantMode.NVFP4),
            \"use_w4a8_qserve\": self._any(QuantMode.W4A8_QSERVE),\
        }


@dataclass
class QuantConfig:
    quant_algo: Optional[QuantAlgo] = None
    kv_cache_quant_algo: Optional[QuantAlgo] = None
    group_size: int = 128
    smoothquant_val: float = 0.5
    clamp_val: Optional[List[float]] = None
    use_meta_recipe: bool = False
    has_zero_point: bool = False
    pre_quant_scale: bool = False
    exclude_modules: Optional[List[str]] = None

    @property
    def quant_mode(self):
        return QuantMode.from_quant_algo(\
            self.quant_algo,\
            self.kv_cache_quant_algo,\
        )

    @property
    def layer_quant_mode(self):
        return QuantMode.from_quant_algo(\
            self.quant_algo,\
            self.kv_cache_quant_algo,\
        )

TConfig = TypeVar(\"TConfig\", bound=transformers.PretrainedConfig)

@dataclass
class MoeLoadBalancerConfig:
    """
    Configuration for MoE load balancing.
    MoE负载均衡的配置类。
    """
    num_slots: Optional[int] = None               # Number of total slots for experts
                                                 # 专家总槽位数
    initial_global_assignments: Optional[Dict[int, List[int]]] = None  # Initial expert assignments per layer
                                                                      # 每层的初始专家分配
    layer_updates_per_iter: int = 0              # Number of layer updates per iteration
                                                # 每次迭代的层更新数量

    num_experts: Optional[int] = field(default=None, init=False)  # Total number of experts
                                                                 # 专家总数
    ep_rank: Optional[int] = field(default=None, init=False)     # Expert parallel rank
                                                                 # 专家并行等级
    ep_size: Optional[int] = field(default=None, init=False)     # Expert parallel size
                                                                 # 专家并行大小

    def setup(self, num_experts: int, ep_rank: int, ep_size: int) -> None:
        """
        Setup the load balancer configuration.
        设置负载均衡器配置。

        Args:
            num_experts (int): Total number of experts
                             专家总数
            ep_rank (int): Expert parallel rank
                          专家并行等级
            ep_size (int): Expert parallel size
                          专家并行大小
        """
        self.num_experts = num_experts
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        if self.num_slots is None:
            self.num_slots = self.num_experts
        assert self.num_slots >= self.num_experts
        assert self.num_slots % self.ep_size == 0

    @property
    def num_local_slots(self) -> int:
        """
        Get number of local slots per expert parallel rank.
        获取每个专家并行等级的本地槽位数。
        """
        return self.num_slots // self.ep_size

    @property
    def slot_start(self) -> int:
        """
        Get starting slot index for current expert parallel rank.
        获取当前专家并行等级的起始槽位索引。
        """
        return self.ep_rank * self.num_local_slots

    @property
    def slot_end(self) -> int:
        """
        Get ending slot index for current expert parallel rank.
        获取当前专家并行等级的结束槽位索引。
        """
        return self.slot_start + self.num_local_slots

    def get_layer_initial_global_assignments(self, layer_idx: int) -> List[int]:
        """
        Get initial global expert assignments for a specific layer.
        获取特定层的初始全局专家分配。

        Args:
            layer_idx (int): Layer index
                           层索引

        Returns:
            List[int]: List of expert assignments
                      专家分配列表
        """
        if self.initial_global_assignments is None:
            # Default round-robin assignment
            # 默认轮询分配方式
            return [(ep_rank * self.num_experts // self.ep_size + i) %\
                    self.num_experts for ep_rank in range(self.ep_size)\
                    for i in range(self.num_local_slots)]
        else:
            assert layer_idx in self.initial_global_assignments
            assert len(self.initial_global_assignments[layer_idx]) == self.num_slots
            assert set(self.initial_global_assignments[layer_idx]) == set(range(self.num_experts))
            return self.initial_global_assignments[layer_idx]

@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    """
    Configuration class for model settings.
    模型设置的配置类。
    """
    pretrained_config: Optional[TConfig] = None          # Configuration from pre-trained model
                                                        # 预训练模型的配置
    mapping: Optional[object] = field(                   # Object for parallel mapping
        default_factory=lambda: None)                    # 并行映射对象
    quant_config: QuantConfig = field(                  # Quantization configuration
        default_factory=QuantConfig)                     # 量化配置
    quant_config_dict: Optional[Dict[str, QuantConfig]] = None  # Dictionary of quantization configs per module
                                                               # 每个模块的量化配置字典
    skip_create_weights_in_init: bool = False           # Whether to skip weight creation in __init__
                                                        # 是否在__init__中跳过权重创建
    is_generation: bool = True                          # Whether model is for generation
                                                        # 是否是用于生成的模型
    max_num_tokens: int = 8192                          # Maximum number of tokens
                                                        # 最大token数量
    moe_max_num_tokens: Optional[int] = None            # Maximum tokens for MoE
                                                        # MoE的最大token数量
    moe_load_balancer: Optional[MoeLoadBalancerConfig] = None  # MoE load balancer config
                                                               # MoE负载均衡器配置

    attn_backend: str = 'TRTLLM'                        # Attention implementation backend
                                                        # 注意力实现后端
    moe_backend: str = 'CUTLASS'                        # MoE implementation backend
                                                        # MoE实现后端

    extra_attrs: Dict = field(default_factory=dict, repr=False, init=False)  # Extra attributes
                                                                            # 额外属性

    def __post_init__(self):
        """
        Post initialization processing.
        初始化后处理。
        """
        if self.pretrained_config and hasattr(self.pretrained_config, "architectures"):
            self.is_generation = self.is_generation_model(
                self.pretrained_config.architectures)

    @property
    def enable_flash_mla(self):
        """
        Check if flash multi-head attention is enabled.
        检查是否启用flash多头注意力。

        Returns:
            bool: Whether flash MLA is enabled
                 是否启用flash MLA
        """
        if self.attn_backend == 'TRTLLM':
            if hasattr(self.pretrained_config, "kv_lora_rank") and hasattr(
                    self.pretrained_config, "qk_rope_head_dim"):
                head_dim = self.pretrained_config.kv_lora_rank + self.pretrained_config.qk_rope_head_dim
                if head_dim == 576 and torch.cuda.get_device_capability() == (9, 0):
                    return True
        return False

    def get_quant_config(self, name: Optional[str] = None) -> QuantConfig:
        """
        Get quantization configuration by name.
        根据名称获取量化配置。

        Args:
            name (Optional[str]): Name of the quantization config
                                量化配置的名称

        Returns:
            QuantConfig: The quantization configuration
                        量化配置

        Raises:
            ValueError: If named config not found
                       如果未找到指定名称的配置
        """
        if name is None or self.quant_config_dict is None:
            return self.quant_config

        if name in self.quant_config_dict:
            return self.quant_config_dict[name]

        raise ValueError(f'quant config of {name} is not found')

    @staticmethod
    def is_generation_model(model_architectures: Optional[List[str]]) -> bool:
        """
        Check if model is for text generation.
        检查模型是否用于文本生成。

        Args:
            model_architectures: List of model architecture names
                               模型架构名称列表

        Returns:
            bool: Whether model is for generation
                 是否是用于生成的模型
        """
        if model_architectures is None:
            return True
        return model_architectures[0] not in [
            "BertForSequenceClassification", "Qwen2ForProcessRewardModel",
            "Qwen2ForRewardModel", "LlamaForTextEmbedding"
        ]

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, trust_remote_code=False, **kwargs):
        """
        Create config from pre-trained model checkpoint.
        从预训练模型检查点创建配置。

        Args:
            checkpoint_dir (str): Path to checkpoint directory
                                检查点目录路径
            trust_remote_code (bool): Whether to trust remote code
                                    是否信任远程代码
            **kwargs: Additional arguments
                     额外参数

        Returns:
            ModelConfig: Initialized model configuration
                        初始化的模型配置
        """
        pretrained_config = transformers.AutoConfig.from_pretrained(\
            checkpoint_dir,\
            trust_remote_code=trust_remote_code,\
        )

        model_dir = Path(\
            transformers.utils.hub.cached_file(checkpoint_dir,\
                                               \'config.json\')).parent
        quant_config = QuantConfig()
        layer_quant_config = None
        quant_config_file = model_dir / \'hf_quant_config.json\'
        if quant_config_file.exists():
            with open(quant_config_file) as f:
                quant_config_dict = json.load(f)

            json_quant_configs = quant_config_dict[\'quantization\']

            quant_config.quant_algo = json_quant_configs.get(\'quant_algo\', None)
            quant_config.kv_cache_quant_algo = json_quant_configs.get(\
                \'kv_cache_quant_algo\', None)
            quant_config.group_size = json_quant_configs.get(\'group_size\', None)
            quant_config.exclude_modules = json_quant_configs.get(\
                \'exclude_modules\', None)

            if quant_config.quant_algo == QuantAlgo.MIXED_PRECISION:
                mixed_quant_config_file = model_dir / \'quant_cfg.json\'
                with open(mixed_quant_config_file) as fm:
                    mixed_quant_configs = json.load(fm)
                    kv_cache_quant_algo = mixed_quant_configs[\
                        \'kv_cache_quant_algo\']
                    mixed_quant_configs = mixed_quant_configs[\
                        \'quantized_layers\']
                    if kv_cache_quant_algo is not None and quant_config.kv_cache_quant_algo is not None:
                        if kv_cache_quant_algo != quant_config.kv_cache_quant_algo:
                            raise RuntimeError(\
                                f\"The kvcache config in \'quant_cfg.json\', {kv_cache_quant_algo},\"\
                                f\"is different from \'hf_quant_config.json\', {quant_config.kv_cache_quant_algo}!\"\
                            )
                    kv_cache_quant_algo = kv_cache_quant_algo or quant_config.kv_cache_quant_algo

                    for layer in mixed_quant_configs:
                        config = QuantConfig()
                        config.kv_cache_quant_algo = kv_cache_quant_algo
                        config.quant_algo = mixed_quant_configs[layer][\
                            \'quant_algo\']
                        config.group_size = mixed_quant_configs[layer].get(\
                            \'group_size\', None)
                        mixed_quant_configs[layer] = config
                layer_quant_config = mixed_quant_configs
            elif quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
                if quant_config.group_size is None:
                    quant_config.group_size = 128

            if kwargs.get(\
                    \'moe_backend\'\
            ) == \'TRTLLM\' and quant_config.quant_algo == \"FP8_BLOCK_SCALES\" and quant_config.exclude_modules is None:
                quant_config.exclude_modules = [\
                    \"*kv_b_proj*\", \"*k_b_proj*\", \"*eh_proj\"\\\
                ]

        elif hasattr(pretrained_config, \"quantization_config\"):\
            hf_quant_config = pretrained_config.quantization_config
            if hf_quant_config.get(\
                    \"quant_method\") == \"fp8\" and hf_quant_config.get(\
                        \"weight_block_size\", []):\
                quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
                if kwargs.get(\'moe_backend\') == \'TRTLLM\':
                    if quant_config.exclude_modules is None:
                        quant_config.exclude_modules = [\\\
                            \"*kv_b_proj*\", \"*k_b_proj*\", \"*eh_proj\"\\\
                        ]\

        return cls(pretrained_config=pretrained_config,\
                   quant_config=quant_config, **kwargs)