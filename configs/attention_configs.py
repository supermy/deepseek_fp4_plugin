from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import List, Optional

import torch

class RotaryScalingType(IntEnum):
    """
    Enum for different types of rotary position embedding scaling.
    旋转位置嵌入缩放的不同类型枚举。
    """
    none = auto()      # No scaling 不缩放
    linear = auto()    # Linear scaling 线性缩放
    dynamic = auto()   # Dynamic scaling 动态缩放
    yarn = auto()      # YaRN scaling YaRN缩放
    llama3 = auto()    # LLaMA-3 scaling LLaMA-3缩放
    mrope = auto()     # mRoPE scaling mRoPE缩放

class PositionEmbeddingType(IntEnum):
    """
    Enum for different types of position embeddings.
    位置嵌入的不同类型枚举。
    """
    learned_absolute = auto()    # Learned absolute position embeddings 学习的绝对位置嵌入
    rope_gptj = auto()          # RoPE (GPT-J style) GPT-J风格的RoPE
    rope_gpt_neox = auto()      # RoPE (GPT-NeoX style) GPT-NeoX风格的RoPE
    alibi = auto()              # ALiBi position embeddings ALiBi位置嵌入
    alibi_with_scale = auto()   # ALiBi with scaling ALiBi带缩放
    relative = auto()           # Relative position embeddings 相对位置嵌入
    chatglm = auto()            # ChatGLM position embeddings ChatGLM位置嵌入
    yarn = auto()               # YaRN position embeddings YaRN位置嵌入
    mrope = auto()              # mRoPE position embeddings mRoPE位置嵌入
    deferred = auto()           # Deferred position embeddings 延迟位置嵌入

@dataclass
class AttentionRuntimeFeatures:
    """
    Runtime features for attention computation.
    注意力计算的运行时特性。
    """
    chunked_prefill: bool = False      # Whether to use chunked prefill
                                      # 是否使用分块预填充
    cache_reuse: bool = False          # Whether to reuse KV cache
                                      # 是否重用KV缓存
    has_speculative_draft_tokens: bool = False  # Whether has speculative draft tokens
                                               # 是否有推测性草稿token

class AttentionInputType(IntEnum):
    """
    Input types for attention computation.
    注意力计算的输入类型。
    """
    mixed = 0              # Mixed context and generation
                          # 混合上下文和生成
    context_only = 1      # Context only
                          # 仅上下文
    generation_only = 2   # Generation only
                          # 仅生成

@dataclass(kw_only=True)
class AttentionMetadata:
    """
    Metadata for attention computation.
    注意力计算的元数据。
    """
    max_num_requests: int                     # Maximum number of requests
                                            # 最大请求数量
    max_num_tokens: int                      # Maximum number of tokens
                                            # 最大token数量
    kv_cache_manager: object                 # KV cache manager
                                            # KV缓存管理器
    mapping: Optional[object] = None         # Optional mapping object
                                            # 可选的映射对象
    enable_flash_mla: bool = False          # Whether to enable flash MLA
                                            # 是否启用flash MLA
    enable_paged_context_mla: bool = False  # Whether to enable paged context MLA
                                            # 是否启用分页上下文MLA
    is_cuda_graph: bool = field(default=False, repr=False)  # Whether is CUDA graph
                                                           # 是否是CUDA图

    seq_lens: Optional[torch.Tensor]
    num_contexts: int

    position_ids: Optional[torch.Tensor] = None

    _num_contexts: int = field(init=False, default=0, repr=False)
    kv_cache_params: Optional[object] = None # Placeholder for KVCacheParams

    seq_lens_kv: Optional[torch.Tensor]

    _seq_lens: Optional[torch.Tensor] = field(init=False,\
                                              repr=False,\
                                              default=None)
    _seq_lens_kv: Optional[torch.Tensor] = field(init=False,\
                                                 repr=False,\
                                                 default=None)

    _seq_lens_cuda: Optional[torch.Tensor] = field(init=False,\
                                                   repr=False,\
                                                   default=None)
    _seq_lens_kv_cuda: Optional[torch.Tensor] = field(init=False,\
                                                      repr=False,\
                                                      default=None)

    cross: Optional["AttentionMetadata"] = None

    request_ids: Optional[List[int]] = None

    prompt_lens: Optional[List[int]] = None

    runtime_features: AttentionRuntimeFeatures = field(\
        default_factory=AttentionRuntimeFeatures)

    all_rank_num_tokens: Optional[List[int]] = None

    _num_generations: int = field(init=False, default=0, repr=False)
    _num_ctx_tokens: int = field(init=False, default=0, repr=False)
    _num_tokens: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.is_cross:
            assert self.cross is None or self.cross is self, "Cross attention metadata should not have sub metadata"
            self.cross = self
            return

        assert self.cross is None or type(self) is type(\
            self.cross\
        ), "Top level and cross attention sub metadata type mismatched"

    def on_update(self):
        if (self._seq_lens is not None\
                and self._seq_lens.shape[0] >= self.num_contexts\
                and self.num_contexts >= 0):
            self._num_ctx_tokens = self._seq_lens[:self.num_contexts].sum(\
            ).item()
            self._num_generations = self._seq_lens.shape[0] - self.num_contexts
        if self._seq_lens_kv is not None:
            self._num_tokens = self._seq_lens_kv.sum().item()
        elif self._seq_lens is not None:
            self._num_tokens = self._seq_lens.sum().item()

    @property
    def seq_lens(self) -> Optional[torch.Tensor]:
        return self._seq_lens

    @seq_lens.setter
    def seq_lens(self, value: Optional[torch.Tensor]):
        value = value if value is not AttentionMetadata.seq_lens else None
        self._seq_lens = value
        self.on_update()

        if self._seq_lens is not None:
            self._seq_lens = self._seq_lens.pin_memory()

            if self.is_cuda_graph and self._seq_lens_cuda is not None:
                self._seq_lens_cuda.copy_(self._seq_lens, non_blocking=True)
            else:
                self._seq_lens_cuda = self._seq_lens.cuda(non_blocking=True)

        if self.has_cross_sub_metadata:
            self.cross._seq_lens = self._seq_lens
            self.cross._seq_lens_cuda = self._seq_lens_cuda

    @property
    def num_contexts(self) -> int:
        return self._num_contexts

    @num_contexts.setter
    def num_contexts(self, value: int):
        value = value if value is not AttentionMetadata.num_contexts else 0
        self._num_contexts = value
        self.on_update()

    @property
    def num_generations(self) -> int:
        return self._num_generations

    @num_generations.setter
    def num_generations(self, value: int):
        value = value if value is not AttentionMetadata.num_generations else 0
        self._num_generations = value
        self.on_update()

    @property
    def seq_lens_cuda(self):
        return self._seq_lens_cuda

    @property
    def seq_lens_kv(self) -> Optional[torch.Tensor]:
        return self._seq_lens_kv if self._seq_lens_kv is not None else self._seq_lens

    @seq_lens_kv.setter
    def seq_lens_kv(self, value: Optional[torch.Tensor]):
        value = value if value is not AttentionMetadata.seq_lens_kv else None
        self._seq_lens_kv = value
        self.on_update()
        if self._seq_lens_kv is not None:
            self._seq_lens_kv = self._seq_lens_kv.pin_memory()
            self._seq_lens_kv_cuda = self._seq_lens_kv.cuda(non_blocking=True)

    @property
    def seq_lens_kv_cuda(self):
        return self._seq_lens_kv_cuda if self._seq_lens_kv_cuda is not None else self._seq_lens_cuda

    @property
    def context_lens(self) -> torch.Tensor:
        return self.seq_lens[:self.num_contexts]

    @property
    def num_seqs(self) -> int:
        return self.seq_lens.shape[0]

    @property
    def is_cross(self) -> bool:
        return self.seq_lens is not self.seq_lens_kv

    @property
    def has_cross_sub_metadata(self) -> bool:
        return self.cross is not None and self.cross is not self

    @property
    def num_ctx_tokens(self) -> int:
        return self._num_ctx_tokens

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def prepare(self):
        pass # Placeholder for actual implementation if needed

    def create_cuda_graph_metadata(self,\
                                   max_batch_size: int,\
                                   sub_cross_metadata: bool = False,\
                                   max_draft_tokens: int = 0) -> "AttentionMetadata":
        return self # Placeholder

@dataclass
class RopeParams:
    """
    Parameters for Rotary Position Embedding (RoPE).
    旋转位置嵌入(RoPE)的参数。
    """
    dim: int = 0                            # Embedding dimension 嵌入维度
    theta: float = 10000.0                  # Base value for frequency computation 频率计算的基准值
    scale_type: RotaryScalingType = RotaryScalingType.none  # Type of scaling to apply 应用的缩放类型
    scale: float = 1.0                      # Scaling factor 缩放因子
    low_freq_factor: float = 1.0            # Factor for low frequency components 低频分量的因子
    high_freq_factor: float = 4.0           # Factor for high frequency components 高频分量的因子
    short_m_scale: float = 1.0              # Scaling for short sequences 短序列的缩放
    long_m_scale: float = 1.0               # Scaling for long sequences 长序列的缩放
    max_positions: int = 1024               # Maximum number of positions 最大位置数
    original_max_positions: int = 1024      # Original maximum positions 原始最大位置数
    beta_fast: int = 32                     # Fast decay parameter 快速衰减参数
    beta_slow: int = 1                      # Slow decay parameter 慢速衰减参数
    mscale: float = 1.0                     # Multiplicative scaling factor 乘性缩放因子
    mscale_all_dim: float = 0.0            # All-dimension multiplicative scaling 全维度乘性缩放

    @staticmethod
    def from_config(config) -> "RopeParams":
        """
        Create RoPE parameters from a config object.
        从配置对象创建RoPE参数。

        Args:
            config: Configuration object containing RoPE parameters
                   包含RoPE参数的配置对象

        Returns:
            RopeParams: Initialized RoPE parameters
                       初始化的RoPE参数
        """
        return RopeParams(
            dim=getattr(config, "rotary_embedding_dim", 0),
            theta=getattr(config, "rotary_embedding_base", 10000.0),
            scale=getattr(config, "rotary_embedding_scale", 1.0),
            scale_type=RotaryScalingType.from_string(
                getattr(config, "rotary_embedding_scaling_type", "none")
            ),
        )

@dataclass
class PositionalEmbeddingParams:
    """
    Parameters for positional embeddings.
    位置嵌入的参数。
    """
    type: PositionEmbeddingType            # Type of position embedding 位置嵌入类型
    embedder: Optional[object] = None       # Position embedder object 位置嵌入器对象
    rope: Optional[RopeParams] = None       # RoPE parameters RoPE参数
    is_neox: bool = True                    # Whether using NeoX-style position embeddings 是否使用NeoX风格的位置嵌入

class PredefinedAttentionMask(str, Enum):
    """
    Predefined attention mask types.
    预定义的注意力掩码类型。
    """
    CAUSAL = "causal"     # Causal (autoregressive) attention mask 因果(自回归)注意力掩码
    FULL = "full"         # Full attention mask 完全注意力掩码

@dataclass
class EagerFusionConfig:
    """
    Configuration for eager fusion optimizations.
    急切融合优化的配置。
    """
    PRE_MOE_FUSION: bool = False    # Whether to fuse operations before MoE 是否融合MoE之前的操作
    POST_MOE_FUSION: bool = False   # Whether to fuse operations after MoE 是否融合MoE之后的操作
    PRE_MLP_FUSION: bool = False    # Whether to fuse operations before MLP 是否融合MLP之前的操作
    POST_MLP_FUSION: bool = False   # Whether to fuse operations after MLP 是否融合MLP之后的操作
