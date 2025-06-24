from dataclasses import dataclass, field
class RotaryScalingType(IntEnum):
    none = 0
    linear = 1
    dynamic = 2
    longrope = 3
    llama3 = 4
    yarn = 5
    mrope = 6

    @staticmethod
    def from_string(s):
        if s == "linear":
            return RotaryScalingType.linear
        elif s == "dynamic":
            return RotaryScalingType.dynamic
        elif s == "longrope":
            return RotaryScalingType.longrope
        elif s == "llama3":
            return RotaryScalingType.llama3
        elif s == "yarn":
            return RotaryScalingType.yarn
        elif s == "mrope":
            return RotaryScalingType.mrope
        return RotaryScalingType.none

class PositionEmbeddingType(IntEnum):
    learned_absolute = 0
    rope_gptj = 1
    rope_gpt_neox = 2
    long_rope = 3
    alibi = 4
    alibi_with_scale = 5
    relative = 6
    chatglm = 7
    yarn = 8
    mrope = 9
    deferred = 10  # Apply customized positional embedding by using an external embedder. K will be cached before embedding.

    def is_rope(self) -> bool:
        return self in [
            PositionEmbeddingType.rope_gptj, PositionEmbeddingType.rope_gpt_neox,
            PositionEmbeddingType.long_rope, PositionEmbeddingType.yarn,
            PositionEmbeddingType.mrope
        ]

    def is_mrope(self) -> bool:
        return self == PositionEmbeddingType.mrope

    def is_alibi(self) -> bool:
        return self in [
            PositionEmbeddingType.alibi, PositionEmbeddingType.alibi_with_scale
        ]

    def is_deferred(self) -> bool:
        return self == PositionEmbeddingType.deferred

    @staticmethod
    def choices() -> List[str]:
        return [e.name for e in list(PositionEmbeddingType)]

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        if s == "learned_absolute":
            return PositionEmbeddingType.learned_absolute
        elif s == "rope_gptj":
            return PositionEmbeddingType.rope_gptj
        elif s == "rope_gpt_neox":
            return PositionEmbeddingType.rope_gpt_neox
        elif s == "long_rope":
            return PositionEmbeddingType.long_rope
        elif s == "alibi":
            return PositionEmbeddingType.alibi
        elif s == "alibi_with_scale":
            return PositionEmbeddingType.alibi_with_scale
        elif s == "relative":
            return PositionEmbeddingType.relative
        elif s == "chatglm":
            return PositionEmbeddingType.chatglm
        elif s == "yarn":
            return PositionEmbeddingType.yarn
        elif s == "mrope":
            return PositionEmbeddingType.mrope
        elif s == "deferred":
            return PositionEmbeddingType.deferred
        raise ValueError(f"PositionEmbeddingType {s} is not supported")

# Re-defining AttentionMetadata, AttentionRuntimeFeatures, AttentionInputType,
# RopeParams, PositionalEmbeddingParams, PredefinedAttentionMask

@dataclass
class AttentionRuntimeFeatures:
    chunked_prefill: bool = False
    cache_reuse: bool = False
    has_speculative_draft_tokens: bool = False


class AttentionInputType(IntEnum):
    mixed = 0
    context_only = 1
    generation_only = 2


@dataclass(kw_only=True)
class AttentionMetadata:
    max_num_requests: int
    max_num_tokens: int
    kv_cache_manager: object
    mapping: Optional[object] = None # Placeholder for Mapping

    enable_flash_mla: bool = False
    enable_paged_context_mla: bool = False
    is_cuda_graph: bool = field(default=False, repr=False)

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
    dim: int = 0
    theta: float = 10000.0
    scale_type: RotaryScalingType = RotaryScalingType.none
    scale: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    short_m_scale: float = 1.0
    long_m_scale: float = 1.0
    max_positions: int = 1024
    original_max_positions: int = 1024
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: float = 0.0

    @staticmethod
    def from_config(config) -> "RopeParams":
        # Simplified: In a real scenario, this would extract values from a config object
        return RopeParams(
            dim=getattr(config, "rotary_embedding_dim", 0),
            theta=getattr(config, "rotary_embedding_base", 10000.0),
            scale=getattr(config, "rotary_embedding_scale", 1.0),
            scale_type=RotaryScalingType.from_string(getattr(config, "rotary_embedding_scaling_type", "none")),
            # Add other relevant parameters
        )
    def create_rope_const_params(self, interleave: bool = True):
        return None # Placeholder

@dataclass
class PositionalEmbeddingParams:
    type: PositionEmbeddingType
    embedder: Optional[object] = None # Placeholder for PositionalEmbedder

    rope: Optional[RopeParams] = None
    is_neox: bool = True

    def __post_init__(self) -> None:
        pass # Placeholder

class PredefinedAttentionMask(str, Enum):
    CAUSAL = "causal"
    FULL = "full"


@dataclass
class EagerFusionConfig:
    PRE_MOE_FUSION: bool = False
    POST_MOE_FUSION: bool = False
    PRE_MLP_FUSION: bool = False
    POST_MLP_FUSION: bool = False
