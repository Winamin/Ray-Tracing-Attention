"""Versioned model geometry shared by coefficient translation and native kernels."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    hidden_size: int
    middle_size: int
    vocabulary_size: int
    layer_types: tuple[str, ...]
    attention_heads: int
    kv_heads: int
    attention_head_dim: int
    rotary_dim: int
    rope_theta: float = 10000000.0
    norm_epsilon: float = 1e-6
    attention_gate: bool = True
    qk_norm: bool = True
    state_heads: int = 1
    state_key_heads: int = 1
    state_head_dim: int = 128
    version: int = 1

    @property
    def depth(self): return len(self.layer_types)
    @property
    def full_layers(self): return tuple(i for i, k in enumerate(self.layer_types) if k == 'full')
    @property
    def recurrent_layers(self): return tuple(i for i, k in enumerate(self.layer_types) if k == 'recurrent')
    @property
    def query_width(self): return self.attention_heads * self.attention_head_dim
    @property
    def kv_width(self): return self.kv_heads * self.attention_head_dim
    @property
    def state_width(self): return self.state_heads * self.state_head_dim
    @property
    def working_width(self): return max(self.query_width, self.state_width)
    @property
    def feed_forward_stride(self): return max(self.middle_size, 2 * self.working_width)
    @property
    def history_bytes(self): return ((self.kv_width + 255) // 256 * 354 + 7) // 8 * 8

    def validate(self):
        for name in ('hidden_size','middle_size','vocabulary_size','attention_heads','kv_heads','attention_head_dim','state_heads','state_key_heads'):
            if getattr(self,name) <= 0: raise ValueError(f'{name} must be positive')
        if not self.depth or any(k not in ('full','recurrent') for k in self.layer_types):
            raise ValueError('The native graph uses full and recurrent optical attention layers')
        if self.attention_heads % self.kv_heads or self.state_heads % self.state_key_heads:
            raise ValueError('Query/value heads must be divisible by their shared key heads')
        if self.attention_head_dim > 256 or self.attention_head_dim % 2:
            raise ValueError('Native attention head dimension must be even and at most 256')
        if self.rotary_dim % 2 or not 0 <= self.rotary_dim <= self.attention_head_dim:
            raise ValueError('Rotary dimension must be even and fit the attention head')
        if self.recurrent_layers and self.state_head_dim != 128:
            raise ValueError('The recurrent optical operator currently uses 128-dimensional state heads')
        for name, value in (('hidden',self.hidden_size),('MLP',self.middle_size),('attention projection',self.query_width)):
            if value % 256: raise ValueError(f'{name} width must be aligned to 256 coefficients')
        if self.rope_theta <= 0 or self.norm_epsilon <= 0: raise ValueError('Invalid RoPE or norm parameters')
        if self.kv_width > 16384: raise ValueError('History packet exceeds a 128-thread packing group')
        return self

    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, value):
        value=dict(value);value['layer_types']=tuple(value['layer_types'])
        return cls(**value).validate()
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()[:16]

    def defines(self):
        return {'RT_HIDDEN_SIZE':self.hidden_size,'RT_LAYER_COUNT':self.depth,
            'RT_MIDDLE_SIZE':self.middle_size,'RT_VOCABULARY_SIZE':self.vocabulary_size,
            'RT_STATE_HEADS':self.state_heads,'RT_STATE_KEY_HEADS':self.state_key_heads,
            'RT_GQA_WIDTH':self.kv_width,'RT_ATTENTION_HEADS':self.attention_heads,
            'RT_ATTENTION_HEAD_DIM':self.attention_head_dim,'RT_ROTARY_DIM':self.rotary_dim,
            'RT_QUERY_WIDTH':self.query_width,'RT_STATE_WIDTH':self.state_width,
            'RT_WORKING_WIDTH':self.working_width,'RT_FF_STRIDE':self.feed_forward_stride,
            'RT_FULL_COUNT':len(self.full_layers),'RT_RECURRENT_COUNT':len(self.recurrent_layers),
            'RT_ROPE_THETA':f'{self.rope_theta:.1f}','RT_NORM_EPSILON':f'{self.norm_epsilon:.9g}',
            'RT_ATTENTION_GATE':int(self.attention_gate),'RT_QK_NORM':int(self.qk_norm),'RT_GGUF_Q4':1}


def asset_spec(directory):
    """Read the compiler's versioned model specification."""
    config=json.loads((Path(directory)/'model.json').read_text(encoding='utf8'))
    if 'model_spec' in config: return ModelSpec.from_dict(config['model_spec'])
    raise ValueError('Run rt-compile on the source weights to generate the model specification')
