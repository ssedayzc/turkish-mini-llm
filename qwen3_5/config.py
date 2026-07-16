"""Model configuration for the tiny Qwen3.5-style transformer.

Qwen3.5 is a *hybrid*: most layers are cheap Gated DeltaNet (linear attention),
and every Nth layer is full softmax attention. `full_attn_every` controls that
ratio (Qwen3.5 uses 4 -> 1 full-attention layer for every 3 linear ones).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 30          # 30 Turkish chars (incl. newline)
    hidden_size: int = 32         # model / embedding dimension
    num_layers: int = 4           # number of transformer blocks
    num_heads: int = 4            # number of query heads
    num_kv_heads: int = 2         # key/value heads for the FULL-attention layers (GQA)
    head_dim: int = 8             # dimension per head (= hidden_size / num_heads)
    intermediate_size: int = 64   # SwiGLU hidden dimension (~2x hidden_size)
    max_seq_len: int = 24         # longest sequence we ever feed in (names are short)
    rope_theta: float = 10000.0   # RoPE base frequency (full-attention layers only)
    rms_norm_eps: float = 1e-6    # epsilon inside RMSNorm
    full_attn_every: int = 4      # every 4th layer is full attention; the rest are Gated DeltaNet
