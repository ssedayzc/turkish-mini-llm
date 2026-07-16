"""Model configuration for the tiny Gemma-style transformer.

Gemma's distinctive choices (vs Qwen):
  * 5 local (sliding-window) attention layers for every 1 global layer.
  * Local layers use a small RoPE base; global layers use a large one.
  * "Sandwich" normalization: an RMSNorm BEFORE and AFTER each sub-layer.
  * GeGLU (gelu) feed-forward instead of SwiGLU (silu).
  * Input embeddings are scaled by sqrt(hidden_size).
  * Per-Layer Embeddings (a small extra embedding added at every layer).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 30           # 30 Turkish chars (incl. newline)
    hidden_size: int = 32          # model / embedding dimension
    num_layers: int = 6            # 6 layers -> exactly one global layer (pattern below)
    num_heads: int = 4             # number of query heads
    num_kv_heads: int = 2          # key/value heads (GQA)
    head_dim: int = 8              # dimension per head
    intermediate_size: int = 64    # GeGLU hidden dimension
    max_seq_len: int = 24          # longest sequence we ever feed in
    rms_norm_eps: float = 1e-6     # epsilon inside RMSNorm

    # Local vs global attention.
    sliding_window: int = 8        # local layers only attend to the last `sliding_window` tokens
    global_every: int = 6          # every 6th layer is global (so 5 local : 1 global)
    rope_theta_local: float = 10000.0     # RoPE base for local layers
    rope_theta_global: float = 1000000.0  # RoPE base for global layers

    # Gemma scales queries by 1/sqrt(query_pre_attn_scalar) (here = head_dim).
    query_pre_attn_scalar: int = 8
