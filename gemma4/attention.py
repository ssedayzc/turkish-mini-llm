"""Gemma attention: GQA + QK-Norm + RoPE, with an explicit attention mask.

The mask is passed in (built once by the model), because Gemma uses two kinds:
  * global layers : ordinary causal mask (attend to all earlier tokens)
  * local layers  : causal AND only within a small sliding window

Queries are scaled by 1/sqrt(query_pre_attn_scalar) before the dot product,
which is Gemma's explicit version of the usual attention scaling.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig
from rms_norm import RMSNorm
from rotary import apply_rotary


def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    if n_repeat == 1:
        return x
    return x.repeat_interleave(n_repeat, dim=1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.n_repeat = cfg.num_heads // cfg.num_kv_heads
        self.scale = cfg.query_pre_attn_scalar ** -0.5   # Gemma's query scaling

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)   # QK-Norm
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(self.q_norm(q), cos, sin)   # QK-Norm, then RoPE
        k = apply_rotary(self.k_norm(k), cos, sin)

        k = repeat_kv(k, self.n_repeat)              # GQA
        v = repeat_kv(v, self.n_repeat)

        # attn_mask is an additive float mask of shape [T, T] (0 = keep, -inf = block).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)
