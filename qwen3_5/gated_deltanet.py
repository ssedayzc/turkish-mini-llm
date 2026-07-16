"""Gated DeltaNet — the linear-attention token mixer that defines Qwen3.5.

Full softmax attention compares every token with every other token: O(T^2) work
and an ever-growing KV cache. Gated DeltaNet instead keeps ONE fixed-size memory
matrix `S` (key-space -> value-space) and updates it one token at a time, so cost
is linear in sequence length.

The update is the *gated delta rule*:

    S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) (k_t)^T

  * alpha_t  (decay gate)  : how much old memory to keep            (0..1)
  * beta_t   (write gate)  : how strongly to write the correction   (0..1)
  * (v_t - alpha_t S_{t-1} k_t)  is the "delta": the part of v we predicted wrong,
    so we only ever store NEW information (the error-correcting "delta rule").

We then read the memory with the query:   o_t = S_t q_t .

For clarity this is written as a plain Python loop over time. That is perfectly
fine for the short sequences here; real implementations parallelize it in chunks.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        dim = cfg.num_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.hidden_size, dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, dim, bias=False)
        self.o_proj = nn.Linear(dim, cfg.hidden_size, bias=False)

        # One decay gate and one write gate per head, produced from the input.
        self.alpha_proj = nn.Linear(cfg.hidden_size, cfg.num_heads)
        self.beta_proj = nn.Linear(cfg.hidden_size, cfg.num_heads)

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        # cos/sin are accepted but unused: Gated DeltaNet needs no RoPE, because
        # position is captured by the order in which the memory is updated.
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, H, D)
        v = self.v_proj(x).view(B, T, H, D)

        # L2-normalize queries and keys (keeps the delta rule numerically stable).
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        alpha = torch.sigmoid(self.alpha_proj(x))   # [B, T, H] decay gate in (0,1)
        beta = torch.sigmoid(self.beta_proj(x))     # [B, T, H] write gate in (0,1)

        # Running memory, one matrix per (batch, head): [B, H, Dv, Dk].
        S = x.new_zeros(B, H, D, D)
        outputs = []
        for t in range(T):
            k_t = k[:, t]                  # [B, H, D]
            v_t = v[:, t]                  # [B, H, D]
            q_t = q[:, t]                  # [B, H, D]
            a_t = alpha[:, t][..., None]   # [B, H, 1]
            b_t = beta[:, t][..., None]    # [B, H, 1]

            pred = torch.einsum("bhvk,bhk->bhv", S, k_t)         # S k_t : current guess for v
            delta = v_t - a_t * pred                              # error after decay
            update = b_t[..., None] * (delta[..., :, None] * k_t[..., None, :])  # b * delta (k)^T
            S = a_t[..., None] * S + update                      # decay old memory, add correction

            o_t = torch.einsum("bhvk,bhk->bhv", S, q_t)          # read out with the query
            outputs.append(o_t)

        out = torch.stack(outputs, dim=1).reshape(B, T, H * D)   # [B, T, H*D]
        return self.o_proj(out)
