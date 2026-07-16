"""The DiT renderer — the coral "diffusion" region.

A small Diffusion Transformer that works on the 25Hz latent. It does not
predict the latent directly; it predicts the flow-matching *velocity* that
drags noise toward the target latent (see flow.py). Four real ACE-Step v1.5
design choices are kept, at toy scale:

  1. **Composite input.** The DiT's input is not just the noisy latent — it is
     channel-wise concat([noised target, FSQ source latent, mask]). The source
     latent (the planner's decoded 5Hz blueprint) rides along *inside* the
     sequence; cross-attention is reserved for the caption. The mask channel
     says which frames to generate (all ones here; the real model uses partial
     masks for repaint/edit).
  2. **Patchify.** Pairs of latent frames are merged into one token, halving
     the sequence (the real 25Hz -> 12.5Hz throughput trick).
  3. **Hybrid attention.** Odd blocks use sliding-window self-attention (local
     nuances), even blocks use global GQA (long-range structure).
  4. **AdaLN-Zero timestep conditioning.** Instead of adding a time embedding
     to the tokens, the timestep *modulates* each block: it produces per-block
     shift/scale for the norms and a gate for each residual branch, with the
     gates initialized to zero so every block starts as the identity.
"""

import torch
from attention import Attention, CrossAttention
from config import AceConfig
from mlp import MLP
from rms_norm import RMSNorm
from rotary import precompute_cos_sin
from torch import nn

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real DiT is a ~2B-parameter hybrid-attention Transformer (the XL is
# ~4B): odd layers use Sliding-Window Attention, even layers Global GQA, all
# with RoPE + RMSNorm + AdaLN-Zero timestep conditioning (a Qwen3-derived
# backbone). Conditioning comes in two roads, same as here: the composite
# input tensor combines Source + Noised Target + Mask through a patchify layer
# (25Hz -> 12.5Hz), while Qwen3-0.6B caption embeddings (plus timbre and lyric
# encoders) are injected via cross-attention. Here (toy): 4 blocks of width
# 64, patchify 50 -> 25 tokens, sliding window +-8, one caption token.
# ---------------------------------------------------------------------------


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar timestep t in [0, 1]. -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device).float() / half * 4.0)
    angles = t[:, None].float() * freqs[None, :] * 10.0
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: scale & shift normalized tokens, per batch item."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    """self-attn (local or global) -> cross-attn (caption) -> MLP, all AdaLN-gated."""

    def __init__(self, cfg: AceConfig, sliding_window: int = None):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, is_causal=False, sliding_window=sliding_window)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.cross_attn = CrossAttention(cfg)
        self.norm3 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

        # AdaLN-Zero: the timestep drives shift/scale/gate for the self-attn
        # and MLP branches. Zero init -> gates start at 0 -> identity blocks.
        self.ada = nn.Linear(cfg.hidden_size, 6 * cfg.hidden_size)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, t_emb, context, cos, sin):
        (shift_sa, scale_sa, gate_sa,
         shift_mlp, scale_mlp, gate_mlp) = self.ada(t_emb).chunk(6, dim=-1)

        x = x + gate_sa[:, None, :] * self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa), cos, sin)
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + gate_mlp[:, None, :] * self.mlp(
            modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class DiT(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.cfg = cfg
        d, p = cfg.latent_dim, cfg.patch_size
        composite_channels = 2 * d + 1          # noised target ++ source ++ mask

        # patchify: p latent frames of the composite input -> one token
        self.in_proj = nn.Linear(composite_channels * p, cfg.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size), nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        # odd blocks local (sliding window), even blocks global — the hybrid.
        self.blocks = nn.ModuleList([
            DiTBlock(cfg, sliding_window=cfg.sliding_window if i % 2 == 1 else None)
            for i in range(cfg.dit_layers)
        ])
        self.norm_out = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ada_out = nn.Linear(cfg.hidden_size, 2 * cfg.hidden_size)  # final shift/scale
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        self.out_proj = nn.Linear(cfg.hidden_size, d * p)  # token -> p velocity frames

        cos, sin = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, noisy: torch.Tensor, t: torch.Tensor,
                text_embed: torch.Tensor, source_latent: torch.Tensor) -> torch.Tensor:
        """noisy [B, d, 50], t [B], text_embed [B, 1, h], source_latent [B, d, 50]
        -> predicted velocity [B, d, 50]."""
        cfg = self.cfg
        B, d, T = noisy.shape
        n_tokens = T // cfg.patch_size
        cos, sin = self.cos[:n_tokens], self.sin[:n_tokens]

        # Composite input: noised target ++ source ++ all-ones mask, then patchify.
        mask = torch.ones(B, 1, T, device=noisy.device)
        composite = torch.cat([noisy, source_latent, mask], dim=1)   # [B, 2d+1, T]
        tokens = composite.transpose(1, 2).reshape(B, n_tokens, -1)  # [B, T/p, (2d+1)*p]
        x = self.in_proj(tokens)                                     # [B, T/p, h]

        t_emb = self.time_mlp(timestep_embedding(t, cfg.hidden_size))  # [B, h]

        for block in self.blocks:
            x = block(x, t_emb, text_embed, cos, sin)

        shift, scale = self.ada_out(t_emb).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), shift, scale)
        v = self.out_proj(x)                                         # [B, T/p, d*p]
        return v.reshape(B, T, d).transpose(1, 2)                    # unpatchify -> [B, d, T]
