"""FSQ bridge — the blue right arm that joins the planner and the DiT.

The planner LM speaks in *discrete tokens*; the DiT renders a *continuous*
latent. FSQ (Finite Scalar Quantization) is the translator between them, and
it also drops the resolution from 25Hz to 5Hz — the same x5 ratio as the real
model:

    latent [B, 16, 50] --attention pool x5--> [B, 10, 16] --proj--> [B, 10, 3]
                                    quantize each dim to its levels (8, 8, 8)
    -> one integer code per 5Hz frame: codes [B, 10]   (codebook = 8^3 = 512)

and the inverse turns codes back into a continuous "source latent" that seeds
the DiT. The pooling is a real (single-head) attention pool: one learned query
attends over the five 25Hz frames inside each 5Hz window, so the model learns
*which part of the window matters* instead of just averaging.

Quantization uses the straight-through trick (round on the forward pass,
identity gradient on the backward pass) so everything upstream still trains.
We index each scalar dim into {0 .. level-1} via a sigmoid, then pack the
per-dim integers into one code with mixed-radix arithmetic — exactly how a
multi-dimensional code becomes a single codebook id.
"""

import torch
import torch.nn.functional as F
from config import AceConfig
from torch import nn

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real FSQ tokenizer uses *attention pooling* to compress 25Hz VAE latents
# into 5Hz discrete codes over a codebook of ~64k (hence ids like
# <|audio_code_35639|>). The codebook is so large that a real failure mode is
# the LM emitting an id beyond its range (ace-step/ACE-Step-1.5 issue #92).
# Here (toy): levels=(8, 8, 8) -> a 512-entry codebook, code ids 0..511, and
# the same 25Hz -> 5Hz (x5) compression via a one-query attention pool.
# ---------------------------------------------------------------------------


class FSQBridge(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.cfg = cfg
        self.dim = len(cfg.fsq_levels)              # scalar dims per frame (3)
        self.pool = cfg.latent_len // cfg.code_len  # 25Hz -> 5Hz pooling factor (5)
        d = cfg.latent_dim

        # levels per dim and the mixed-radix basis to pack/unpack a single code id.
        levels = torch.tensor(cfg.fsq_levels)                  # [dim]
        basis = torch.cumprod(torch.tensor([1] + list(cfg.fsq_levels[:-1])), 0)
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("basis", basis, persistent=False)

        # Attention pooling: one learned query per 5Hz frame (shared), keys and
        # values projected from the five 25Hz frames inside that window.
        self.pool_query = nn.Parameter(torch.randn(d) / d**0.5)
        self.pool_key = nn.Linear(d, d)
        self.pool_value = nn.Linear(d, d)

        # pooled 5Hz vector <-> the few FSQ scalars for that frame, and back
        # up to the full window of 25Hz latent frames.
        self.proj_in = nn.Linear(d, self.dim)
        self.proj_out = nn.Linear(self.dim, d * self.pool)

    # ---- 25Hz -> 5Hz: attention pooling ------------------------------------
    def _attn_pool(self, latent: torch.Tensor) -> torch.Tensor:
        """[B, d, 50] -> [B, code_len, d]: each 5Hz frame attends over its window."""
        B, d, T = latent.shape
        windows = latent.transpose(1, 2).reshape(B, self.cfg.code_len, self.pool, d)
        k = self.pool_key(windows)                             # [B, 10, 5, d]
        v = self.pool_value(windows)                           # [B, 10, 5, d]
        scores = (k @ self.pool_query) / d**0.5                # [B, 10, 5]
        weights = F.softmax(scores, dim=-1)                    # who matters in the window
        return (weights.unsqueeze(-1) * v).sum(dim=2)          # [B, 10, d]

    def _unpool(self, x: torch.Tensor) -> torch.Tensor:
        """[B, code_len, pool*d] -> [B, d, 50]: scatter 5Hz frames back over time."""
        B = x.shape[0]
        x = x.reshape(B, self.cfg.latent_len, self.cfg.latent_dim)
        return x.transpose(1, 2)

    # ---- the quantizer ------------------------------------------------------
    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Map reals to per-dim integers in {0..level-1}, straight-through."""
        qf = torch.sigmoid(z) * (self.levels - 1)              # [..., dim] in [0, L-1]
        return qf + (torch.round(qf) - qf).detach()            # round, but pass gradient

    def _pack(self, q: torch.Tensor) -> torch.Tensor:
        """Per-dim integers [..., dim] -> one code id [...]  (mixed radix)."""
        return (q.long() * self.basis).sum(-1)

    def _unpack(self, codes: torch.Tensor) -> torch.Tensor:
        """One code id [...] -> per-dim integers [..., dim]."""
        return (codes.unsqueeze(-1) // self.basis) % self.levels

    @staticmethod
    def _centered(q: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Integer codes -> continuous values in [-1, 1] for reconstruction."""
        return q / (levels - 1) * 2 - 1

    # ---- public API ---------------------------------------------------------
    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent [B, d, 50] -> discrete codes [B, code_len] (the 5Hz blueprint)."""
        q = self.quantize(self.proj_in(self._attn_pool(latent)))
        return self._pack(q)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, code_len] -> source latent [B, d, 50] (seeds the DiT)."""
        q = self._unpack(codes)
        x = self.proj_out(self._centered(q.float(), self.levels))
        return self._unpool(x)

    def forward(self, latent: torch.Tensor):
        """Round-trip used in training. Returns (source_latent, codes).

        Uses the straight-through quantized values so gradients reach the
        pooling and both projections; the integer codes come along for the
        planner's targets.
        """
        q = self.quantize(self.proj_in(self._attn_pool(latent)))   # [B, code_len, dim]
        codes = self._pack(q)
        source = self._unpool(self.proj_out(self._centered(q, self.levels)))
        return source, codes
