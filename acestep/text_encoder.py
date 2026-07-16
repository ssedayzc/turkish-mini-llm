"""Tag encoder — the blue left arm of the conditioning bridge.

In ACE-Step a Qwen3-Embedding text encoder turns the caption/lyrics into
vectors that feed the DiT's cross-attention. Our caption is a single letter,
so the "encoder" is an embedding table that lifts the tag id into one
conditioning token of width hidden_size.

It also owns a learned **null embedding** — the "no caption" token. During
training the caption is randomly replaced by it (cfg.cond_dropout), which is
what makes classifier-free guidance possible at sampling time: the DiT learns
both "render this tag" and "render something", and CFG extrapolates between
the two predictions.
"""

import torch
from config import AceConfig
from torch import nn

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real conditioning encoder is Qwen3-Embedding-0.6B: it turns the caption
# (free-text tags / description) into embeddings consumed by the DiT's cross-
# attention, concatenated with dedicated *timbre* and *lyric* encoders. CFG on
# the base/sft checkpoints uses a learned `null_condition_emb`, exactly like
# our null row (the distilled turbo model skips CFG entirely). Here (toy): one
# nn.Embedding row per single-letter tag + one learned null vector.
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.n_letters, cfg.hidden_size)
        self.null_condition_emb = nn.Parameter(torch.zeros(cfg.hidden_size))

    def forward(self, tag_ids: torch.Tensor) -> torch.Tensor:
        """tag_ids [B] -> conditioning [B, 1, hidden_size] (a length-1 context)."""
        return self.embed(tag_ids).unsqueeze(1)

    def null_embed(self, batch_size: int) -> torch.Tensor:
        """The 'no caption' conditioning, for cond-dropout in training and CFG."""
        return self.null_condition_emb.expand(batch_size, 1, -1)
