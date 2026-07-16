"""The 5Hz-lm planner — the purple "Composer Agent".

This is a Qwen3-style causal language model (the very same TinyQwen recipe used
in ../qwen3), but its vocabulary is not letters-of-a-name: it is

    [ letter tags ] ++ [<think>, </think>] ++ [ degree tokens ] ++ [ 512 audio codes ]

and its one job is: given a tag token, write the whole song plan

    tag  <think>  d0 d1 d2 d3  </think>  c0 c1 ... c9

First it "reasons" inside a <think> block — the four scale degrees of the
melody, the toy stand-in for the real model's YAML metadata (bpm, key,
duration, caption). Then it emits the coarse 5Hz blueprint: ``code_len``
audio-code tokens, answering *what to play* before any audio is rendered.
It never produces a continuous signal; that is the DiT's job.

Because every position in the plan has a fixed token type, ``generate`` masks
the logits to the legal type at each step (structured decoding) — the toy
version of constraining the LM to the audio-code range.
"""

import torch
import torch.nn.functional as F
from block import TransformerBlock
from config import AceConfig
from rms_norm import RMSNorm
from rotary import precompute_cos_sin
from torch import nn

# --- Real ACE-Step v1.5, for comparison ------------------------------------
# The real 5Hz-lm is Qwen3-based (0.6B / 1.7B / 4B). It first *reasons* inside
# a <think> block of YAML metadata, then emits one <|audio_code_N|> token per
# 200ms (5/sec -> 1200 tokens for a 240s song), N indexing a ~64k codebook:
#
#     <think>
#     bpm: 187
#     keyscale: D major
#     timesignature: 4
#     language: ja
#     duration: 344
#     caption: <expanded tags / description>
#     </think>
#     <|audio_code_5434|><|audio_code_20161|><|audio_code_7418|> ...
#         ... <|audio_code_35639|><|audio_code_35847|><|audio_code_15174|>
#
# Here (toy): the "caption" is one letter, the <think> metadata is the 4 scale
# degrees of the melody, the codebook is 512 (not ~64k), and the blueprint is
# code_len=10 tokens (a real 5Hz for our 2s bar, vs 1200 tokens for 240s).
# ---------------------------------------------------------------------------


def token_layout(cfg: AceConfig):
    """Where each token family starts: (think, end_think, degrees, codes)."""
    think = cfg.n_letters
    return think, think + 1, think + 2, think + 2 + cfg.n_degrees


def make_batch(tag_ids: torch.Tensor, degrees: torch.Tensor,
               codes: torch.Tensor, cfg: AceConfig):
    """Build (input, target) for next-token training over the full plan.

    For a tag with degrees [d0..d3] and codes [c0..c9]:
        seq    = [tag, <think>, d0..d3, </think>, c0..c9]      (17 tokens)
        input  = seq[:-1],  target = seq[1:]
    """
    think, end_think, deg_off, code_off = token_layout(cfg)
    B = tag_ids.shape[0]
    filler = lambda tok: torch.full((B, 1), tok, dtype=torch.long, device=tag_ids.device)
    seq = torch.cat([tag_ids[:, None], filler(think), degrees + deg_off,
                     filler(end_think), codes + code_off], dim=1)
    return seq[:, :-1], seq[:, 1:]


class Planner(nn.Module):
    def __init__(self, cfg: AceConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.planner_vocab, cfg.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.planner_vocab, bias=False)
        self.lm_head.weight = self.embed_tokens.weight   # weight tying

        cos, sin = precompute_cos_sin(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        cos, sin = self.cos[:T], self.sin[:T]

        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tag_ids: torch.Tensor):
        """tag_ids [B] -> (codes [B, code_len], think_degrees [B, n_think]).

        Structured decoding: the plan's shape is fixed, so each step only the
        legal token family is allowed — <think>, then degrees, then </think>,
        then exactly code_len audio codes.
        """
        cfg = self.cfg
        think, end_think, deg_off, code_off = token_layout(cfg)
        n_think = 4                                       # degrees inside <think>

        # legal (lo, hi) id range for each generated position
        ranges = ([(think, think + 1)] + [(deg_off, deg_off + cfg.n_degrees)] * n_think
                  + [(end_think, end_think + 1)]
                  + [(code_off, code_off + cfg.num_codes)] * cfg.code_len)

        idx = tag_ids[:, None]                            # [B, 1]
        for lo, hi in ranges:
            logits, _ = self(idx[:, -cfg.max_seq_len:])
            logits = logits[:, -1, :]
            mask = torch.full_like(logits, float("-inf"))
            mask[:, lo:hi] = 0.0                          # only this family is legal
            next_token = (logits + mask).argmax(dim=-1, keepdim=True)  # deterministic -> greedy
            idx = torch.cat([idx, next_token], dim=1)

        degrees = idx[:, 2:2 + n_think] - deg_off         # inside the <think> block
        codes = idx[:, -cfg.code_len:] - code_off         # the 5Hz blueprint
        return codes, degrees
