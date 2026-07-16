"""A Gemma transformer block with "sandwich" normalization.

Where Qwen normalizes only BEFORE each sub-layer, Gemma normalizes before AND
after, with the residual wrapping the whole thing:

    x = x + post_attn_norm( attn( input_norm(x) ) )
    x = x + post_ffn_norm(  mlp(  pre_ffn_norm(x) ) )

That extra post-norm keeps activations well-scaled and is a Gemma signature.
"""

import torch
from torch import nn

from config import ModelConfig
from rms_norm import RMSNorm
from attention import Attention
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        self.pre_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
        self.post_feedforward_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        h = self.attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.post_attention_layernorm(h)

        h = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(h)
        return x
