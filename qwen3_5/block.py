"""A Qwen3.5 transformer block.

Identical pre-norm structure to Qwen3, except the token mixer is chosen per layer:
either full softmax Attention or the linear Gated DeltaNet.
"""

import torch
from torch import nn

from config import ModelConfig
from rms_norm import RMSNorm
from attention import Attention
from gated_deltanet import GatedDeltaNet
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, use_full_attention: bool):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # The only difference between layers: which mixer runs here.
        self.mixer = Attention(cfg) if use_full_attention else GatedDeltaNet(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
