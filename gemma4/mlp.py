"""GeGLU feed-forward network, the MLP Gemma uses.

Same gated shape as SwiGLU, but the gate is passed through GELU instead of SiLU:

    down( gelu(gate(x)) * up(x) )
"""

from torch import nn
import torch.nn.functional as F

from config import ModelConfig


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
