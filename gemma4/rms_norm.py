"""Gemma's RMSNorm.

Same root-mean-square rescaling as everyone else, with one Gemma quirk: the
learnable weight is initialized to ZERO and applied as (1 + weight). So at the
start every layer is an identity-ish scale, and the network learns how far to
move away from 1. Math is done in float32, then cast back.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))   # note: zeros, not ones

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(dtype)   # (1 + weight)
