"""LoRA — Low-Rank Adaptation (Hu et al., 2021), plus the rsLoRA scaling fix.

Full fine-tuning updates every entry of a weight matrix W [out, in].
LoRA freezes W and learns a *low-rank correction* next to it instead:

    y = W x + s * B A x          A: [r, in]   B: [out, r]   s = alpha / r

Why this works, in three observations:

  1. B @ A is an [out, in] matrix, just like W — but built from only
     r*(in + out) numbers instead of in*out. For r << min(in, out)
     that is a tiny fraction (here: r=4 on a 32x32 W -> 256 vs 1024).
  2. B starts at ZERO, so at step 0 the model is exactly the base model.
     A starts random (Kaiming uniform, like a fresh nn.Linear); it gives
     B something to work with. (If both were zero, the gradient of A
     would also be zero — see by_hand.py — and nothing could ever move.)
  3. After training you can either keep (A, B) as a tiny swappable
     "adapter" file, or fold the correction into the base weight once:
     W' = W + s*BA  ->  zero extra cost at inference ("merge").

The scale s = alpha/r is a knob so that changing r does not change the
size of the update you tuned alpha for.

rsLoRA (rank-stabilized LoRA, Kalajdzievski 2023) is a one-line subtype:
it uses s = alpha / sqrt(r) instead. Reason: the entries of A x sum over
r random-ish terms, so |B A x| grows like sqrt(r); dividing by r (classic
LoRA) over-shrinks high ranks, dividing by sqrt(r) keeps the update the
same size at every rank.
"""

# math.sqrt for the rsLoRA scale; math.sqrt(5) for the Kaiming init below.
import math
# dataclass gives us a tidy, typed config object with default values.
from dataclasses import dataclass

# torch for tensors; nn for Module/Parameter/Linear.
import torch
from torch import nn


# A plain settings bag shared by every adapter type in this folder.
@dataclass
class LoRAConfig:
    # r = the rank of the correction. This single number IS the whole trick:
    # it sets how many independent directions the adapter can move.
    r: int = 4
    # alpha = the scale numerator. The effective scale is s = alpha/r, so with
    # r=4 and alpha=8 the update is multiplied by 2.0 before being added.
    alpha: float = 8.0
    # dropout applied to the *input* of the adapter path only (0.0 = disabled).
    dropout: float = 0.0
    # False -> classic LoRA scale alpha/r ; True -> rsLoRA scale alpha/sqrt(r).
    rank_stabilized: bool = False
    # Which nn.Linear sub-modules (matched by their attribute name) get an
    # adapter. (q,v) is the original paper's choice; adapting all of
    # q,k,v,o,gate,up,down is common in modern practice.
    targets: tuple = ("q_proj", "v_proj")


# Wraps ONE existing nn.Linear (its W is frozen) and adds the trainable A,B.
class LoRALinear(nn.Module):

    # base: the pretrained Linear to adapt. cfg: the LoRAConfig above.
    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        # Register this object as an nn.Module so PyTorch tracks its params.
        super().__init__()
        # Keep the original layer as a child module (its W stays and is reused).
        self.base = base
        # Freeze the base: its weight (and bias, if any) gets no gradient and
        # will not be touched by the optimizer. Only A and B will learn.
        for p in self.base.parameters():
            p.requires_grad = False

        # Read the two matrix dimensions off the wrapped Linear.
        in_f, out_f = base.in_features, base.out_features
        # Pick the scale: rsLoRA divides alpha by sqrt(r); classic by r.
        self.scale = (cfg.alpha / math.sqrt(cfg.r)) if cfg.rank_stabilized \
                     else (cfg.alpha / cfg.r)

        # A: shape [r, in]. Allocated empty, then filled with a random init
        # below so the adapter has a non-degenerate starting direction.
        self.lora_A = nn.Parameter(torch.empty(cfg.r, in_f))
        # B: shape [out, r]. Initialized to ZEROS so that s*B@A = 0 at step 0
        # and the wrapped model starts out identical to the base model.
        self.lora_B = nn.Parameter(torch.zeros(out_f, cfg.r))
        # Kaiming-uniform is exactly how nn.Linear seeds its own weight; using
        # it here means A is scaled sensibly for its fan-in.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Optional dropout on the adapter input (a light regularizer).
        self.dropout = nn.Dropout(cfg.dropout)
        # A run-time switch: set False to make forward() ignore the adapter
        # entirely, which is how we prove "adapter off == the base model".
        self.enabled = True

    # The [out, in] correction this adapter currently represents: s * B @ A.
    # Handy for inspection and for the merge step; not used on the fast path.
    def delta_weight(self) -> torch.Tensor:
        # Matrix-multiply B [out, r] by A [r, in] -> [out, in], then scale.
        return self.scale * (self.lora_B @ self.lora_A)

    # The forward pass: base output plus the low-rank correction.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always compute the frozen base result first: y = W x (+ bias).
        y = self.base(x)
        # If the adapter is switched off, return the base result unchanged.
        if not self.enabled:
            return y
        # The low-rank path, read left to right on the last dimension:
        #   dropout(x)        : [.., in]   (optionally zero some inputs)
        #   @ lora_A.T        : [.., in] @ [in, r] -> [.., r]   (squeeze to r)
        #   @ lora_B.T        : [.., r]  @ [r, out] -> [.., out] (lift to out)
        #   * scale           : multiply the whole correction by s
        # Crucially we never build the [out, in] matrix B@A, so this is cheap.
        return y + self.scale * (self.dropout(x) @ self.lora_A.T @ self.lora_B.T)

    # Fold the correction into the base weight, then behave like a plain Linear.
    # torch.no_grad(): this is a weight edit, not part of any backward pass.
    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight and return a plain Linear.

        We also flip `enabled` off so a later forward uses the (now updated)
        base only, instead of adding the correction a second time.
        """
        # W <- W + s*B@A. After this the base alone reproduces the adapted map.
        self.base.weight += self.delta_weight()
        # Switch the adapter path off so forward() does not add s*B@A again.
        self.enabled = False
        # Return the updated base in case the caller wants the bare Linear.
        return self.base
