"""VeRA — Vector-based Random-matrix Adaptation (Kopiczko et al., 2024).

LoRA's question: how few numbers can steer a frozen model? VeRA's answer
is "far fewer than you think": don't even train A and B. Freeze them as
RANDOM matrices, shared by every layer, and train only two small vectors
per layer that rescale their rows:

    y = W x + b * ( B ( d * (A x) ) )      A: [r, in] frozen random
                                           B: [out, r] frozen random
                                           d: [r]   trainable, init 0.1
                                           b: [out] trainable, init 0  -> starts silent

Equivalently  dW = diag(b) B diag(d) A: the random matrices provide a
fixed bank of directions, and d/b learn how much of each to mix in.

Trainable numbers per layer:  r + out            (VeRA)
                       versus r * (in + out)     (LoRA)

On this model's q_proj (32->32, r=4): 36 numbers instead of 256.

Because A and B are never trained, they are not even worth saving: we
regenerate them from a fixed random seed at load time. An adapter file
holds only the little d and b vectors. (The paper shares one big A/B pair
sliced to each layer's shape; we regenerate per shape from the same seed —
same spirit: random, frozen, reproducible, not stored.)
"""

# torch for tensors and the seeded RNG; nn for Module/Parameter.
import torch
from torch import nn


# Build a deterministic "random" matrix: identical numbers for a given
# (seed, shape) on any machine. This is what lets us NOT save A and B — we
# just regenerate them on load.
def shared_random(shape: tuple, seed: int = 0) -> torch.Tensor:
    # A private RNG whose seed is mixed with the shape, so different-shaped
    # layers get different matrices, but the same layer is always reproducible.
    gen = torch.Generator().manual_seed(seed + 31 * shape[0] + shape[1])
    # Standard normal values, divided by sqrt(fan_in) to keep the output
    # variance ~1 (the usual random-projection scaling).
    return torch.randn(shape, generator=gen) / (shape[1] ** 0.5)


# Frozen base + frozen random A/B, with only the small vectors d, b trainable.
class VeRALinear(nn.Module):

    # base: the Linear to adapt. cfg: reuses LoRAConfig (we only read cfg.r).
    # seed: chooses which reproducible random A/B this layer gets.
    def __init__(self, base: nn.Linear, cfg, seed: int = 0):
        # Register as an nn.Module.
        super().__init__()
        # Keep and freeze the base weight — VeRA never trains W either.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        # Matrix dimensions of the wrapped Linear.
        in_f, out_f = base.in_features, base.out_features
        # Store A and B as *buffers*, not Parameters: buffers move with
        # .to(device) and are saved/restored, but never receive gradients.
        # (We exclude them from the saved adapter and regenerate instead.)
        self.register_buffer("vera_A", shared_random((cfg.r, in_f), seed))
        self.register_buffer("vera_B", shared_random((out_f, cfg.r), seed))

        # d: length-r trainable scale on the r intermediate features. The paper
        # initializes it to a small constant (0.1) rather than 0.
        self.vera_d = nn.Parameter(torch.full((cfg.r,), 0.1))
        # b: length-out trainable scale on the outputs. Initialized to 0 so the
        # whole correction is 0 at step 0 (the model starts as the base model).
        self.vera_b = nn.Parameter(torch.zeros(out_f))
        # Same on/off switch as LoRALinear.
        self.enabled = True

    # The [out, in] correction VeRA currently represents.
    def delta_weight(self) -> torch.Tensor:
        # dW = diag(b) B diag(d) A. Implemented as elementwise row/col scaling:
        #   (d.unsqueeze(1) * A) scales A's rows by d          -> [r, in]
        #   B @ (...)            mixes them through frozen B    -> [out, in]
        #   b.unsqueeze(1) * ... scales the output rows by b    -> [out, in]
        return self.vera_b.unsqueeze(1) * (self.vera_B @ (self.vera_d.unsqueeze(1) * self.vera_A))

    # Forward pass: base output plus the scaled random-projection correction.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base path first.
        y = self.base(x)
        # Respect the on/off switch.
        if not self.enabled:
            return y
        # Project x down through frozen A, then scale the r features by d:
        #   x @ vera_A.T : [.., in] -> [.., r]
        #   * vera_d     : elementwise scale of the r features
        h = (x @ self.vera_A.T) * self.vera_d
        # Lift back up through frozen B, then scale the outputs by b:
        #   h @ vera_B.T : [.., r] -> [.., out]
        #   * vera_b     : elementwise scale of the out features
        h = (h @ self.vera_B.T) * self.vera_b
        # Add the correction to the base output.
        return y + h

    # Fold the correction into the base weight (adapter becomes a no-op).
    @torch.no_grad()
    def merge(self) -> nn.Linear:
        # W <- W + dW.
        self.base.weight += self.delta_weight()
        # Prevent forward() from adding the correction again.
        self.enabled = False
        return self.base
