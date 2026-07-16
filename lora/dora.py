"""DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).

Every weight matrix can be split into two ingredients, row by row:

    magnitude m_i = ||W_i||          (the length of output-neuron i's row)
    direction V_i = W_i / ||W_i||    (a unit vector: where the row points)

DoRA's observation: full fine-tuning changes directions and magnitudes in
loosely independent ways, but plain LoRA is forced to change them together
(one BA term moves both). So DoRA splits them explicitly:

    W' = m * (W + s*BA) / ||W + s*BA||_row

  * the LoRA part (A, B) only steers the DIRECTION — whatever length
    W + s*BA ends up with is normalized away, row by row;
  * a new trainable vector m [out] (one number per output neuron) owns
    the MAGNITUDE, initialized to ||W||_row so that step 0 is exactly W.

Cost: the same A and B as LoRA, plus just `out` extra numbers for m.

Hand-check (worked in by_hand.py): with W row [3, 4], ||row|| = 5.
If LoRA nudges the row to [3, 4] + [1.5, 0] = [4.5, 4] (length ~6.02),
DoRA re-normalizes it back to length m = 5 unless m itself learns to grow.
"""

# torch for tensors; nn for Module/Parameter; F.linear for the y = xW^T + b op.
import torch
from torch import nn
import torch.nn.functional as F

# Reuse everything LoRA already set up (frozen base, A, B, scale, enabled).
from lora import LoRAConfig, LoRALinear


# DoRA IS a LoRALinear plus one extra trainable vector (the magnitudes).
class DoRALinear(LoRALinear):

    # Same constructor arguments as LoRALinear.
    def __init__(self, base: nn.Linear, cfg: LoRAConfig):
        # Build the LoRA half first: frozen base + A (random) + B (zeros).
        super().__init__(base, cfg)
        # Add the magnitude vector m, one entry per output row. We seed it with
        # the base rows' current lengths (||W_i|| for each row i) so that at
        # init  m * W/||W|| == W  exactly — step 0 changes nothing.
        # .clone() detaches it from the frozen base weight's storage.
        self.dora_mag = nn.Parameter(base.weight.norm(dim=1).clone())

    # Assemble the full DoRA weight: renormalize (W + s*BA) row-wise, then
    # rescale each row to the learned length m_i.
    def merged_weight(self) -> torch.Tensor:
        # First form the LoRA-updated weight  W + s*B@A  (shape [out, in]).
        # delta_weight() is inherited from LoRALinear and returns s*B@A.
        W = self.base.weight + self.delta_weight()
        # Compute each row's length ||row|| -> [out, 1]. We .detach() it so the
        # gradient flows into the *direction* of W, not through the normalizer;
        # this matches the DoRA paper and stabilizes training.
        norm = W.norm(dim=1, keepdim=True).detach()
        # m_i * (row_i / ||row_i||): unsqueeze m to [out,1] so it scales rows.
        return self.dora_mag.unsqueeze(1) * W / norm

    # Forward pass: apply the composed weight as an ordinary linear map.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If the adapter is off, fall back to the plain frozen base map.
        if not self.enabled:
            return self.base(x)
        # y = x @ merged_weight.T + base.bias, computed by F.linear.
        return F.linear(x, self.merged_weight(), self.base.bias)

    # Bake the composed weight into the base so inference needs no adapter.
    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Bake m and the normalized direction into a plain Linear."""
        # Overwrite the base weight in place with the full DoRA weight.
        self.base.weight.copy_(self.merged_weight())
        # Turn the adapter off: the base now already IS the composed weight,
        # so forward() must not re-apply DoRA on top of it.
        self.enabled = False
        # Hand back the updated base Linear.
        return self.base
