"""PiSSA — Principal Singular values and Singular vectors Adaptation
(Meng et al., 2024). Not a new layer — a smarter INITIALIZATION for LoRA.

Classic LoRA starts with B = 0: the adapter begins as nothing and must
discover useful directions from scratch, while the "knowledge" sits frozen
in W. PiSSA flips who holds what. Take the SVD of the base weight,

    W = U S V^T     (S holds the singular values, largest first)

and split it into the top-r "principal" part plus the leftover:

    W = U_r S_r V_r^T  +  residual
        ^^^^^^^^^^^^^     ^^^^^^^^
        where W's energy   everything else
        is concentrated

Then:  * the adapter STARTS as that principal part:
           B0 = U_r sqrt(S_r/s),  A0 = sqrt(S_r/s) V_r^T   (s = alpha/r,
         so that s*B0@A0 reconstructs the principal part exactly)
       * the frozen base becomes the residual:  W <- W - s*B0@A0

At step 0 the model still computes exactly Wx (residual + principal = W),
but now the TRAINABLE part is the subspace where W already does most of
its work — so gradient steps immediately push on directions that matter,
instead of first having to find them. Training then proceeds as plain LoRA.

Hand-check (worked in by_hand.py): W = [[2,1],[1,2]] has singular values
3 and 1; the rank-1 principal part is [[1.5,1.5],[1.5,1.5]] and the
residual [[.5,-.5],[-.5,.5]] — add them back and you get W.

One catch for saving: PiSSA edits the frozen base weight, so the adapter
file must also carry (A0, B0) — at load time we redo  W <- W - s*B0@A0
on the fresh base checkpoint, then drop the trained A, B on top.
"""

# Only torch is needed: SVD lives in torch.linalg.
import torch

# PiSSA re-inits an existing LoRALinear, so we import the type for clarity.
from lora import LoRALinear


# Re-initialize a freshly-built LoRALinear in place, PiSSA style.
# no_grad: we are editing parameters/weights directly, outside autograd.
@torch.no_grad()
def pissa_init(m: LoRALinear) -> None:
    """Re-initialize a LoRALinear in place, PiSSA style."""
    # The frozen base weight we are about to decompose, shape [out, in].
    W = m.base.weight
    # r = the adapter rank (rows of A); s = the LoRA scale alpha/r already set.
    r, s = m.lora_A.shape[0], m.scale

    # Economy SVD: W = U diag(S) Vh, with U [out,k], S [k], Vh [k,in],
    # k = min(out,in). Singular values in S come sorted largest-first.
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # Split each of the top-r singular values evenly between A and B by taking
    # the square root of S_r/s, so that s * (U root)(root Vh) = U S_r Vh.
    root = (S[:r] / s).sqrt()

    # B0 = U_r * sqrt(S_r/s): take the first r left singular vectors (columns
    # of U), scale each column by root -> [out, r].
    m.lora_B.copy_(U[:, :r] * root)
    # A0 = sqrt(S_r/s) * V_r^T: take the first r right singular vectors (rows of
    # Vh), scale each row by root -> [r, in].
    m.lora_A.copy_(root.unsqueeze(1) * Vh[:r])
    # Make the frozen base the RESIDUAL: subtract the principal part we just put
    # into the adapter, so residual + s*B0@A0 == original W (step 0 unchanged).
    W -= m.delta_weight()

    # Save the starting adapter (A0, B0) as buffers so a *fresh* base checkpoint
    # can be turned back into this residual at load time (see inject.py).
    m.register_buffer("pissa_A0", m.lora_A.detach().clone())
    m.register_buffer("pissa_B0", m.lora_B.detach().clone())


# At load time: given a fresh (un-edited) base and the saved A0/B0, reproduce
# the residual the model was actually trained on.
@torch.no_grad()
def pissa_apply_residual(m: LoRALinear, A0: torch.Tensor, B0: torch.Tensor) -> None:
    """At load time: turn the fresh base weight back into the residual."""
    # W <- W - s*B0@A0, exactly the subtraction pissa_init did at train time.
    m.base.weight -= m.scale * (B0 @ A0)
    # Keep the A0/B0 around again (so this loaded model could be re-saved).
    m.register_buffer("pissa_A0", A0.clone())
    m.register_buffer("pissa_B0", B0.clone())
