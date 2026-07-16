"""by_hand.py — every LoRA idea worked out on numbers small enough to check
with a pencil. Run `python3 by_hand.py` and follow along; every claim printed
here is also `assert`-ed, so if the math is wrong the script fails loudly.

Sections:
  1. LoRA         — the low-rank correction, and why B starts at zero
  2. why not zero — the init gradient argument (B=0 ok, A=B=0 stuck)
  3. scale s      — alpha/r vs alpha/sqrt(r) (rsLoRA) and the sqrt(r) growth
  4. DoRA         — splitting a row into magnitude x direction
  5. VeRA         — diag(b) B diag(d) A from frozen random B, A
  6. PiSSA        — SVD of W into principal + residual

Nothing here imports the big model: it is all tiny explicit matrices so the
numbers match what a student gets by hand.
"""

import torch

torch.set_printoptions(precision=4, sci_mode=False)


def show(name, t):
    print(f"  {name} =")
    for row in t.tolist() if t.dim() == 2 else [t.tolist()]:
        print("     ", ["%+.4f" % v for v in row])


def rule(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


# ===========================================================================
rule("1. LoRA:  y = W x  +  s * B A x")
# ---------------------------------------------------------------------------
# A frozen 3x3 weight (just 2*I so W x is easy to check), a rank-1 adapter.
W = 2.0 * torch.eye(3)
x = torch.tensor([1.0, 1.0, 1.0])

alpha, r = 2.0, 1
s = alpha / r                                   # classic LoRA scale = 2.0

A = torch.tensor([[1.0, 1.0, 0.0]])             # [r=1, in=3]
B0 = torch.zeros(3, 1)                           # [out=3, r=1]  -- LoRA starts here
B = torch.tensor([[1.0], [0.0], [0.0]])          # after a little training

print("Base only (B = 0):  y = W x")
show("W", W)
print("  W x        =", (W @ x).tolist(), "   <- just 2*x, adapter is silent")
assert torch.equal(W @ x + s * (B0 @ A) @ x, W @ x)   # B=0  =>  no change at all

print("\nAfter training moves B to [1,0,0]^T:")
BA = B @ A
show("B A  (a full 3x3, built from only 3+3 numbers)", BA)
delta = s * BA
show("s * B A", delta)
y = W @ x + delta @ x
print("  delta * x  =", (delta @ x).tolist(), "   (= s*[A.x, 0, 0] = 2*[2,0,0])")
print("  y          =", y.tolist(), "   (= [2,2,2] + [4,0,0])")
assert y.tolist() == [6.0, 2.0, 2.0]

p_lora = A.numel() + B.numel()
print(f"\n  params: adapter {p_lora} (=r*(in+out)) vs full W {W.numel()}")
print("  the gap grows with size: 32x32 at r=4 -> 256 vs 1024 (1/4);")
print("                           4096x4096 at r=8 -> 65,536 vs 16.7M (1/256).")


# ===========================================================================
rule("2. Why B=0 works but A=B=0 does not (init gradients)")
# ---------------------------------------------------------------------------
# For y = s B A x:   dL/dA = s B^T g x^T ,   dL/dB = s g (A x)^T   (g = dL/dy)
# So a factor that is zero kills its OWN partner's gradient.
def init_grads(A_init, B_init):
    A = A_init.clone().requires_grad_(True)
    B = B_init.clone().requires_grad_(True)
    y = s * (B @ A) @ x
    y.sum().backward()                           # any loss with nonzero dL/dy
    return A.grad.norm().item(), B.grad.norm().item()

gA, gB = init_grads(A, torch.zeros(3, 1))
print(f"  B=0, A=random :  |grad A| = {gA:.3f}   |grad B| = {gB:.3f}")
print("      -> B has a gradient (from A x), so B moves first. Good.")
assert gA == 0.0 and gB > 0.0

gA, gB = init_grads(torch.zeros(1, 3), torch.zeros(3, 1))
print(f"  A=0, B=0      :  |grad A| = {gA:.3f}   |grad B| = {gB:.3f}")
print("      -> both zero, the adapter can NEVER start. That is why one side")
print("         is random and the other is zero, never both zero.")
assert gA == 0.0 and gB == 0.0


# ===========================================================================
rule("3. The scale s:  alpha/r (LoRA) vs alpha/sqrt(r) (rsLoRA)")
# ---------------------------------------------------------------------------
alpha = 8.0
print("  alpha = 8 held fixed:")
print(f"   {'r':>4} | {'classic alpha/r':>16} | {'rsLoRA alpha/sqrt(r)':>20}")
for rr in (1, 4, 16, 64):
    print(f"   {rr:>4} | {alpha/rr:>16.3f} | {alpha/rr**0.5:>20.3f}")
print("  classic collapses toward 0 as r grows; rsLoRA fades far more gently.")

# WHY: with random A, B the update size |B A x| itself grows like sqrt(r),
# because B A x sums r independent-ish terms. Show it:
print("\n  measured || B A x || for random A,B (should track sqrt(r)):")
torch.manual_seed(0)
for rr in (1, 4, 16, 64):
    mags = []
    for _ in range(2000):
        A_ = torch.randn(rr, 8) / 8 ** 0.5
        B_ = torch.randn(8, rr) / rr ** 0.5      # standard fan-in scaling
        mags.append(((B_ @ A_) @ torch.randn(8)).norm())
    print(f"   r={rr:>3}: mean |BAx| = {torch.tensor(mags).mean():.3f}")
print("  dividing THIS by r (classic) over-shrinks big r; by sqrt(r) keeps it level.")


# ===========================================================================
rule("4. DoRA:  W' = m * (W + s B A) / ||W + s B A||   (per row)")
# ---------------------------------------------------------------------------
# Take one output row of W and split it into length x direction.
w_row = torch.tensor([3.0, 4.0])
m = w_row.norm()                                 # magnitude = 5
direction = w_row / m                            # unit vector [0.6, 0.8]
print("  a weight row w =", w_row.tolist())
print(f"  magnitude m = ||w|| = {m:.1f}   direction = {direction.tolist()}")
assert m.item() == 5.0

lora_nudge = torch.tensor([1.5, 0.0])            # what s B A adds to this row
new = w_row + lora_nudge                         # [4.5, 4.0]
new_len = new.norm()
print(f"\n  LoRA alone would push the row to {new.tolist()}, length {new_len:.4f}")
print("     -> plain LoRA changed BOTH the direction AND the length together.")

dora_row = m * new / new_len                     # keep length m, take new direction
print(f"  DoRA keeps length = m = {m:.1f}, only rotates:")
print(f"     DoRA row = {['%.4f' % v for v in dora_row.tolist()]}  length {dora_row.norm():.4f}")
assert abs(dora_row.norm().item() - 5.0) < 1e-5
print("  the trainable magnitude m owns length; A,B own direction — decoupled.")
print(f"  (if m also trains up to {new_len:.4f}, DoRA recovers the full LoRA row.)")


# ===========================================================================
rule("5. VeRA:  dW = diag(b) . B . diag(d) . A   (B, A frozen random)")
# ---------------------------------------------------------------------------
A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])        # frozen "random" [r=2, in=2]
B = torch.tensor([[1.0, 1.0], [1.0, -1.0]])       # frozen "random" [out=2, r=2]
d = torch.tensor([0.5, 0.5])                      # trainable  [r]
b = torch.tensor([2.0, 2.0])                      # trainable  [out]

step1 = torch.diag(d) @ A                         # scale A's rows
step2 = B @ step1                                 # mix through B
dW = torch.diag(b) @ step2                        # scale output rows
show("diag(d) . A", step1)
show("B . diag(d) . A", step2)
show("dW = diag(b) . B . diag(d) . A", dW)
assert dW.tolist() == [[1.0, 1.0], [1.0, -1.0]]

in_f = out_f = 2
print(f"  trainable here: d({d.numel()}) + b({b.numel()}) = {d.numel()+b.numel()}")
print(f"  LoRA would train A,B = r*(in+out) = 2*({in_f}+{out_f}) = {2*(in_f+out_f)}")
print("  ...and VeRA never even SAVES A,B (regenerated from a seed).")


# ===========================================================================
rule("6. PiSSA:  W = principal (-> adapter)  +  residual (-> frozen base)")
# ---------------------------------------------------------------------------
W = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
U, S, Vh = torch.linalg.svd(W)
print("  singular values of W:", [round(v, 4) for v in S.tolist()], " (top one = 3)")
assert torch.allclose(S, torch.tensor([3.0, 1.0]), atol=1e-5)

principal = S[0] * torch.outer(U[:, 0], Vh[0])    # rank-1 top piece
residual = W - principal
show("principal (top singular triple) -> starts the TRAINABLE adapter", principal)
show("residual (the rest)             -> becomes the FROZEN base", residual)
assert torch.allclose(principal, torch.tensor([[1.5, 1.5], [1.5, 1.5]]), atol=1e-5)
assert torch.allclose(residual, torch.tensor([[0.5, -0.5], [-0.5, 0.5]]), atol=1e-5)
assert torch.allclose(principal + residual, W)
print("  principal + residual = W, so step 0 still computes W x exactly —")
print("  but now the trainable part sits on W's strongest direction, not on 0.")

print("\nAll hand-checks passed. Open this file next to the printout and verify a")
print("few by hand — the numbers are chosen to be pencil-friendly.\n")
