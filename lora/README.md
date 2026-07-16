# lora — LoRA and its subtypes, small enough to calculate by hand

Low-Rank Adaptation and four of its variants (**LoRA, rsLoRA, DoRA, VeRA, PiSSA**),
implemented from scratch so every idea can be
**read in one file, watched while it trains in seconds, and checked on paper**.

This folder does **not** re-implement any transformer. The adapters are
**architecture-agnostic** — the exact same code specializes **any** of the repo's
tiny models ([`qwen3`](../qwen3/), [`qwen3_5`](../qwen3_5/), [`gemma4`](../gemma4/),
[`deepseek3`](../deepseek3/)). It imports the chosen base from its sibling folder and
only adds the adapters, because the whole point of LoRA is that you _keep the base
model and bolt something tiny onto it_.

> **The one-sentence idea.** Instead of retraining a weight matrix `W` (that's
> `in × out` numbers), freeze it and learn a **low-rank correction** beside it —
> two skinny matrices whose product is the same shape as `W` but costs only
> `r × (in + out)` numbers. On qwen3 that turns **19,584 frozen parameters**
> into as few as **704 trainable ones**, and it still fully re-steers the model.

> **Why it works on every model unchanged.** `inject()` walks the model, finds
> `nn.Linear` layers by name, and swaps them for adapter-wrapped versions. Every
> architecture here is built from `nn.Linear`s, so nothing about the adapter cares
> whether the attention around it is GQA (qwen3, gemma), Gated DeltaNet (qwen3.5),
> or MLA (deepseek3). Only the *names* of the good target layers differ, and those
> live in one registry ([`base_model.py`](base_model.py)).

---

## Table of contents

1. [The core idea (LoRA)](#1-the-core-idea-lora)
2. [Why `B = 0` and `A = random` (never both zero)](#2-why-b--0-and-a--random-never-both-zero)
3. [The scale `s = alpha / r`, and rsLoRA](#3-the-scale-s--alpha--r-and-rslora)
4. [The five methods at a glance](#4-the-five-methods-at-a-glance)
5. [Each method in detail](#5-each-method-in-detail)
6. [Merging and the on/off switch](#6-merging-and-the-onoff-switch)
7. [File-by-file guide](#7-file-by-file-guide)
8. [How to run it](#8-how-to-run-it)
9. [Results on the toy task](#9-results-on-the-toy-task)
10. [`by_hand.py` — the pencil-and-paper checks](#10-by_handpy--the-pencil-and-paper-checks)
11. [What is simplified](#11-what-is-simplified)
12. [Exercises for students](#12-exercises-for-students)

---

## 1. The core idea (LoRA)

A linear layer computes `y = W x`, where `W` has shape `[out, in]`. Fine-tuning
normally nudges **every entry** of `W`. LoRA (Hu et al., 2021) asks: _do we really
need all those degrees of freedom to specialize a model?_ Empirically, no — the
useful update `ΔW` tends to be **low rank**. So LoRA freezes `W` and writes the
update as a product of two skinny matrices:

```
        ┌──────────── frozen ────────────┐   ┌──── trainable ────┐
  y  =            W x                       +      s · B A x

  W : [out, in]   (frozen, never updated)
  A : [r,   in]   (trainable)   r is the "rank", the whole knob
  B : [out, r ]   (trainable)
  s : scalar      = alpha / r   (see §3)
```

`B @ A` is an `[out, in]` matrix — **exactly the shape of `W`** — but it is built
from only `r·(in + out)` numbers instead of `in·out`. For `r ≪ min(in, out)` that
is a tiny fraction:

| matrix `W`    | full `in·out` | LoRA `r·(in+out)` at `r=4` | ratio   |
| ------------- | ------------- | -------------------------- | ------- |
| `32 × 32`     | 1,024         | 256                        | 1 / 4   |
| `4096 × 4096` | 16,777,216    | 65,536 (`r=8`)             | 1 / 256 |

The forward pass never actually forms `B @ A`. It multiplies right-to-left, so the
big `[out, in]` matrix is never materialized ([`lora.py`](lora.py), `forward`):

```python
y + s * (x @ lora_A.T @ lora_B.T)
#         [.,in]@[in,r] = [.,r]        squeeze down to r dims
#                      @[r,out] = [.,out]  lift back up to out dims
```

### The worked example (`by_hand.py`, section 1)

Take `W = 2·I` (3×3), a rank-1 adapter, `alpha = 2`, `r = 1` so `s = 2`, and input
`x = [1, 1, 1]`:

```
A = [[1, 1, 0]]        (shape [1, 3])
B = [[1], [0], [0]]    (shape [3, 1])   ← after a little training

B @ A = [[1,1,0],      s·B@A = [[2,2,0],       W x        = [2, 2, 2]
         [0,0,0],               [0,0,0],       (s·B@A) x  = [4, 0, 0]
         [0,0,0]]               [0,0,0]]       y = Wx + …  = [6, 2, 2]
```

Every number here is checkable by hand — that is the point.

---

## 2. Why `B = 0` and `A = random` (never both zero)

At initialization LoRA sets **`B = 0`** and **`A = random`** (Kaiming-uniform, the
same init a fresh `nn.Linear` uses). Two things must both be true, and this is the
only way to get both:

**(a) Start equal to the base model.** Because `B = 0`, we have `s·B@A = 0`, so at
step 0 the wrapped model computes exactly `W x`. Fine-tuning begins from the
pretrained model, not from a random perturbation of it.

**(b) Be able to move.** For `y = s·B·A·x`, the gradients are

```
dL/dA = s · Bᵀ · g · xᵀ         dL/dB = s · g · (A x)ᵀ        (g = dL/dy)
```

Each factor gates its **partner's** gradient. So:

| init            | `dL/dA` | `dL/dB` | result                                    |
| --------------- | ------- | ------- | ----------------------------------------- |
| `B=0, A=random` | `0`     | ≠ `0`   | ✅ `B` moves first (driven by `A x`)      |
| `A=0, B=0`      | `0`     | `0`     | ❌ both stuck forever — nothing can start |

`by_hand.py` section 2 verifies both rows numerically with autograd. This is why
one side is zero and the other is random — **never both zero**.

---

## 3. The scale `s = alpha / r`, and rsLoRA

The correction is multiplied by `s = alpha / r`. Why divide by `r` at all? So that
`alpha` — the strength you actually tuned — stays meaningful when you change the
rank. But this classic choice **over-damps high ranks**.

Here is why. With random `A, B`, the vector `B A x` is a sum of `r` roughly
independent terms, so its length grows like **`√r`**, not `r`. `by_hand.py`
section 3 measures exactly this:

```
measured ‖B A x‖ for random A,B   (r →  ‖BAx‖)
   r=1: 2.13     r=16: 2.62
   r=4: 2.52     r=64: 2.65      ← grows sub-linearly, ~√r then saturates
```

So dividing by `r` shrinks big-`r` updates too aggressively. **rsLoRA**
(rank-stabilized LoRA, Kalajdzievski 2023) fixes it with a one-character change:

```
classic LoRA :  s = alpha / r          → collapses toward 0 as r grows
rsLoRA       :  s = alpha / sqrt(r)     → fades far more gently
```

| `r` | `alpha/r` (classic) | `alpha/√r` (rsLoRA), `alpha=8` |
| --- | ------------------- | ------------------------------ |
| 1   | 8.000               | 8.000                          |
| 4   | 2.000               | 4.000                          |
| 16  | 0.500               | 2.000                          |
| 64  | 0.125               | 1.000                          |

In code it is literally the single line in [`lora.py`](lora.py):

```python
self.scale = (cfg.alpha / math.sqrt(cfg.r)) if cfg.rank_stabilized \
             else (cfg.alpha / cfg.r)
```

---

## 4. The five methods at a glance

All five freeze the base weight `W`. They differ in _what small thing_ they add and
_how they initialize it_.

| method     | what it adds                                      | trainable / layer  | key idea                                             | file                   |
| ---------- | ------------------------------------------------- | ------------------ | ---------------------------------------------------- | ---------------------- |
| **LoRA**   | `A [r,in]`, `B [out,r]`                           | `r·(in+out)`       | low-rank correction, `s = alpha/r`                   | [`lora.py`](lora.py)   |
| **rsLoRA** | same as LoRA                                      | `r·(in+out)`       | only the scale changes: `s = alpha/√r`               | [`lora.py`](lora.py)   |
| **DoRA**   | LoRA + magnitude vector `m [out]`                 | `r·(in+out) + out` | split each weight row into **length × direction**    | [`dora.py`](dora.py)   |
| **VeRA**   | trainable `d [r]`, `b [out]`; `A,B` frozen random | `r + out`          | don't train `A,B` at all — only rescale them         | [`vera.py`](vera.py)   |
| **PiSSA**  | same tensors as LoRA                              | `r·(in+out)`       | **initialize** `A,B` from `W`'s top singular vectors | [`pissa.py`](pissa.py) |

---

## 5. Each method in detail

### LoRA — [`lora.py`](lora.py)

The base class everything else builds on. `LoRALinear` wraps one frozen
`nn.Linear`, freezes its weight, and adds `lora_A` (random) and `lora_B` (zeros).
The correction is `s · B A`. `delta_weight()` returns that `[out, in]` matrix for
inspection and merging; `forward()` computes it the cheap way.

### rsLoRA — [`lora.py`](lora.py), `rank_stabilized=True`

Not a separate class. It is LoRA with `s = alpha/√r` (§3). Turn it on with
`LoRAConfig(rank_stabilized=True)` or `python3 train.py rslora`.

### DoRA — [`dora.py`](dora.py)

Weight-Decomposed LoRA (Liu et al., 2024). Every weight matrix can be split
**row by row** into a magnitude (the row's length) and a direction (a unit vector):

```
W' = m ⊙ (W + s·B·A) / ‖W + s·B·A‖_row
     └┬┘  └──────┬──────┘  └────┬────┘
   trainable    LoRA steers    row length is
   length m     the direction  normalized away
```

- `A, B` (as in LoRA) change only the **direction** of each row — whatever length
  `W + s·B·A` happens to have is divided out per row.
- A new trainable vector `m` (length `out`, one number per output neuron) owns the
  **magnitude**, initialized to `‖W‖_row` so step 0 is exactly `W`.

The paper detaches the norm during backprop (gradients flow into the _direction_,
not through the normalizer); we do the same with `.detach()` in `merged_weight()`.

**Hand-check (`by_hand.py` §4).** Row `w = [3, 4]`, `‖w‖ = 5`. If LoRA nudges it to
`[4.5, 4]` (length ≈ 6.02), DoRA renormalizes back to length `5`:
`5 · [4.5, 4] / 6.02 = [3.737, 3.322]`, which has length exactly 5 again — until the
trainable `m` itself grows.

### VeRA — [`vera.py`](vera.py)

Vector-based Random-matrix Adaptation (Kopiczko et al., 2024). The most aggressive
compression: **don't train `A` and `B` at all.** Freeze them as _random_ matrices
and train only two small scaling vectors:

```
y = W x + b ⊙ ( B ( d ⊙ (A x) ) )      A [r,in]  frozen random
                                        B [out,r] frozen random
                                        d [r]     trainable, init 0.1
                                        b [out]   trainable, init 0 → silent start
```

Equivalently `ΔW = diag(b) · B · diag(d) · A`: the frozen random matrices are a
fixed bank of directions, and `d`, `b` learn how much of each to mix.

Because `A, B` are never trained, they are not even **saved** — we regenerate them
from a fixed seed (`shared_random`), so an adapter file holds only `d` and `b`.

> **VeRA runs at high rank on purpose.** Its trainable count is `r + out`, and the
> `r` part is tiny, so raising `r` is nearly free. `train.py` therefore uses
> `r = 64` for VeRA (vs `r = 4` for the others) — that is the intended trade:
> **704** trainable numbers, and it still reaches ~98% on the task. At `r=4` VeRA
> has only 224 numbers and noticeably underfits; scaling `r` is the fix.

**Hand-check (`by_hand.py` §5).** With `A = I`, `B = [[1,1],[1,-1]]`, `d = [.5,.5]`,
`b = [2,2]`, the product `diag(b)·B·diag(d)·A` works out to `[[1,1],[1,-1]]` — and
VeRA trains **4** numbers where LoRA would train **8**.

### PiSSA — [`pissa.py`](pissa.py)

Principal Singular values and Singular vectors Adaptation (Meng et al., 2024).
**Not a new layer — a smarter initialization for LoRA.** Classic LoRA starts the
adapter at zero and must _discover_ useful directions. PiSSA instead starts the
adapter on the directions where `W` already does most of its work.

Take the SVD `W = U S Vᵀ` (singular values largest first) and split off the top `r`:

```
W  =  U_r S_r V_rᵀ   +   residual
      └────┬────┘        └───┬───┘
   principal part:        the rest:
   → STARTS the adapter    → becomes the FROZEN base
   B0 = U_r √(S_r/s)        W ← W − s·B0·A0
   A0 = √(S_r/s) V_rᵀ
```

At step 0, `residual + s·B0·A0 = W`, so the model still computes `W x` — but now the
**trainable** subspace is `W`'s dominant one, so gradients immediately push on
directions that matter. Training then proceeds as ordinary LoRA.

**Hand-check (`by_hand.py` §6).** `W = [[2,1],[1,2]]` has singular values `3, 1`. The
rank-1 principal part is `[[1.5,1.5],[1.5,1.5]]`, the residual `[[.5,-.5],[-.5,.5]]`,
and they add back to `W`.

> **One honest catch.** PiSSA _edits the frozen base_ (`W ← residual`). So "adapter
> off" is the residual, **not** the original model (you can see this in `train.py`'s
> output: PiSSA's off-rate is 0%, not the base rate). To reload it later the adapter
> file must also carry `A0, B0`; `load_adapter` replays `W ← W − s·B0·A0` on a fresh
> base before dropping the trained `A, B` on top. [`inject.py`](inject.py) handles
> this automatically.

---

## 6. Merging and the on/off switch

Two structural promises of LoRA, both checked in `train.py` on every run:

**Disable → base returns.** Every adapter has an `enabled` flag.
`set_adapters(model, False)` makes each `forward` skip the correction, so you get
the exact frozen base back (except PiSSA, which edited the base — see above).

**Merge → zero inference cost.** `merge_adapters(model)` folds each correction into
its base weight (`W ← W + s·B·A`, or the composed weight for DoRA) and switches the
adapter off, so a merged model is a plain model — **no extra layers, no extra
latency at inference.** `train.py` verifies this: it compares logits before and
after merging and prints `max|logit diff|`, which is `~1e-5` or smaller every time
(exactly `0` for DoRA):

```
merge check: max|logit diff| = 3.53e-05  (OK, identical)
```

That tiny residual is float32 rounding, not a modeling difference.

---

## 7. File-by-file guide

Every source file is commented **line by line** — read the code top to bottom.

| file                           | what it is                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| [`lora.py`](lora.py)           | `LoRAConfig` + `LoRALinear` (LoRA and rsLoRA). The base every other adapter subclasses.  |
| [`dora.py`](dora.py)           | `DoRALinear` — `LoRALinear` plus the trainable magnitude vector.                         |
| [`vera.py`](vera.py)           | `VeRALinear` + `shared_random` — frozen random `A,B`, trainable `d,b`.                   |
| [`pissa.py`](pissa.py)         | `pissa_init` / `pissa_apply_residual` — the SVD-based re-init and its reload fixup.      |
| [`inject.py`](inject.py)       | swap targeted `Linear`s for adapters, freeze the rest, count params, save/load, inspect. |
| [`base_model.py`](base_model.py) | the **model registry**: name → folder / class / tokenizer / default targets. The only model-specific code. |
| [`by_hand.py`](by_hand.py)     | **the centerpiece:** every idea worked on pencil-friendly numbers, each `assert`-ed.     |
| [`pretrain.py`](pretrain.py)   | train a base model on all 921 names → `base_<model>.pt` (works for any architecture).    |
| [`train.py`](train.py)         | freeze `base_<model>.pt`, inject an adapter, fine-tune only it, report + verify.          |
| [`generate.py`](generate.py)   | load the base + a saved adapter and generate names (the adapter records its own arch).    |

Nothing in the adapter files (`lora/dora/vera/pissa/inject`) imports a transformer —
they are pure `nn.Linear` surgery. Only `pretrain/train/generate` touch a model, and
they get it from [`base_model.py`](base_model.py), which imports it from the matching
sibling folder.

### How to adapt a model this repo does not list yet

1. Add a row to `MODELS` in [`base_model.py`](base_model.py): `name → (folder, class, targets)`.
2. Run `python3 train.py <name> --list` to see every `nn.Linear` and its shape.
3. Put the projection names you want in that row's `targets` (or pass `targets="all"`
   in a `LoRAConfig` to adapt every `Linear` except the tied `lm_head`).

That is the entire porting cost — the adapter math never changes.

---

## 8. How to run it

```bash
cd lora

# 0. see all the math verified on tiny numbers (no training, ~1 second)
python3 by_hand.py

# 1. train a base model once (the "expensive" model we then freeze).
#    First arg = which architecture; default qwen3.  → base_<model>.pt
python3 pretrain.py              # base_qwen3.pt
python3 pretrain.py gemma4       # base_gemma4.pt
python3 pretrain.py deepseek3    # base_deepseek3.pt   (MLA + MoE, a bit slower)

# 2. adapt it. Args are positional: model, method, target letter (all optional).
python3 train.py qwen3 lora z        # qwen3,   LoRA,  steer names to start with 'z'
python3 train.py gemma4 dora z       # gemma,   DoRA
python3 train.py deepseek3 lora s    # deepseek (MLA targets), letter 's'
python3 train.py qwen35 vera z       # qwen3.5, VeRA (auto-uses r=64)
python3 train.py qwen3 pissa z       # qwen3,   PiSSA
python3 train.py qwen3 --list        # just print the model's Linear layers, then exit

# 3. load the base + a saved adapter and generate (adapter knows its own arch)
python3 generate.py adapter_qwen3_lora_z.pt 20
python3 generate.py adapter_qwen3_lora_z.pt 20 base   # same base, adapter OFF, for contrast
```

The adaptation task is deliberately eyeball-checkable: the base picks a first letter
roughly by frequency (it starts a name with `z` about **2%** of the time), and the
adapter re-steers it to **~100% `z`** — while touching only the adapter's few hundred
numbers.

---

## 9. Results on the toy task

**Same method (LoRA `r=4`), every architecture.** Task: steer the frozen base (which
starts a name with `z` ~2–3% of the time) to start names with `z`. In every case the
whole base stays frozen and only the attention-projection adapters train:

| model         | attention | base params | adapter params | `z`-rate (base → adapted) | merge check  |
| ------------- | --------- | ----------- | -------------- | ------------------------- | ------------ |
| **qwen3**     | GQA       | 19,584      | 1,792 (8.4%)   | 2.0% → 100.0%             | `3.5e-05` ✅ |
| **gemma4**    | GQA + SWA | 62,912      | 5,376 (7.9%)   | 1.0% → 99.0%              | `3.0e-05` ✅ |
| **deepseek3** | MLA       | 47,976      | 2,400 (4.8%)   | 3.0% → 99.0%              | `2.8e-05` ✅ |

Adapted samples: qwen3 `zümrüt, zerin, zeliha`; gemma4 `zehra, zeliha, zümrüt`;
deepseek3 `zeliha, zekeriya, zülfiye` — all real Turkish `z`-names, from adapters
smaller than this README.

**All five methods on qwen3.** Base **19,584** params, all frozen:

| method     | rank | trainable params | % of model | `z`-rate after | adapter file | merge check  |
| ---------- | ---- | ---------------- | ---------- | -------------- | ------------ | ------------ |
| base       | —    | —                | —          | 2.0%           | —            | —            |
| **LoRA**   | 4    | 1,792            | 8.4%       | 100.0%         | 13.1 KB      | `3.5e-05` ✅ |
| **rsLoRA** | 4    | 1,792            | 8.4%       | 99.5%          | 13.1 KB      | `4.0e-05` ✅ |
| **DoRA**   | 4    | 1,984            | 9.2%       | 99.5%          | 16.0 KB      | `0.0e+00` ✅ |
| **VeRA**   | 64   | 704              | 3.5%       | 97.5%          | 8.7 KB       | `6.2e-05` ✅ |
| **PiSSA**  | 4    | 1,792            | 8.4%       | 99.0%          | 24.5 KB      | `4.3e-05` ✅ |

_(Numbers are from a fixed seed on CPU; yours will match closely.)_

---

## 10. `by_hand.py` — the pencil-and-paper checks

This is the file to open next to a calculator. It imports **no model** — just tiny
explicit matrices — and prints every intermediate value, then `assert`s the result,
so if any claim in this README is wrong the script fails. Sections:

1. **LoRA** — `y = Wx + s·B·A·x`, and that `B=0` means the adapter is silent.
2. **Init gradients** — `B=0,A=random` can move; `A=0,B=0` is stuck (shown via autograd).
3. **Scale `s`** — `alpha/r` vs `alpha/√r`, plus the measured `√r` growth of `‖BAx‖`.
4. **DoRA** — one row split into magnitude × direction and renormalized.
5. **VeRA** — `diag(b)·B·diag(d)·A` multiplied out by hand.
6. **PiSSA** — the SVD of `W` into principal + residual.

```bash
python3 by_hand.py     # prints every step; all asserts must pass
```

---

## 11. What is simplified

Faithful to the folder's "signature idea in the simplest correct form" rule, these
are intentionally left out:

- **Quantization.** Real QLoRA/QDoRA keep the frozen base in 4-bit. Here the base is
  plain float32 — the memory story is about _trainable_ params, not base storage.
- **Adapter dropout, bias terms, and per-layer ranks** exist in `LoRAConfig` but are
  kept at defaults; production setups tune them per layer.
- **VeRA** shares _one_ random `A,B` pair across layers in the paper; we regenerate a
  per-shape pair from the same seed (same spirit: random, frozen, reproducible).
- **DoRA** here decomposes the full weight; the paper applies the decomposition to
  the same target projections and is otherwise identical.
- **Targets** default to each architecture's attention projections (`q,k,v,o` for
  qwen3/qwen35/gemma, `q_up,kv_up,o_proj` for deepseek's MLA — see
  [`base_model.py`](base_model.py)). Adapting the MLP (`gate,up,down`) or everything
  (`targets="all"`) is a one-line change to the `LoRAConfig`.

---

## 12. Exercises for students

1. **Rank sweep.** Change `rank` in `train.py` to `1, 2, 8`. At what rank does the
   `z`-rate stop improving? Relate that to `by_hand.py` §1's parameter count.
2. **Both-zero init.** In `lora.py`, initialize `lora_A` to zeros as well. Train and
   watch it never leave the base — then connect it to `by_hand.py` §2.
3. **rsLoRA at high rank.** Run `lora` vs `rslora` at `rank=64` (edit `train.py`).
   Which one still learns? Tie it to the `alpha/r` vs `alpha/√r` table in §3.
4. **VeRA capacity.** Set VeRA's rank back to `4`. How far does the `z`-rate fall?
   Now try `128`. What does that tell you about where VeRA's capacity comes from?
5. **Merge by hand.** For a single `q_proj`, print `W`, `s·B·A`, and `W + s·B·A`.
   Confirm `merge_adapters` produces exactly the third one.
6. **Compose two adapters.** Train `adapter_qwen3_lora_z.pt` and `adapter_qwen3_lora_s.pt`,
   then load both `delta_weight()`s and add them to one base. Do you get a model that
   starts names with `z` _or_ `s`? Why or why not?
7. **Same method, different attention.** Run `python3 train.py qwen3 lora z` and
   `python3 train.py deepseek3 lora z`. The adapter code is byte-for-byte identical;
   only the target names differ (GQA's `q,k,v,o` vs MLA's `q_up,kv_up,o_proj`). Use
   `--list` to see why.

---

**References:** Hu et al. 2021 (LoRA), Kalajdzievski 2023 (rsLoRA), Liu et al. 2024
(DoRA), Kopiczko et al. 2024 (VeRA), Meng et al. 2024 (PiSSA).
