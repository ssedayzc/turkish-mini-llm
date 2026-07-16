# gemma4 — Gemma-style decoder

Same letter-by-letter task, but with Gemma's distinctive design choices.

**What to notice while reading:**
- `block.py` — **sandwich norm**: an RMSNorm *before and after* each sub-layer.
- `model.py` — **local vs global** attention: 5 local sliding-window layers per 1 global
  layer; local layers use a small RoPE base (10k), global a large one (1M). Also
  **embedding scaling** by `sqrt(hidden_size)` and small **per-layer embeddings**.
- `rms_norm.py` — weight starts at 0 and is applied as **`(1 + weight)`** (a Gemma quirk).
- `mlp.py` — **GeGLU** (gelu gate) instead of SwiGLU.
- `attention.py` — QK-Norm + explicit query scaling + an additive mask (causal, plus the
  sliding window for local layers). `train.py` prints which layers are local/global.

**On "Gemma 4" specifically:** public details are limited. This folder implements the
well-documented Gemma-family core plus a *simplified* Per-Layer Embedding as a nod to the
Gemma 4 direction. Larger Gemma 4 ideas (MatFormer nesting, MoE experts, proportional
RoPE) are intentionally **omitted** to keep the code readable.

Run: `python3 train.py` then `python3 generate.py 20`.
