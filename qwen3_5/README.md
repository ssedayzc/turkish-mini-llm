# qwen3_5 — Qwen3.5 hybrid

Same skeleton as [`qwen3/`](../qwen3/), with one big change: most layers use
**Gated DeltaNet** (linear attention) instead of softmax attention.

**The new file to read:** [`gated_deltanet.py`](gated_deltanet.py). It keeps a single
fixed-size memory matrix `S` and updates it one token at a time with the *gated delta rule*:

```
S_t = alpha_t * S_{t-1} + beta_t * (v_t - alpha_t * S_{t-1} k_t) k_t^T
```

- `alpha_t` decay gate — how much old memory to keep
- `beta_t` write gate — how strongly to write the correction
- the `(v_t - prediction)` term is the "delta" — store only what was predicted wrong

`config.full_attn_every = 4`, so every 4th layer is full attention and the rest are
Gated DeltaNet (`train.py` prints the layer types).

**Simplified for clarity:** the delta-rule recurrence is a plain Python loop over time
(real Qwen3.5 parallelizes it in chunks); no MoE; small dense sizes.

Run: `python3 train.py` then `python3 generate.py 20`.
