"""train.py — adapt any frozen base to a narrow new behavior with an adapter.

The base (base_<model>.pt) generates Turkish names starting with all sorts of
letters. Here we freeze it completely and train ONLY a small adapter so that
names come out starting with one chosen letter (default 'z', which the base
almost never picks). Nothing about the base changes — a few hundred adapter
numbers do all the steering. The SAME code works for every architecture.

Run  (args are positional: model, method, letter — all optional):
    python3 pretrain.py                 # once, makes base_qwen3.pt
    python3 train.py                    # qwen3, LoRA, letter 'z'
    python3 train.py gemma4 dora        # gemma, DoRA, letter 'z'
    python3 train.py deepseek3 lora s   # deepseek (MLA targets), letter 's'
    python3 train.py qwen35 vera z      # qwen3.5, VeRA
    python3 train.py qwen3 --list       # just print the model's Linear layers

Models : qwen3 | qwen35 | gemma4 | deepseek3     (see base_model.py)
Methods: lora | rslora | dora | vera | pissa
"""

# os/sys for CLI args + paths; torch for training.
import os
import sys
import torch

# Registry: name -> class / config / tokenizer / default targets / checkpoint.
from base_model import (model_class, tokenizer_class, default_targets,
                        base_checkpoint)
# Adapter config + the inject/merge/save/inspect helpers.
from lora import LoRAConfig
from inject import (inject, set_adapters, merge_adapters, print_parameter_report,
                    print_linear_names, save_adapter)

# The full names file (we filter it down to one starting letter below).
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "temiz_yaratik_isimleri.txt")
# Fine-tuning hyperparameters. STEPS is small — adapting is cheap.
BLOCK_SIZE, BATCH_SIZE, STEPS, LR = 16, 64, 800, 5e-3
device = "cpu"
torch.manual_seed(0)


# Build the narrow training corpus: only names starting with target_letter.
def make_corpus(tokenizer, target_letter):
    # Keep only names that begin with the target letter.
    names = [n for n in open(DATA_FILE, encoding="utf-8").read().split("\n")
             if n and n[0] == target_letter]
    # Join with newlines and pad so the first/last names also carry the
    # start/end-of-name marker the model learns from.
    text = "\n" + "\n".join(names) + "\n"
    return torch.tensor(tokenizer.encode(text), dtype=torch.long), len(names)


# Sample one batch of (input, shifted-target) windows.
def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x, y


# Generate n names and return them as decoded strings.
@torch.no_grad()
def sample(model, tokenizer, n=200):
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long)
    out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                         temperature=0.8, eos_id=tokenizer.eos_id)
    names = [tokenizer.decode(r[1:]).split("\n")[0] for r in out.tolist()]
    return [nm for nm in names if nm]


# Fraction of names whose first character is `letter` — our steering metric.
def hit_rate(names, letter):
    return sum(nm[0] == letter for nm in names) / max(len(names), 1)


def main():
    # Positional args: model name, method, target letter (each optional).
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    method = sys.argv[2] if len(sys.argv) > 2 else "lora"
    letter = sys.argv[3] if len(sys.argv) > 3 else "z"

    # Rebuild the pretrained base for this architecture.
    Model = model_class(model_name)
    Tokenizer = tokenizer_class(model_name)
    ckpt = torch.load(base_checkpoint(model_name), map_location="cpu", weights_only=False)
    tokenizer = Tokenizer(ckpt["chars"])
    model = Model(ckpt["cfg"]); model.load_state_dict(ckpt["model"]); model.eval()

    # `--list` as the method: just show every adaptable Linear and exit. Handy
    # for choosing targets on an architecture you have not adapted before.
    if method == "--list":
        print_linear_names(model)
        return
    assert method in ("lora", "rslora", "dora", "vera", "pissa"), method

    # --- base behavior, before any adapter -------------------------------
    base_names = sample(model, tokenizer)
    base_hr = hit_rate(base_names, letter)
    print(f"model={model_name}   method={method}   target first letter = '{letter}'")
    print(f"\nBASE (frozen, untouched):  '{letter}'-names {base_hr:5.1%}   "
          f"e.g. {base_names[:6]}")

    # --- bolt on the adapter, freeze everything else ---------------------
    # VeRA's count barely grows with rank (only the length-r vector d does), so
    # it is *meant* to run at high rank; the others pay r*(in+out) per layer and
    # stay small at r=4. Same alpha, so the comparison is fair.
    rank = 64 if method == "vera" else 4
    # Targets come from the registry (per architecture — MLA differs from GQA).
    lcfg = LoRAConfig(r=rank, alpha=8.0, targets=default_targets(model_name))
    adapted = inject(model, lcfg, method=method)
    print(f"\ninjected {method} on {adapted}")
    print_parameter_report(model)

    # --- train ONLY the adapter on the narrow corpus ---------------------
    data, n_names = make_corpus(tokenizer, letter)
    print(f"\nfine-tuning on {n_names} '{letter}'-names, {STEPS} steps "
          f"(only the adapter moves):")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    model.train()
    for step in range(1, STEPS + 1):
        x, y = get_batch(data)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")
    model.eval()

    # --- adapted behavior ------------------------------------------------
    ad_names = sample(model, tokenizer)
    print(f"\nADAPTED (adapter on):      '{letter}'-names {hit_rate(ad_names, letter):5.1%}   "
          f"e.g. {ad_names[:6]}")

    # --- promise 1: turning the adapter off restores the base ------------
    # (True for lora/rslora/dora/vera. PiSSA edits the base weight into the
    #  residual, so 'off' is the residual, not the original base — shown here.)
    set_adapters(model, False)
    off_hr = hit_rate(sample(model, tokenizer), letter)
    set_adapters(model, True)
    tag = "  <- PiSSA edits the base, so 'off' != original" if method == "pissa" else ""
    print(f"\nadapter OFF:               '{letter}'-names {off_hr:5.1%}{tag}")

    # --- save just the adapter (tiny!) — while it is still intact ---------
    path = f"adapter_{model_name}_{method}_{letter}.pt"
    save_adapter(model, path, method, lcfg, arch=model_name)
    kb = os.path.getsize(path) / 1024
    print(f"\nsaved {path}  ({kb:.1f} KB adapter vs "
          f"{os.path.getsize(base_checkpoint(model_name))/1024:.0f} KB base)")

    # --- promise 2: merging reproduces the adapted model exactly ---------
    xb, _ = get_batch(data)
    before = model(xb)[0]
    merge_adapters(model)
    after = model(xb)[0]
    max_diff = (before - after).abs().max().item()
    print(f"merge check: max|logit diff| = {max_diff:.2e}  "
          f"({'OK, identical' if max_diff < 1e-4 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
