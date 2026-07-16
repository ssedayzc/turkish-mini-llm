"""generate.py — load a base + a saved adapter and generate names.

This is the payoff of the whole folder: you keep ONE base per architecture and
swap tiny adapter files to get different behavior, without retraining. The
adapter file records which architecture it belongs to, so you only pass the
adapter path — generate figures out the rest.

Run (after pretrain.py and a train.py run):
    python3 generate.py adapter_qwen3_lora_z.pt         # 20 names, adapter on
    python3 generate.py adapter_qwen3_lora_z.pt 40      # 40 names
    python3 generate.py adapter_qwen3_lora_z.pt 40 base # 40 names, adapter OFF

The optional third argument "base" disables the adapter so you can see the
untouched base model for comparison from the very same process.
"""

# os/sys for args; torch for inference.
import os
import sys
import torch

# Registry loaders + the adapter loader and on/off switch.
from base_model import model_class, tokenizer_class, base_checkpoint
from inject import load_adapter, set_adapters


# Generate `n` names, cutting each at the end-of-name newline.
@torch.no_grad()
def generate_names(model, tokenizer, n, temperature=0.8):
    start = torch.full((n, 1), tokenizer.newline_id, dtype=torch.long)
    out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                         temperature=temperature, eos_id=tokenizer.eos_id)
    names = [tokenizer.decode(r[1:]).split("\n")[0] for r in out.tolist()]
    return [nm for nm in names if nm]


def main():
    # First arg = adapter file; default to the qwen3 LoRA 'z' adapter.
    adapter_path = sys.argv[1] if len(sys.argv) > 1 else "adapter_qwen3_lora_z.pt"
    # Second arg = how many names to print.
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    # Third arg = "base" to disable the adapter (show the untouched base).
    use_base = len(sys.argv) > 3 and sys.argv[3] == "base"

    # Peek at the adapter file to learn which architecture it belongs to.
    arch = torch.load(adapter_path, map_location="cpu", weights_only=False)["arch"]

    # Rebuild that architecture's frozen base from its checkpoint.
    Model = model_class(arch)
    Tokenizer = tokenizer_class(arch)
    ckpt = torch.load(base_checkpoint(arch), map_location="cpu", weights_only=False)
    tokenizer = Tokenizer(ckpt["chars"])
    model = Model(ckpt["cfg"]); model.load_state_dict(ckpt["model"]); model.eval()

    # Inject the adapter shape onto the base and load its trained numbers.
    # (For PiSSA this also rebuilds the residual base from the saved A0/B0.)
    load_adapter(model, adapter_path)

    # Optionally switch the adapter off to compare against the base model.
    if use_base:
        set_adapters(model, False)

    # Announce what we are running, then print the names.
    tag = f"{arch} BASE (adapter off)" if use_base else f"{arch} + {adapter_path}"
    print(f"# {tag}")
    for name in generate_names(model, tokenizer, n):
        print(name)


if __name__ == "__main__":
    main()
