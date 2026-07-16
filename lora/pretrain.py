"""pretrain.py — make the base model we will later adapt (any architecture).

This is the "expensive, general" model: a tiny LM trained on ALL 921 Turkish
names. LoRA's whole premise is that you train this once and then never touch
its weights again — you bolt small adapters on instead.

Run:  python3 pretrain.py            # qwen3 (default) -> base_qwen3.pt
      python3 pretrain.py gemma4     # gemma        -> base_gemma4.pt
      python3 pretrain.py deepseek3  # deepseek      -> base_deepseek3.pt
      python3 pretrain.py qwen35     # qwen3.5       -> base_qwen35.pt

The architecture itself is NOT re-implemented here: base_model.py imports it
from the matching sibling folder (../qwen3, ../gemma4, ...). This file only
trains it and saves it, so the same script works for every model.
"""

# os/sys for the CLI arg and paths; torch for the training loop.
import os
import sys
import torch

# The registry that turns a short model name into its class, config, tokenizer
# and checkpoint path — the only model-specific knowledge in this folder.
from base_model import (model_class, config_class, tokenizer_class,
                        base_checkpoint)

# The shared names dataset lives one level up in ../data/.
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "temiz_isimler.txt")
# Training hyperparameters (same recipe every architecture uses).
BATCH_SIZE, BLOCK_SIZE, STEPS, LR = 64, 16, 3000, 3e-3
# CPU is plenty for these ~20-60k parameter models.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix the seed so every student gets the same base checkpoint.
torch.manual_seed(1337)


# Sample one training batch: random windows, targets shifted one step right.
def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    # First CLI arg picks the architecture; default to qwen3.
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    # Pull the right class / config / tokenizer for that architecture.
    Model = model_class(model_name)
    Config = config_class(model_name)
    Tokenizer = tokenizer_class(model_name)

    # Build the vocabulary from the names file (29 Turkish letters + newline).
    tokenizer = Tokenizer.from_file(DATA_FILE)
    # Encode the whole corpus into one flat tensor of token ids.
    data = torch.tensor(tokenizer.encode(open(DATA_FILE, encoding="utf-8").read()),
                        dtype=torch.long)

    # Build the model config (real vocab size, everything else default), then
    # the model itself, on the training device.
    cfg = Config(vocab_size=tokenizer.vocab_size)
    model = Model(cfg).to(device)
    # Report which model and how many parameters we are training.
    print(f"model={model_name}  device={device}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # Plain AdamW over ALL parameters — this is full training of the base.
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(1, STEPS + 1):
        # Forward pass returns (logits, loss); we only need the loss here.
        x, y = get_batch(data)
        _, loss = model(x, y)
        # Standard backprop step.
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    # Save weights + vocabulary + config so train/generate can rebuild it
    # exactly. The filename encodes the architecture (base_<name>.pt).
    path = base_checkpoint(model_name)
    torch.save({"model": model.state_dict(), "chars": tokenizer.chars, "cfg": cfg},
               path)
    print(f"saved {path}")
