"""base_model.py — pick ANY of the repo's tiny models to adapt.

The adapters in this folder are architecture-agnostic: inject() wraps
`nn.Linear` layers by name, so the exact same LoRA / DoRA / VeRA / PiSSA code
specializes qwen3, qwen3.5, gemma, or deepseek without a single change. This
registry is the ONLY place that knows per-model details — where each
architecture lives, its class name, and which linear layers make good adapter
targets.

Why a registry at all? Because the model folders all share module names
(`model.py`, `config.py`, `tokenizer.py`), so you cannot import two of them in
one process. We therefore import exactly one architecture per run, chosen by a
short name — matching the repo's "one architecture per kernel" rule.
"""

# importlib lets us import the shared module names ('model', 'config', ...)
# from whichever architecture folder we put on the path.
import importlib
# os/sys to locate sibling folders and add the chosen one to the import path.
import os
import sys

# Absolute path to this file's folder (lora/) and to the repo root (its parent).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# name -> where the architecture lives, its model class, and default targets.
# The targets differ because architectures name their projections differently:
#   * qwen3 / qwen35 / gemma  use standard attention: q_proj,k_proj,v_proj,o_proj
#   * deepseek3 uses MLA, which has NO q/k/v_proj — it up-projects from a
#     compressed latent, so we target q_up, kv_up and o_proj instead.
# Run `python3 train.py <model> --list` to print every Linear in a model and
# pick your own targets.
MODELS = {
    "qwen3":     dict(folder="qwen3",     cls="TinyQwen",     targets=("q_proj", "k_proj", "v_proj", "o_proj")),
    "qwen35":    dict(folder="qwen3_5",   cls="TinyQwen35",   targets=("q_proj", "k_proj", "v_proj", "o_proj")),
    "gemma4":    dict(folder="gemma4",    cls="TinyGemma",    targets=("q_proj", "k_proj", "v_proj", "o_proj")),
    "deepseek3": dict(folder="deepseek3", cls="TinyDeepSeek", targets=("q_up", "kv_up", "o_proj")),
}


# Import `module_name` (e.g. "model") from architecture `name`'s folder.
def _import_from_arch(name, module_name):
    # Reject typos early with a helpful list of valid names.
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. choose from {list(MODELS)}")
    # Absolute path to that architecture's folder.
    folder = os.path.join(ROOT, MODELS[name]["folder"])
    # Put it first on the import path (once) so `import model` resolves there.
    if folder not in sys.path:
        sys.path.insert(0, folder)
    # Import and return the requested module from that folder.
    return importlib.import_module(module_name)


# The nn.Module subclass for architecture `name` (e.g. TinyQwen, TinyGemma).
def model_class(name):
    # Importing 'model' also registers 'config' (model.py imports it), which is
    # what lets torch.load unpickle a saved cfg later.
    return getattr(_import_from_arch(name, "model"), MODELS[name]["cls"])


# The ModelConfig dataclass for architecture `name`.
def config_class(name):
    return getattr(_import_from_arch(name, "config"), "ModelConfig")


# The CharTokenizer for architecture `name` (identical across all folders).
def tokenizer_class(name):
    return getattr(_import_from_arch(name, "tokenizer"), "CharTokenizer")


# The recommended Linear layer names to adapt for architecture `name`.
def default_targets(name):
    return MODELS[name]["targets"]


# Where pretrain.py saves, and train.py / generate.py load, this base model.
def base_checkpoint(name):
    return f"base_{name}.pt"
