"""inject.py — swap a model's Linear layers for adapter-wrapped versions,
freeze everything else, and save/load just the tiny adapter.

This is the plumbing that turns any of the four adapter classes into a
fine-tuning setup. Nothing here is method-specific math — it only:

  1. walks the model, finds Linears whose name is in cfg.targets,
     and replaces each with the chosen adapter wrapper;
  2. freezes every base parameter so ONLY adapter numbers train;
  3. saves / loads an adapter file that holds those few numbers.

The point students should feel: after inject(), `count_trainable` drops
from "the whole model" to a few hundred numbers, yet training still works.
"""

# torch for save/load; nn for the module tree and isinstance checks.
import torch
from torch import nn

# The four adapter implementations we can inject.
from lora import LoRAConfig, LoRALinear
from dora import DoRALinear
from vera import VeRALinear
from pissa import pissa_init, pissa_apply_residual

# Every adapter parameter/buffer name contains one of these substrings. We use
# this list both to decide what to train and what to save. Note vera_A/vera_B
# are deliberately absent: they are frozen random buffers we regenerate from a
# seed, never train and never save.
ADAPTER_KEYS = ("lora_A", "lora_B", "dora_mag", "vera_d", "vera_b", "pissa_")

# Linear layers we never adapt automatically, even under targets="all". lm_head
# shares its weight with the token embedding (weight tying), so wrapping it
# would freeze the embeddings and make "merge" also rewrite them — a footgun.
EXCLUDE = ("lm_head",)


# Does this (name, module) pair qualify as an adapter target?
def _is_target(name: str, module: nn.Module, targets) -> bool:
    # Only ever adapt Linear layers.
    if not isinstance(module, nn.Linear):
        return False
    # The leaf attribute name, e.g. "q_proj" from "layers.0.attn.q_proj".
    leaf = name.split(".")[-1]
    # Never touch the excluded (tied) layers.
    if leaf in EXCLUDE:
        return False
    # targets="all" adapts every remaining Linear; otherwise match by name.
    return True if targets == "all" else (leaf in targets)


# List every adaptable Linear in a model (name, shape) — call this to choose
# targets for a new architecture you have not adapted before.
def print_linear_names(model: nn.Module) -> None:
    print("  adaptable nn.Linear layers (leaf name in brackets):")
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            leaf = name.split(".")[-1]
            note = "   <- excluded (tied to embeddings)" if leaf in EXCLUDE else ""
            print(f"    {name:34s} [{m.out_features:4d} x {m.in_features:4d}]  ({leaf}){note}")


# Replace the attribute named by a dotted path (e.g. "layers.0.attn.q_proj")
# on `model` with `new_module`.
def _replace(model: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    """setattr the deepest attribute named by a dotted path (a.b.c)."""
    # Split "a.b.c" into parents ["a","b"] and the final attribute "c".
    *parents, last = qualified_name.split(".")
    # Walk down from the model through each parent attribute.
    obj = model
    for p in parents:
        obj = getattr(obj, p)
    # Overwrite the leaf attribute with the wrapped module.
    setattr(obj, last, new_module)


# Wrap every targeted Linear in `model` with the chosen adapter type.
def inject(model: nn.Module, cfg: LoRAConfig, method: str = "lora", seed: int = 0) -> list[str]:
    """Wrap every targeted Linear in `model` with the chosen adapter.

    method: "lora" | "rslora" | "dora" | "vera" | "pissa".
    ("rslora" is just LoRA with cfg.rank_stabilized=True; we set it for you.)
    Returns the list of layer names that were adapted.
    """
    # rsLoRA is plain LoRA with the sqrt(r) scale, so flip that flag for the user.
    if method == "rslora":
        cfg.rank_stabilized = True

    # Collect the names we adapt so callers can print/inspect them.
    replaced = []
    # named_modules() yields (dotted_name, module) for every sub-module. We
    # wrap it in list() because we mutate the tree while iterating.
    for name, module in list(model.named_modules()):
        # Adapt only Linear layers selected by cfg.targets (name list or "all").
        if _is_target(name, module, cfg.targets):
            # VeRA needs the seed (to pick its frozen random matrices).
            if method == "vera":
                wrapped = VeRALinear(module, cfg, seed)
            # DoRA is the magnitude/direction decomposition.
            elif method == "dora":
                wrapped = DoRALinear(module, cfg)
            # lora, rslora and pissa all use the LoRALinear body.
            else:
                wrapped = LoRALinear(module, cfg)
                # PiSSA additionally re-initializes A,B from W's SVD and turns
                # the base into the residual.
                if method == "pissa":
                    pissa_init(wrapped)
            # Splice the wrapper into the model in place of the raw Linear.
            _replace(model, name, wrapped)
            # Record the layer name.
            replaced.append(name)

    # Guard against a silent no-op: if the targets matched nothing (e.g. asking
    # for q_proj on deepseek's MLA), tell the user which names DO exist.
    if not replaced:
        avail = sorted({n.split(".")[-1] for n, m in model.named_modules()
                        if isinstance(m, nn.Linear) and n.split(".")[-1] not in EXCLUDE})
        raise ValueError(f"targets {cfg.targets} matched no Linear layer. "
                         f"available leaf names: {avail}")

    # Freeze everything except the adapter parameters we just added.
    mark_only_adapters_trainable(model)
    # Return the list of adapted layer names.
    return replaced


# Freeze the whole model, then unfreeze only the adapter parameters.
def mark_only_adapters_trainable(model: nn.Module) -> None:
    """Freeze the whole model, then unfreeze only the adapter parameters."""
    # Start by turning OFF gradients for every parameter.
    for p in model.parameters():
        p.requires_grad = False
    # Then turn gradients back ON only for parameters whose name marks them as
    # adapter numbers (lora_A/B, dora_mag, vera_d/b — pissa reuses lora_A/B).
    for name, p in model.named_parameters():
        if any(k in name for k in ADAPTER_KEYS):
            p.requires_grad = True


# Turn every adapter path on or off at once.
def set_adapters(model: nn.Module, enabled: bool) -> None:
    """Turn every adapter path on/off — off == the original frozen model."""
    # Every adapter module exposes an `enabled` flag; flip them all.
    for m in model.modules():
        if hasattr(m, "enabled"):
            m.enabled = enabled


# Fold every adapter into its base weight, in place.
def merge_adapters(model: nn.Module) -> None:
    """Fold every adapter into its base weight (in place). Inference is then
    identical to the base model's cost — the whole selling point of LoRA."""
    # Each adapter module knows how to merge itself (and disables afterward).
    for m in model.modules():
        if hasattr(m, "merge"):
            m.merge()


# ---------------------------------------------------------------------------
# Parameter accounting — the number that makes the whole idea land.
# ---------------------------------------------------------------------------
# Return (trainable count, total count) over all parameters.
def count_parameters(model: nn.Module) -> tuple[int, int]:
    # Sum the element counts of every parameter tensor.
    total = sum(p.numel() for p in model.parameters())
    # Sum only those still requiring gradients (the adapters, after inject()).
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# Pretty-print the trainable-vs-total ratio.
def print_parameter_report(model: nn.Module) -> None:
    trainable, total = count_parameters(model)
    print(f"  trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}% of the model)")


# ---------------------------------------------------------------------------
# Saving / loading just the adapter. An adapter file is tiny — that is the
# practical payoff: ship a 1 KB file, not a whole new model.
# ---------------------------------------------------------------------------
# Extract only the tensors that define the adapter (trained params + the pissa
# reconstruction buffers), skipping the frozen base and VeRA's random A/B.
def adapter_state_dict(model: nn.Module) -> dict:
    """Only the numbers that define the adapter (params + pissa buffers)."""
    # state_dict() has every param and buffer keyed by dotted name; keep the
    # ones whose name contains an adapter marker.
    return {n: t for n, t in model.state_dict().items()
            if any(k in n for k in ADAPTER_KEYS)}


# Save {arch, method, cfg, adapter tensors} to a file.
def save_adapter(model: nn.Module, path: str, method: str, cfg: LoRAConfig,
                 arch: str = None) -> None:
    # We store arch (which base it belongs to), the method name and cfg, so
    # load can find the right base and rebuild the exact adapter shapes.
    torch.save({"arch": arch, "method": method, "cfg": cfg,
                "adapter": adapter_state_dict(model)}, path)


# Rebuild the adapter on a fresh base model and load its trained numbers.
def load_adapter(model: nn.Module, path: str, seed: int = 0) -> list[str]:
    """Inject the right adapter shape onto a fresh base, then load its numbers.

    For PiSSA we must first turn the fresh base back into the residual using
    the saved A0/B0 (PiSSA edited the base at train time), THEN load the
    trained A/B on top.
    """
    # Read back {method, cfg, adapter}. weights_only=False because cfg is a
    # pickled dataclass, not a bare tensor.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    method, cfg, state = ckpt["method"], ckpt["cfg"], ckpt["adapter"]

    # Recreate the adapter modules with the right shapes. PiSSA is injected as
    # plain "lora" here (same A,B shapes); we fix its base next.
    names = inject(model, cfg, method="lora" if method == "pissa" else method, seed=seed)

    # PiSSA edited the base at train time, so replay that edit on the fresh base
    # using the saved A0/B0 before loading the trained A/B on top.
    if method == "pissa":
        for name in names:
            # Navigate to the wrapped layer object by its dotted name.
            *parents, last = name.split(".")
            obj = model
            for p in parents:
                obj = getattr(obj, p)
            layer = getattr(obj, last)
            # Subtract the principal part so the base becomes the residual again.
            pissa_apply_residual(layer, state[f"{name}.pissa_A0"], state[f"{name}.pissa_B0"])

    # Load the adapter tensors. strict=False because the frozen base keys are
    # intentionally absent from `state` (we only saved adapter numbers).
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Any key in the file that the model did NOT expect is a real error.
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected}")
    # Return the adapted layer names.
    return names
