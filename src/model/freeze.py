"""
Layer freezing for Whisper + PEFT finetuning.

Freeze order:
    1. load_model()   — loads weights
    2. apply_lora()   — wraps with adapters, freezes non-adapter params
    3. apply_freeze() — enforces encoder freeze on top via named_parameters()

We iterate named_parameters() on the PeftModel directly (not the unwrapped
base) and match by name string. This avoids triggering PEFT's forward hooks
which can stall when using .parameters() on submodules of a wrapped model.
"""


def apply_freeze(model, cfg) -> None:
    """
    Freeze parameters by name-matching on the full PeftModel.
    Modifies requires_grad in-place — fast, hook-safe, version-agnostic.
    """
    freeze_cfg = cfg.get("freeze", {})

    frozen_count = 0

    for name, param in model.named_parameters():
        if _should_freeze(name, freeze_cfg):
            param.requires_grad = False
            frozen_count += 1

    print(f"Frozen: {frozen_count} parameter tensors")
    _print_freeze_summary(model)


def _should_freeze(name: str, freeze_cfg) -> bool:
    """Return True if this parameter should be frozen based on config."""

    # Always freeze encoder if configured
    if freeze_cfg.get("encoder", True):
        if "model.encoder" in name or "base_model.model.model.encoder" in name:
            return True

    # Freeze embed_positions in both encoder and decoder
    if freeze_cfg.get("freeze_embed_positions", True):
        if "embed_positions" in name:
            return True

    # Freeze first N decoder layers
    n_frozen = freeze_cfg.get("freeze_decoder_layers", 0)
    if n_frozen > 0:
        for i in range(n_frozen):
            if f"decoder.layers.{i}." in name:
                return True

    return False


def _print_freeze_summary(model) -> None:
    """Print a one-line summary of trainable vs frozen parameter counts."""
    trainable, frozen = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        if p.requires_grad:
            trainable += n
        else:
            frozen += n
    total = trainable + frozen
    print(f"  Trainable : {trainable:>12,}  ({100*trainable/total:.3f}%)")
    print(f"  Frozen    : {frozen:>12,}  ({100*frozen/total:.3f}%)")
    print(f"  Total     : {total:>12,}")