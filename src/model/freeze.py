"""
Freezing utilities for Whisper.
"""

from __future__ import annotations


def apply_freeze(model, cfg) -> None:
    """
    Freeze parts of the model based on cfg.freeze.

    Supported:
    - freeze encoder
    - freeze embed positions
    - freeze first N decoder layers
    """
    if not hasattr(cfg, "freeze"):
        print("No freeze config found.")
        return

    freeze_cfg = cfg.freeze

    # Freeze encoder
    if freeze_cfg.get("encoder", False):
        for name, param in model.named_parameters():
            if name.startswith("model.encoder"):
                param.requires_grad = False

    # Freeze positional embeddings
    if freeze_cfg.get("embed_positions", False):
        for name, param in model.named_parameters():
            if "embed_positions" in name:
                param.requires_grad = False

    # Freeze first N decoder layers
    n_decoder_layers = int(freeze_cfg.get("decoder_layers", 0))
    if n_decoder_layers > 0:
        for name, param in model.named_parameters():
            for i in range(n_decoder_layers):
                if f"model.decoder.layers.{i}." in name:
                    param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"After freezing: {trainable:,}/{total:,} parameters trainable")