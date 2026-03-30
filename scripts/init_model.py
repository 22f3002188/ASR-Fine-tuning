"""
Step 3 entrypoint: initialise and verify the model setup.

Loads Whisper large-v3, optionally applies LoRA and freezing, then prints:
  - Trainable vs total parameter counts
  - Which parameter groups are frozen / trainable
  - A small sample of trainable parameter names

Run:
    python scripts/init_model.py
    SMOKE_TEST=true python scripts/init_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.freeze import apply_freeze
from src.model.lora import apply_lora, print_trainable_parameters
from src.model.model import load_model


def main() -> None:
    cfg = load_config()

    lora_enabled = hasattr(cfg, "lora") and cfg.lora.get("enabled", False)
    lora_rank = cfg.lora.get("r", "N/A") if hasattr(cfg, "lora") else "N/A"
    lora_alpha = cfg.lora.get("lora_alpha", "N/A") if hasattr(cfg, "lora") else "N/A"
    lora_apply_to = (
        cfg.lora.get("apply_to", "decoder_only") if hasattr(cfg, "lora") else "N/A"
    )

    print(f"\n{'=' * 60}")
    print("  Model initialisation")
    print(f"  Checkpoint : {cfg.model.name}")
    print(f"  Language   : {cfg.model.language}")
    print(f"  LoRA       : {lora_enabled}")
    print(f"  LoRA rank  : {lora_rank}")
    print(f"  Alpha      : {lora_alpha}")
    print(f"  Apply to   : {lora_apply_to}")
    print(f"{'=' * 60}\n")

    # ── Step 1: load base model ────────────────────────────────────────────────
    print("Loading base model...")
    model = load_model(cfg)
    print(f"Loaded: {cfg.model.name}\n")

    # ── Step 2: apply LoRA if enabled ──────────────────────────────────────────
    if lora_enabled:
        print("Applying LoRA adapters...")
        model = apply_lora(model, cfg)
        print("LoRA applied.\n")
    else:
        print("LoRA disabled in config. Skipping LoRA.\n")

    # ── Step 3: apply freezing if config exists ────────────────────────────────
    if hasattr(cfg, "freeze"):
        print("Applying layer freezing...")
        apply_freeze(model, cfg)
        print()
    else:
        print("No freeze config found. Skipping freezing.\n")

    # ── Step 4: parameter audit ────────────────────────────────────────────────
    print("── Parameter audit ─────────────────────────────────────────")
    print_trainable_parameters(model)
    print()

    # Show sample trainable names
    print("── Sample trainable parameters ─────────────────────────────")
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]

    if not trainable_names:
        print("  No trainable parameters found.")
    else:
        for name in trainable_names[:10]:
            print(f"  {name}")
        if len(trainable_names) > 10:
            print(f"  ... ({len(trainable_names)} total trainable tensors)")

    print("\nModel initialisation complete.")
    print("Run SMOKE_TEST=true python scripts/train.py on GPU for end-to-end check.\n")


if __name__ == "__main__":
    main()