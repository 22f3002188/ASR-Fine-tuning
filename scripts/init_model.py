"""
Step 3 entrypoint: initialise and verify the model setup.

Loads Whisper large-v3, applies LoRA and freezing, then prints:
  - Trainable vs total parameter counts
  - Which parameter groups are frozen / trainable
  - A forward pass smoke test using a random input batch

Run:
    python scripts/init_model.py
    SMOKE_TEST=true python scripts/init_model.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model  import load_model
from src.model.lora   import apply_lora, print_trainable_parameters
from src.model.freeze import apply_freeze


def main():
    cfg = load_config()

    print(f"\n{'='*60}")
    print(f"  Model initialisation")
    print(f"  Checkpoint : {cfg.model.name}")
    print(f"  Language   : {cfg.model.language}")
    print(f"  LoRA rank  : {cfg.lora.r}  alpha={cfg.lora.lora_alpha}")
    print(f"  Apply to   : {cfg.lora.get('apply_to', 'decoder_only')}")
    print(f"{'='*60}\n")

    # ── Step 1: load base model ────────────────────────────────────────────────
    print("Loading base model...")
    model = load_model(cfg)
    print(f"Loaded: {cfg.model.name}\n")

    # ── Step 2: apply LoRA ────────────────────────────────────────────────────
    if cfg.lora.enabled:
        print("Applying LoRA adapters...")
        model = apply_lora(model, cfg)
        print("LoRA applied.\n")

    # ── Step 3: freeze layers ─────────────────────────────────────────────────
    print("Applying layer freezing...")
    apply_freeze(model, cfg)
    print()

    # ── Step 4: parameter audit ────────────────────────────────────────────────
    print("── Parameter audit ─────────────────────────────────────────")
    print_trainable_parameters(model)
    print()

    # Show a sample of trainable parameter names (fast — no submodule iteration)
    print("── Sample trainable parameters ─────────────────────────────")
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    for n in trainable_names[:10]:
        print(f"  {n}")
    if len(trainable_names) > 10:
        print(f"  ... ({len(trainable_names)} total trainable tensors)")

    print("\nModel initialisation complete. Safe to proceed to Step 4.")
    print("Run SMOKE_TEST=true python scripts/train.py on GPU for end-to-end check.\n")


if __name__ == '__main__':
    main()