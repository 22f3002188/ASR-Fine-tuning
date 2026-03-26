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

    # Show which top-level modules have any trainable params
    print("── Trainable modules ───────────────────────────────────────")
    for name, module in model.named_modules():
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if trainable > 0 and "." not in name.split("lora_")[-1]:
            # Only print leaf-ish names to avoid flooding
            print(f"  {name:<60} {trainable:>12,} params")

    # ── Step 5: forward pass smoke test ───────────────────────────────────────
    print("\n── Forward pass smoke test ─────────────────────────────────")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    batch_size = 2
    dummy_features = torch.zeros(batch_size, 80, 3000, device=device)
    dummy_labels   = torch.zeros(batch_size, 10,       device=device, dtype=torch.long)

    with torch.no_grad():
        out = model(input_features=dummy_features, labels=dummy_labels)

    print(f"  Device     : {device}")
    print(f"  Loss       : {out.loss.item():.4f}  (non-zero = forward pass OK)")
    print(f"  Logits     : {tuple(out.logits.shape)}")
    print("\nModel initialisation complete. Safe to proceed to Step 4.\n")


if __name__ == "__main__":
    main()