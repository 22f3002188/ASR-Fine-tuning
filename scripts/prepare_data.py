from __future__ import annotations

import os
import sys
import yaml
import gc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import DataConfig
from src.training.trainer import (
    build_processor,
    build_data_collator,
    build_train_dataloader,
    build_eval_dataloader,
    summarize_batch,
)
from src.model.model import hf_login_if_needed


# ============================================================
# LOAD YAML
# ============================================================

def load_yaml(path: str) -> dict:
    print(f"\n Loading config from: {path}", flush=True)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    print("\n Starting Data Preparation Pipeline...\n", flush=True)

    cfg = load_yaml("configs/data.yaml")

    data_cfg = DataConfig.from_dict(cfg["data"])
    model_cfg = cfg["model"]
    batching_cfg = cfg["batching"]

    # -------------------------
    # HF LOGIN
    # -------------------------
    hf_token = os.getenv("HF_TOKEN", "")
    hf_login_if_needed(hf_token)

    # -------------------------
    # PROCESSOR
    # -------------------------
    print("\n Building processor...", flush=True)
    processor = build_processor(model_cfg["processor_name"])

    # -------------------------
    # COLLATOR
    # -------------------------
    print(" Building data collator...", flush=True)
    collator = build_data_collator(
        processor=processor,
        max_label_length=batching_cfg["max_label_length"],
    )

    print("\n No domain filtering is applied.", flush=True)
    print(" Random shuffled batching is enabled.\n", flush=True)

    # -------------------------
    # TRAIN LOADER
    # -------------------------
    print(" Building train dataloader...", flush=True)
    train_loader = build_train_dataloader(
        data_config=data_cfg,
        processor=processor,
        collator=collator,
        batch_size=batching_cfg["train_batch_size"],
        hf_token=hf_token,
    )

    # -------------------------
    # EVAL LOADER
    # -------------------------
    print(" Building eval dataloader...", flush=True)
    eval_loader = build_eval_dataloader(
        data_config=data_cfg,
        processor=processor,
        collator=collator,
        batch_size=batching_cfg["eval_batch_size"],
        hf_token=hf_token,
    )

    # ============================================================
    # TRAIN BATCH DEBUG
    # ============================================================

    print("\n================ TRAIN DATA ================\n", flush=True)

    for i, batch in enumerate(train_loader):
        print(f"\n Train Batch {i + 1} received", flush=True)
        summarize_batch(batch)

        # Show only first 2 batches
        if i >= 1:
            break

    # ============================================================
    # EVAL BATCH DEBUG
    # ============================================================

    if eval_loader is not None:
        print("\n================ EVAL DATA ================\n", flush=True)

        for i, batch in enumerate(eval_loader):
            print(f"\n Eval Batch {i + 1} received", flush=True)
            summarize_batch(batch)

            # Show only 1 batch
            break
    else:
        print("\n No validation/test split available.\n", flush=True)

    # ============================================================
    # FINAL MESSAGE
    # ============================================================

    print("\n Punjabi dataset preparation, preprocessing, and batching completed.\n", flush=True)

    # ============================================================
    # SAFE SHUTDOWN FIX (VERY IMPORTANT)
    # ============================================================

    print("🧹 Cleaning up memory and safely exiting...\n", flush=True)

    try:
        del train_loader
        del eval_loader
        del processor
        del collator
    except Exception:
        pass

    gc.collect()

    # Force exit to avoid PyGILState_Release crash
    os._exit(0)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()