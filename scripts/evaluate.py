"""
Evaluate trained Whisper + LoRA model on streaming IndicVoices.

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
os.environ["HF_HOME"] = "/home/harsh/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/home/harsh/hf_cache/datasets"
os.environ["HF_HUB_CACHE"] = "/home/harsh/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/home/harsh/hf_cache/transformers"
import torch
from jiwer import cer, wer
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data.dataset import DataConfig, build_eval_dataset, build_train_dataset


# ────────────────────────────────────────────────
# Setup HF cache
# ────────────────────────────────────────────────

def setup_cache() -> None:
    base = os.environ.get("HF_HOME", "/home/harsh/hf_cache")

    os.environ["HF_HOME"] = base
    os.environ["HF_HUB_CACHE"] = f"{base}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{base}/transformers"

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)


def main():
    setup_cache()
    cfg = load_config()

    print(f"\n{'='*50}")
    print(" Whisper Evaluation")
    print(f" Model   : {cfg.model.name}")
    print(f" Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"{'='*50}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set.")

    # ── Load processor ─────────────────────────
    print("Loading processor...")
    model_path = Path(cfg.training.output_dir) / "final_model"

    processor = WhisperProcessor.from_pretrained(str(model_path))
    processor.tokenizer.set_prefix_tokens(
        language="pa",
        task="transcribe",
    )

    # ── Load base model + LoRA ─────────────────
    print("Loading base model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name,
        token=hf_token,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )

    print(f"Loading LoRA adapters from {model_path} ...")
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model = model.to(device)
    model.eval()

    # ── Dataset ────────────────────────────────
    print("Loading validation dataset...")
    data_cfg = DataConfig.from_omega(cfg)

    eval_ds = build_eval_dataset(
        config=data_cfg,
        processor=processor,
        token=hf_token,
    )

    if eval_ds is None:
        print("Validation split could not be loaded. Falling back to train split for quick evaluation.\n")
        eval_ds = build_train_dataset(
            config=data_cfg,
            processor=processor,
            token=hf_token,
        )

    if eval_ds is None:
        raise RuntimeError("Could not build any evaluation dataset.")

    print("Dataset ready.\n")

    preds = []
    refs = []

    print("Running evaluation...\n")

    for i, sample in enumerate(eval_ds):
        if i >= 100:   # quick evaluation; increase later
            break

        input_features = torch.tensor(
            sample["input_features"],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        attention_mask = torch.ones(
            (input_features.shape[0], input_features.shape[-1]),
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language="pa",
                task="transcribe",
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                max_new_tokens=64,
                do_sample=False,
            )

        pred_text = processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        ref_text = processor.tokenizer.decode(
            sample["labels"],
            skip_special_tokens=True,
        ).strip()

        preds.append(pred_text)
        refs.append(ref_text)

        if i < 3:
            print(f"[Sample {i}]")
            print(f"  REF : {ref_text}")
            print(f"  PRED: {pred_text}\n")

    if not preds:
        raise RuntimeError("No predictions were generated; dataset may be empty after filtering.")

    final_wer = wer(refs, preds)
    final_cer = cer(refs, preds)

    print("\nFinal Results:")
    print(f"  wer: {final_wer:.4f}")
    print(f"  cer: {final_cer:.4f}")


if __name__ == "__main__":
    main()