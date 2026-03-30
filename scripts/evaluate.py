"""
Evaluate trained Whisper model on validation set.

Run:
    python scripts/evaluate.py
"""

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model import load_model
from src.data.dataset import DataConfig, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.evaluation.metrics import make_compute_metrics


def main():
    cfg = load_config()

    print(f"\n{'='*50}")
    print(" Whisper Evaluation")
    print(f" Model   : {cfg.model.name}")
    print(f" Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"{'='*50}\n")

    # ── Device ─────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── HF Token ───────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set.")

    # ── Load processor ─────────────────────────
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
        token=hf_token,
    )

    # ── Load model (from checkpoint) ───────────
    model_path = Path(cfg.training.output_dir) / "final_model"

    print(f"Loading model from {model_path} ...")
    model = load_model(cfg)
    model = model.from_pretrained(model_path).to(device)
    model.eval()

    # ── Dataset ────────────────────────────────
    print("Loading validation dataset...")
    data_cfg = DataConfig.from_omega(cfg)
    eval_ds = build_eval_dataset(data_cfg, processor, token=hf_token)

    print("Dataset ready.\n")

    # ── Metrics ────────────────────────────────
    compute_metrics = make_compute_metrics(processor.tokenizer)

    # ── Evaluation loop ────────────────────────
    preds = []
    refs = []

    print("Running evaluation...\n")

    for i, sample in enumerate(eval_ds):
        if i >= 200:   # limit for quick eval (increase later)
            break

        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_features)

        pred_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        ref_text = processor.tokenizer.decode(sample["labels"], skip_special_tokens=True)

        preds.append(pred_text)
        refs.append(ref_text)

        if i < 3:
            print(f"[Sample {i}]")
            print(f"  REF : {ref_text}")
            print(f"  PRED: {pred_text}\n")

    # ── Compute metrics ────────────────────────
    metrics = compute_metrics({"predictions": preds, "references": refs})

    print("\nFinal Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()