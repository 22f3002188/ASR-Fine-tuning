"""
Whisper full fine-tuning (clean version).

Run:
    SMOKE_TEST=true python scripts/train.py
    python scripts/train.py
"""

import os
import sys
from pathlib import Path

import torch
from transformers import Seq2SeqTrainingArguments, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model import load_model
from src.data.dataset import DataConfig, build_train_dataset, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.trainer import WhisperTrainer
from src.evaluation.metrics import make_compute_metrics


# ────────────────────────────────────────────────
# Setup HF cache (once, clean)
# ────────────────────────────────────────────────

def setup_cache():
    base = os.environ.get("HF_HOME", "/home/harsh/hf_cache")

    os.environ["HF_HOME"] = base
    os.environ["HF_HUB_CACHE"] = f"{base}/hub"
    os.environ["HF_DATASETS_CACHE"] = f"{base}/datasets"
    os.environ["TRANSFORMERS_CACHE"] = f"{base}/transformers"

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main():
    setup_cache()

    cfg = load_config()
    t = cfg.training
    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    print(f"\n{'='*50}")
    print(" Whisper Finetuning")
    print(f" Model   : {cfg.model.name}")
    print(f" Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f" Smoke   : {smoke}")
    print(f"{'='*50}\n")

    # ── Device ─────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── HF Token ───────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set (IndicVoices is gated).")

    # ── Processor ──────────────────────────────
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
        feature_size=cfg.data.get("feature_size", 128),
        token=hf_token,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )

    # ── Model ──────────────────────────────────
    print("Loading model...")
    model = load_model(cfg).to(device)
    model.config.use_cache = False

    # Precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
    elif use_fp16:
        model = model.to(dtype=torch.float16)

    print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}\n")

    # ── Dataset ────────────────────────────────
    print("Loading dataset...")
    data_cfg = DataConfig.from_omega(cfg)

    train_ds = build_train_dataset(data_cfg, processor, token=hf_token)
    eval_ds  = build_eval_dataset(data_cfg, processor, token=hf_token)

    print("Dataset ready.\n")

    # ── Collator ───────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        model_dtype=next(model.parameters()).dtype,
    )

    # ── Metrics ────────────────────────────────
    compute_metrics = make_compute_metrics(processor.tokenizer)

    # ── Training args ──────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=t.output_dir,
        max_steps=t.smoke_test_steps if smoke else t.max_steps,

        per_device_train_batch_size=1 if smoke else t.per_device_train_batch_size,
        per_device_eval_batch_size=1 if smoke else t.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if smoke else t.gradient_accumulation_steps,

        learning_rate=t.learning_rate,
        warmup_steps=t.warmup_steps,

        eval_strategy="no" if smoke else t.eval_strategy,
        save_strategy="no" if smoke else t.save_strategy,

        bf16=use_bf16,
        fp16=use_fp16,

        logging_steps=1 if smoke else t.logging_steps,
        predict_with_generate=not smoke,
        generation_max_length=t.generation_max_length,

        remove_unused_columns=False,
        report_to="none",
    )

    # ── Trainer ────────────────────────────────
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    # ── Train ──────────────────────────────────
    print("Starting training...\n")
    trainer.train()

    # ── Save ───────────────────────────────────
    save_path = Path(t.output_dir) / "final_model"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    print(f"\nModel saved → {save_path}")


if __name__ == "__main__":
    main()