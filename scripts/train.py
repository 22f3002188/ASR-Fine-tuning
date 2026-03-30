"""
Whisper fine-tuning with optional LoRA.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import Seq2SeqTrainingArguments, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model import load_model
from src.model.lora import apply_lora, print_trainable_parameters
from src.model.freeze import apply_freeze
from src.data.dataset import DataConfig, build_train_dataset, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.trainer import WhisperTrainer
from src.evaluation.metrics import make_compute_metrics


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


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────

def main() -> None:
    setup_cache()

    cfg = load_config()
    t = cfg.training
    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    print(f"\n{'=' * 50}")
    print(" Whisper Finetuning")
    print(f" Model   : {cfg.model.name}")
    print(f" Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f" Smoke   : {smoke}")
    print(f"{'=' * 50}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set.")

    # ── Processor ──────────────────────────────
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
        token=hf_token,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )

    # ── Model ──────────────────────────────────
    print("Loading model...")
    model = load_model(cfg)
    model.config.use_cache = False

    if hasattr(cfg, "lora") and cfg.lora.get("enabled", False):
        print("Applying LoRA adapters...")
        model = apply_lora(model, cfg)
        print("LoRA applied.\n")

    if hasattr(cfg, "freeze"):
        print("Applying parameter freezing...")
        apply_freeze(model, cfg)
        print()

    print("Parameter summary:")
    print_trainable_parameters(model)
    print()

    model = model.to(device)

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
    eval_ds = build_eval_dataset(data_cfg, processor, token=hf_token)

    if eval_ds is None:
        print("⚠️ No validation dataset found → disabling evaluation\n")
        eval_strategy = "no"
        eval_ds = None
    else:
        eval_strategy = t.eval_strategy

    print("Dataset ready.\n")

    # ── Collator ───────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        model_dtype=next(model.parameters()).dtype,
    )

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

        eval_strategy="no" if smoke else eval_strategy,
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
        eval_dataset=eval_ds if eval_strategy != "no" else None,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...\n")
    trainer.train()

    # ── Save ───────────────────────────────────
    save_path = Path(t.output_dir) / "final_model"
    save_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(save_path))
    processor.save_pretrained(str(save_path))

    print(f"\nModel saved → {save_path}")
    os._exit(0)


if __name__ == "__main__":
    main()