"""
Step 4 entrypoint: finetune Whisper on IndicVoices Punjabi.

Wires together:
  config → model init → LoRA → freeze → dataset → collator → trainer → train

Run:
    python scripts/train.py
    SMOKE_TEST=true python scripts/train.py   # 10-step end-to-end check
"""

import sys
import os
from pathlib import Path

import torch
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model   import load_model
from src.model.lora    import apply_lora, print_trainable_parameters
from src.model.freeze  import apply_freeze
from src.data.dataset  import StreamingASRDataset, DataConfig
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.trainer   import WhisperTrainer
from src.training.callbacks import SavePeftAdapterCallback, EarlyStoppingOnWER
from src.evaluation.metrics import make_compute_metrics


def main():
    cfg = load_config()
    t   = cfg.training

    print(f"\n{'='*60}")
    print(f"  Whisper finetuning")
    print(f"  Model   : {cfg.model.name}")
    print(f"  Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"  Steps   : {t.max_steps}  |  LR: {t.learning_rate}")
    print(f"{'='*60}\n")

    # ── 1. Processor ──────────────────────────────────────────────────────────
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
        feature_size=cfg.data.get("feature_size", 128),
    )

    # ── 2. Model → LoRA → Freeze ──────────────────────────────────────────────
    print("Loading model...")
    model = load_model(cfg)

    if cfg.lora.enabled:
        print("Applying LoRA...")
        model = apply_lora(model, cfg)

    print("Applying freeze...")
    apply_freeze(model, cfg)

    print("\nParameter audit:")
    print_trainable_parameters(model)

    # gradient_checkpointing must be enabled AFTER peft wrapping
    if t.get("gradient_checkpointing", False):
        model.enable_input_require_grads()   # required for grad checkpointing + PEFT
        model.gradient_checkpointing_enable()

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("\nConnecting to dataset (streaming)...")
    data_config = DataConfig.from_omega(cfg)
    ds = StreamingASRDataset(data_config, processor.feature_extractor, processor.tokenizer)
    ds.load()

    train_dataset = ds.get_split("train", shuffle=True)
    eval_dataset  = ds.get_split("val",   shuffle=False)
    print("Dataset ready.\n")

    # ── 4. Collator ───────────────────────────────────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    compute_metrics = make_compute_metrics(processor.tokenizer)

    # ── 6. Training arguments ─────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=t.output_dir,

        # Steps-based schedule (required for IterableDataset)
        max_steps=t.max_steps,
        warmup_steps=t.warmup_steps,

        # Batch
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,

        # Optimiser
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,

        # Eval + save
        eval_strategy=t.eval_strategy,
        eval_steps=t.eval_steps,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,

        # Precision
        bf16=t.bf16,
        fp16=t.fp16,

        # DataLoader
        dataloader_num_workers=t.dataloader_num_workers,
        dataloader_pin_memory=t.dataloader_pin_memory,

        # Generation during eval
        predict_with_generate=t.predict_with_generate,
        generation_max_length=t.generation_max_length,

        # Logging
        logging_steps=t.logging_steps,
        report_to=t.report_to,

        # Required for IterableDataset — disables length-based sampler
        remove_unused_columns=False,
    )

    # ── 7. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        SavePeftAdapterCallback(),
        EarlyStoppingOnWER(patience=5, min_delta=0.001),
    ]

    # ── 8. Trainer ────────────────────────────────────────────────────────────
    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        processing_class=processor.feature_extractor,  # used for saving processor config
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    print("Starting training...\n")

    # Resume from checkpoint if one exists
    last_checkpoint = _find_last_checkpoint(t.output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}\n")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── 10. Save final adapter ────────────────────────────────────────────────
    final_dir = Path(t.output_dir) / "final_adapter"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nFinal adapter saved → {final_dir}")
    print("Training complete. Proceed to Step 5: evaluation.\n")


def _find_last_checkpoint(output_dir: str):
    """Return the latest checkpoint directory, or None if none exists."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoints = sorted(
        output_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


if __name__ == "__main__":
    main()