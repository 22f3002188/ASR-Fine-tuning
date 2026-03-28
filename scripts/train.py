"""
Vanilla Whisper finetuning — no LoRA, no freezing.
All parameters are updated end-to-end.

Run:
    SMOKE_TEST=true python scripts/train.py   # 10-step check
    python scripts/train.py                   # full training
"""

import os
import sys
from pathlib import Path

# Single GPU only — DataParallel is not needed for a single large model
# and causes issues with some transformers/accelerate versions.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model   import load_model
from src.data.dataset  import DataConfig, build_train_dataset, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.trainer   import WhisperTrainer
from src.training.callbacks import SaveCheckpointCallback, EarlyStoppingOnWER
from src.evaluation.metrics import make_compute_metrics


def main():
    cfg = load_config()
    t   = cfg.training
    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    print(f"\n{'='*60}")
    print(f"  Whisper finetuning  (vanilla — full model)")
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

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("Loading model...")
    model = load_model(cfg)

    _has_gpu = torch.cuda.is_available()
    _use_bf16 = t.bf16 and _has_gpu
    _use_fp16 = t.fp16 and _has_gpu and not _use_bf16

    if not _has_gpu:
        print("WARNING: No GPU — training in fp32 on CPU (smoke test only).\n")

    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Precision  : {'bf16' if _use_bf16 else 'fp16' if _use_fp16 else 'fp32'}\n")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("Connecting to dataset (streaming)...")
    data_config   = DataConfig.from_omega(cfg)
    hf_token      = os.environ.get("HF_TOKEN")
    train_dataset = build_train_dataset(data_config, processor, token=hf_token)
    eval_dataset  = build_eval_dataset(data_config,  processor, token=hf_token)
    print("Dataset ready.\n")

    # ── 4. Collator ───────────────────────────────────────────────────────────
    _model_dtype = next(model.parameters()).dtype
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        model_dtype=_model_dtype,
    )

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    compute_metrics = make_compute_metrics(processor.tokenizer)

    # ── 6. Training arguments ─────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=t.output_dir,

        max_steps=t.max_steps,
        warmup_steps=t.warmup_steps,

        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,

        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,

        # During smoke test skip mid-training eval — only run final eval
        # against the capped eval_dataset set above.
        eval_strategy='no' if smoke else t.eval_strategy,
        eval_steps=None if smoke else t.eval_steps,
        save_strategy='no' if smoke else t.save_strategy,
        save_steps=None if smoke else t.save_steps,
        load_best_model_at_end=False if smoke else t.load_best_model_at_end,
        metric_for_best_model=None if smoke else t.metric_for_best_model,
        greater_is_better=None if smoke else t.greater_is_better,

        bf16=_use_bf16,
        fp16=_use_fp16,
        gradient_checkpointing=t.get("gradient_checkpointing", True) and _has_gpu,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        predict_with_generate=t.predict_with_generate,
        generation_max_length=t.generation_max_length,

        logging_steps=t.logging_steps,
        report_to=t.report_to,

        remove_unused_columns=False,
    )

    # ── 7. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        SaveCheckpointCallback(),
        EarlyStoppingOnWER(patience=5, min_delta=0.001),
    ]

    # ── 8. Trainer ────────────────────────────────────────────────────────────
    # During smoke test, cap eval at a small fixed number of batches so it
    # doesn't stream the entire eval split (which has no __len__).
    import itertools
    if smoke and eval_dataset is not None:
        smoke_eval_samples = t.get("smoke_test_steps", 10) * t.per_device_eval_batch_size
        eval_dataset = list(itertools.islice(iter(eval_dataset), smoke_eval_samples))
        print(f"[SMOKE TEST] Eval capped at {len(eval_dataset)} samples.\n")

    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        processing_class=processor.feature_extractor,
    )

    # ── 9. Train ──────────────────────────────────────────────────────────────
    print("Starting training...\n")
    last_checkpoint = _find_last_checkpoint(t.output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}\n")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── 10. Save ──────────────────────────────────────────────────────────────
    final_dir = Path(t.output_dir) / "final_model"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nModel saved → {final_dir}")
    print("Training complete. Proceed to Step 5: evaluation.\n")


def _find_last_checkpoint(output_dir: str):
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