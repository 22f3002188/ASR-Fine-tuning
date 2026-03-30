"""
Vanilla Whisper finetuning — no LoRA, no freezing.
All parameters are updated end-to-end.

Run:
    SMOKE_TEST=true python scripts/train.py   # 10-step check
    python scripts/train.py                   # full training
"""

import os
import sys
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------
# Force Hugging Face caches to writable user paths BEFORE other imports
# ---------------------------------------------------------------------
HF_HOME = os.environ.get("HF_HOME", "/home/harsh/hf_cache")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", f"{HF_HOME}/hub")
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", f"{HF_HOME}/datasets")
HF_TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", f"{HF_HOME}/transformers")
HF_ASSETS_CACHE = os.environ.get("HUGGINGFACE_ASSETS_CACHE", f"{HF_HOME}/assets")

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_TRANSFORMERS_CACHE
os.environ["HUGGINGFACE_ASSETS_CACHE"] = HF_ASSETS_CACHE

for path in [
    HF_HOME,
    HF_HUB_CACHE,
    HF_DATASETS_CACHE,
    HF_TRANSFORMERS_CACHE,
    HF_ASSETS_CACHE,
]:
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------
# Runtime env
# ---------------------------------------------------------------------
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import Seq2SeqTrainingArguments, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model import load_model
from src.data.dataset import DataConfig, build_train_dataset, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.trainer import WhisperTrainer
from src.training.callbacks import SaveCheckpointCallback, EarlyStoppingOnWER
from src.evaluation.metrics import make_compute_metrics


def _gpu_report():
    if not torch.cuda.is_available():
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return [x.strip() for x in result.stdout.strip().splitlines() if x.strip()]
    except Exception:
        return None


def _ensure_gpu_has_headroom(min_free_gb: float = 20.0):
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    print(f"GPU free memory: {free_gb:.2f} GiB / {total_gb:.2f} GiB")
    report = _gpu_report()
    if report:
        print("GPU report:")
        for line in report:
            print(f"  {line}")
        print()

    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough free GPU memory. Need ~{min_free_gb:.1f} GiB free, found {free_gb:.2f} GiB."
        )


def _convert_model_precision_for_run(model, use_bf16: bool, use_fp16: bool):
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
    elif use_fp16:
        model = model.to(dtype=torch.float16)
    else:
        model = model.to(dtype=torch.float32)
    return model


def _check_model_has_finite_params(model):
    for name, param in model.named_parameters():
        if param is None:
            continue
        if not torch.isfinite(param.data).all():
            raise RuntimeError(f"Non-finite parameter detected before training: {name}")


def _normalize_report_to(report_to_value):
    if report_to_value is None:
        return "none"

    if isinstance(report_to_value, str):
        value = report_to_value.strip().lower()
        if value in {"", "none", "null", "false", "off"}:
            return "none"
        return [report_to_value]

    if isinstance(report_to_value, (list, tuple)):
        cleaned = [str(x).strip() for x in report_to_value if str(x).strip()]
        if len(cleaned) == 0:
            return "none"
        return cleaned

    return "none"


def main():
    cfg = load_config()
    t = cfg.training
    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    print(f"\n{'='*60}")
    print("  Whisper finetuning")
    print(f"  Model   : {cfg.model.name}")
    print(f"  Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"  Smoke   : {smoke}")
    print(f"  HF_HOME : {os.environ.get('HF_HOME')}")
    print(f"  HF_DATASETS_CACHE : {os.environ.get('HF_DATASETS_CACHE')}")
    print(f"  HF_HUB_CACHE      : {os.environ.get('HF_HUB_CACHE')}")
    print(f"{'='*60}\n")

    _ensure_gpu_has_headroom(min_free_gb=20.0 if smoke else 35.0)

    has_gpu = torch.cuda.is_available()
    supports_bf16 = has_gpu and torch.cuda.is_bf16_supported()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. IndicVoices is gated, so export HF_TOKEN before running."
        )

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
        feature_size=cfg.data.get("feature_size", 128),
        token=hf_token,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )

    print("Loading model...")
    model = load_model(cfg)

    if smoke:
        use_bf16 = bool(supports_bf16)
        use_fp16 = bool(has_gpu and not use_bf16)

        train_bs = 1
        eval_bs = 1
        grad_accum = 1
        use_gradient_checkpointing = False

        print("[SMOKE TEST] Low-VRAM settings enabled.")
        print("[SMOKE TEST] Gradient checkpointing disabled.")
    else:
        use_bf16 = bool(t.bf16 and has_gpu)
        use_fp16 = bool(getattr(t, "fp16", False) and has_gpu and not use_bf16)

        train_bs = t.per_device_train_batch_size
        eval_bs = t.per_device_eval_batch_size
        grad_accum = t.gradient_accumulation_steps
        use_gradient_checkpointing = True

    model = _convert_model_precision_for_run(model, use_bf16=use_bf16, use_fp16=use_fp16)
    model.config.use_cache = False

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()

    _check_model_has_finite_params(model)

    print(f"Run precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    print(f"Model dtype before trainer: {next(model.parameters()).dtype}\n")

    print("Connecting to dataset (streaming)...")
    data_config = DataConfig.from_omega(cfg)

    train_dataset = build_train_dataset(data_config, processor, token=hf_token)
    eval_dataset = build_eval_dataset(data_config, processor, token=hf_token)
    print("Dataset ready.\n")

    model_dtype = next(model.parameters()).dtype
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        model_dtype=model_dtype,
    )

    compute_metrics = make_compute_metrics(processor.tokenizer)

    report_to_value = _normalize_report_to(getattr(t, "report_to", None))
    print(f"Reporting integrations: {report_to_value}\n")

    training_args = Seq2SeqTrainingArguments(
        output_dir=t.output_dir,
        max_steps=t.max_steps,
        warmup_steps=t.warmup_steps,

        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,

        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        weight_decay=t.weight_decay,

        eval_strategy="no" if smoke else t.eval_strategy,
        eval_steps=None if smoke else t.eval_steps,
        save_strategy="no" if smoke else t.save_strategy,
        save_steps=None if smoke else t.save_steps,
        load_best_model_at_end=False if smoke else t.load_best_model_at_end,
        metric_for_best_model=None if smoke else t.metric_for_best_model,
        greater_is_better=None if smoke else t.greater_is_better,

        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gradient_checkpointing else None,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        predict_with_generate=False if smoke else t.predict_with_generate,
        generation_max_length=t.generation_max_length,

        logging_steps=1 if smoke else t.logging_steps,
        report_to=report_to_value,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    callbacks = [
        SaveCheckpointCallback(),
        EarlyStoppingOnWER(patience=5, min_delta=0.001),
    ]

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

    print("Starting training...\n")
    trainer.train()

    final_dir = Path(t.output_dir) / "final_model"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nModel saved → {final_dir}")


if __name__ == "__main__":
    main()