"""
Unified Whisper fine-tuning script.

Features:
- optional LoRA
- optional parameter freezing
- HF cache setup before HF imports
- streaming dataset support
- smoke test mode
- MLflow forced ON
- GPU headroom check
- safer precision handling
- optional validation/evaluation
- final model save
"""

from __future__ import annotations

import os
from pathlib import Path

# ==========================================================
# FORCE HF CACHE PATHS BEFORE ANY HF IMPORTS
# ==========================================================
HF_BASE = os.environ.get("HF_HOME", str(Path.home() / "hf_cache"))

os.environ["HF_HOME"] = HF_BASE
os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", f"{HF_BASE}/hub")
os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", f"{HF_BASE}/datasets")
os.environ["HF_ASSETS_CACHE"] = os.environ.get("HF_ASSETS_CACHE", f"{HF_BASE}/assets")
os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", f"{HF_BASE}/transformers")
os.environ["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", HF_BASE)

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

for p in [
    os.environ["HF_HOME"],
    os.environ["HF_HUB_CACHE"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["HF_ASSETS_CACHE"],
    os.environ["TRANSFORMERS_CACHE"],
]:
    Path(p).mkdir(parents=True, exist_ok=True)

print("DEBUG HF_HOME =", os.environ["HF_HOME"])
print("DEBUG HF_HUB_CACHE =", os.environ["HF_HUB_CACHE"])
print("DEBUG HF_DATASETS_CACHE =", os.environ["HF_DATASETS_CACHE"])
print("DEBUG HF_ASSETS_CACHE =", os.environ["HF_ASSETS_CACHE"])
print("DEBUG TRANSFORMERS_CACHE =", os.environ["TRANSFORMERS_CACHE"])
print("DEBUG XDG_CACHE_HOME =", os.environ["XDG_CACHE_HOME"])

import inspect
import socket
import subprocess
import sys
from typing import Any, Dict, Optional

import torch
from transformers import Seq2SeqTrainingArguments, WhisperProcessor

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    _MLFLOW_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.model.model import load_model
from src.data.dataset import DataConfig, build_train_dataset, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.trainer import WhisperTrainer
from src.evaluation.metrics import make_compute_metrics

try:
    from src.model.lora import apply_lora, print_trainable_parameters
except Exception:
    apply_lora = None
    print_trainable_parameters = None

try:
    from src.model.freeze import apply_freeze
except Exception:
    apply_freeze = None

try:
    from src.training.callbacks import SaveCheckpointCallback, EarlyStoppingOnWER
except Exception:
    SaveCheckpointCallback = None
    EarlyStoppingOnWER = None


def gpu_report() -> Optional[list[str]]:
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
        return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    except Exception:
        return None


def ensure_gpu_has_headroom(min_free_gb: float = 20.0) -> None:
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training will run on CPU.")
        return

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)

        print(f"GPU free memory: {free_gb:.2f} GiB / {total_gb:.2f} GiB")

        report = gpu_report()
        if report:
            print("GPU report:")
            for line in report:
                print(f"  {line}")
            print()

        if free_gb < min_free_gb:
            raise RuntimeError(
                f"Not enough free GPU memory. Need about {min_free_gb:.1f} GiB free, found {free_gb:.2f} GiB."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"WARNING: Could not verify GPU headroom: {exc}")


def get_precision_flags(cfg_training: Any, smoke: bool) -> tuple[bool, bool]:
    has_gpu = torch.cuda.is_available()
    supports_bf16 = has_gpu and torch.cuda.is_bf16_supported()

    if smoke:
        use_bf16 = bool(supports_bf16)
        use_fp16 = bool(has_gpu and not use_bf16)
        return use_bf16, use_fp16

    requested_bf16 = bool(getattr(cfg_training, "bf16", False))
    requested_fp16 = bool(getattr(cfg_training, "fp16", False))

    use_bf16 = bool(requested_bf16 and has_gpu and supports_bf16)
    use_fp16 = bool(requested_fp16 and has_gpu and not use_bf16)
    return use_bf16, use_fp16


def convert_model_precision_for_run(model: torch.nn.Module, use_bf16: bool, use_fp16: bool):
    if use_bf16:
        return model.to(dtype=torch.bfloat16)
    if use_fp16:
        return model.to(dtype=torch.float16)
    return model.to(dtype=torch.float32)


def check_model_has_trainable_params(model: torch.nn.Module) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters found after LoRA/freeze setup.")


def check_model_has_finite_params(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param is None:
            continue
        if not torch.isfinite(param.data).all():
            raise RuntimeError(f"Non-finite parameter detected before training: {name}")


def get_cfg_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj.get(key, default)
    except Exception:
        return default


def has_enabled_section(cfg: Any, section_name: str) -> bool:
    section = getattr(cfg, section_name, None)
    if section is None:
        return False
    return bool(get_cfg_value(section, "enabled", False))


def build_run_name(cfg: Any, smoke: bool) -> str:
    model_name = str(cfg.model.name).split("/")[-1]
    suffix = "smoke" if smoke else "full"
    return f"{cfg.data.language}_{model_name}_{suffix}"


def numeric_metrics_only(metrics: Dict[str, Any]) -> Dict[str, float]:
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            clean[k] = float(v)
    return clean


def build_training_args_kwargs(base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    params = sig.parameters

    kwargs = dict(base_kwargs)
    eval_value = kwargs.pop("evaluation_strategy", None)

    if "evaluation_strategy" in params and eval_value is not None:
        kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in params and eval_value is not None:
        kwargs["eval_strategy"] = eval_value

    return kwargs


def enable_input_require_grads_if_needed(model: torch.nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        print("Enabled input requires_grad via model.enable_input_require_grads()")
        return

    if hasattr(model, "get_input_embeddings"):
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is not None:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            input_embeddings.register_forward_hook(make_inputs_require_grad)
            print("Enabled input requires_grad via embedding forward hook")


def main() -> None:
    cfg = load_config()
    t = cfg.training
    smoke = os.environ.get("SMOKE_TEST", "false").lower() == "true"

    print(f"\n{'=' * 60}")
    print("  Whisper finetuning")
    print(f"  Model   : {cfg.model.name}")
    print(f"  Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"  Smoke   : {smoke}")
    print(f"{'=' * 60}\n")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set.")

    print("HF cache paths:")
    print(f"  HF_HOME              : {os.environ['HF_HOME']}")
    print(f"  HF_HUB_CACHE         : {os.environ['HF_HUB_CACHE']}")
    print(f"  HF_DATASETS_CACHE    : {os.environ['HF_DATASETS_CACHE']}")
    print(f"  HF_ASSETS_CACHE      : {os.environ['HF_ASSETS_CACHE']}")
    print(f"  TRANSFORMERS_CACHE   : {os.environ['TRANSFORMERS_CACHE']}")
    print(f"  XDG_CACHE_HOME       : {os.environ['XDG_CACHE_HOME']}\n")

    ensure_gpu_has_headroom(min_free_gb=20.0 if smoke else 35.0)

    use_bf16, use_fp16 = get_precision_flags(t, smoke)

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
        token=hf_token,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )

    print("Loading model...")
    model = load_model(cfg)
    model.config.use_cache = False

    if has_enabled_section(cfg, "lora"):
        if apply_lora is None:
            raise RuntimeError("LoRA is enabled in config, but src.model.lora could not be imported.")
        print("Applying LoRA adapters...")
        model = apply_lora(model, cfg)
        print("LoRA applied.\n")

    if hasattr(cfg, "freeze") and get_cfg_value(cfg, "freeze", None) is not None:
        if apply_freeze is None:
            print("WARNING: freeze config found, but src.model.freeze could not be imported. Skipping freeze.")
        else:
            print("Applying parameter freezing...")
            apply_freeze(model, cfg)
            print()

    if print_trainable_parameters is not None:
        print("Parameter summary:")
        print_trainable_parameters(model)
        print()

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = convert_model_precision_for_run(model, use_bf16=use_bf16, use_fp16=use_fp16)

    check_model_has_trainable_params(model)
    check_model_has_finite_params(model)

    precision_name = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print(f"Run precision: {precision_name}")
    print(f"Model dtype before trainer: {next(model.parameters()).dtype}\n")

    print("Connecting to dataset (streaming)...")
    data_cfg = DataConfig.from_omega(cfg)

    train_ds = build_train_dataset(data_cfg, processor, token=hf_token)
    eval_ds = build_eval_dataset(data_cfg, processor, token=hf_token)

    if eval_ds is None:
        print("No validation dataset found. Evaluation will be disabled.\n")
        evaluation_strategy = "no"
    else:
        evaluation_strategy = "no" if smoke else get_cfg_value(t, "eval_strategy", "steps")

    print("Dataset ready.\n")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        model_dtype=next(model.parameters()).dtype,
    )

    compute_metrics = make_compute_metrics(processor.tokenizer)

    # ==========================================================
    # FORCE MLFLOW ON
    # ==========================================================
    use_mlflow = _MLFLOW_AVAILABLE
    if not use_mlflow:
        raise RuntimeError("MLflow is not installed, but this script forces MLflow logging.")

    report_to_value = ["mlflow"]
    run_name = build_run_name(cfg, smoke)

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "whisper-asr-finetuning")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    print("MLflow forced ON")
    print(f"Reporting integrations: {report_to_value}")
    print(f"Run name: {run_name}")
    print(f"MLflow tracking URI : {mlflow_tracking_uri}")
    print(f"MLflow experiment   : {mlflow_experiment_name}\n")

    train_bs = 1 if smoke else get_cfg_value(t, "per_device_train_batch_size", 1)
    eval_bs = 1 if smoke else get_cfg_value(t, "per_device_eval_batch_size", 1)
    grad_accum = 1 if smoke else get_cfg_value(t, "gradient_accumulation_steps", 1)
    max_steps = get_cfg_value(t, "smoke_test_steps", 10) if smoke else get_cfg_value(t, "max_steps", -1)

    gradient_checkpointing = False if smoke else True

    training_args_base = dict(
        output_dir=get_cfg_value(t, "output_dir"),
        max_steps=max_steps,
        warmup_steps=get_cfg_value(t, "warmup_steps", 0),

        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,

        learning_rate=get_cfg_value(t, "learning_rate", 1e-4),
        lr_scheduler_type=get_cfg_value(t, "lr_scheduler_type", "linear"),
        weight_decay=get_cfg_value(t, "weight_decay", 0.0),

        evaluation_strategy=evaluation_strategy,
        eval_steps=None if (smoke or evaluation_strategy == "no") else get_cfg_value(t, "eval_steps", None),

        save_strategy="no" if smoke else get_cfg_value(t, "save_strategy", "steps"),
        save_steps=None if smoke else get_cfg_value(t, "save_steps", None),

        load_best_model_at_end=False if (smoke or evaluation_strategy == "no") else get_cfg_value(t, "load_best_model_at_end", False),
        metric_for_best_model=None if (smoke or evaluation_strategy == "no") else get_cfg_value(t, "metric_for_best_model", None),
        greater_is_better=None if (smoke or evaluation_strategy == "no") else get_cfg_value(t, "greater_is_better", None),

        bf16=use_bf16,
        fp16=use_fp16,

        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        predict_with_generate=False if smoke else get_cfg_value(t, "predict_with_generate", True),
        generation_max_length=get_cfg_value(t, "generation_max_length", 225),

        logging_steps=1 if smoke else get_cfg_value(t, "logging_steps", 10),
        report_to=report_to_value,
        run_name=run_name,
        remove_unused_columns=False,
        max_grad_norm=1.0,
    )

    training_args = Seq2SeqTrainingArguments(**build_training_args_kwargs(training_args_base))

    if getattr(training_args, "gradient_checkpointing", False):
        enable_input_require_grads_if_needed(model)

    callbacks = []
    if SaveCheckpointCallback is not None and not smoke:
        callbacks.append(SaveCheckpointCallback())
    if EarlyStoppingOnWER is not None and not smoke and evaluation_strategy != "no":
        callbacks.append(EarlyStoppingOnWER(patience=5, min_delta=0.001))

    trainer = WhisperTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if evaluation_strategy != "no" else None,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        processing_class=processor.feature_extractor,
    )

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.set_tags(
            {
                "project": "ASR-Fine-tuning",
                "task": "automatic-speech-recognition",
                "framework": "huggingface-transformers",
                "model_family": "whisper",
                "model_name": str(cfg.model.name),
                "dataset_name": str(cfg.data.dataset_name),
                "language": str(cfg.data.language),
                "smoke_test": str(smoke),
                "host": socket.gethostname(),
                "precision": precision_name,
                "lora_enabled": str(has_enabled_section(cfg, "lora")),
            }
        )

        mlflow.log_params(
            {
                "model_name": str(cfg.model.name),
                "model_language": str(cfg.model.language),
                "model_task": str(cfg.model.task),
                "dataset_name": str(cfg.data.dataset_name),
                "dataset_language": str(cfg.data.language),
                "feature_size": int(get_cfg_value(cfg.data, "feature_size", 128)),
                "output_dir": str(get_cfg_value(t, "output_dir")),
                "learning_rate": float(get_cfg_value(t, "learning_rate", 1e-4)),
                "lr_scheduler_type": str(get_cfg_value(t, "lr_scheduler_type", "linear")),
                "weight_decay": float(get_cfg_value(t, "weight_decay", 0.0)),
                "max_steps": int(max_steps),
                "warmup_steps": int(get_cfg_value(t, "warmup_steps", 0)),
                "per_device_train_batch_size": int(train_bs),
                "per_device_eval_batch_size": int(eval_bs),
                "gradient_accumulation_steps": int(grad_accum),
                "bf16": bool(use_bf16),
                "fp16": bool(use_fp16),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "predict_with_generate": bool(False if smoke else get_cfg_value(t, "predict_with_generate", True)),
                "generation_max_length": int(get_cfg_value(t, "generation_max_length", 225)),
                "logging_steps": int(1 if smoke else get_cfg_value(t, "logging_steps", 10)),
            }
        )

        print("Starting training...\n")
        trainer.train()

        if not smoke and evaluation_strategy != "no":
            print("\nRunning final evaluation...\n")
            final_metrics = trainer.evaluate()
            print(f"Final eval metrics: {final_metrics}")
            mlflow.log_metrics(
                {f"final_{k}": v for k, v in numeric_metrics_only(final_metrics).items()}
            )

        save_path = Path(get_cfg_value(t, "output_dir")) / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)

        trainer.save_model(str(save_path))
        processor.save_pretrained(str(save_path))
        mlflow.log_artifacts(str(save_path), artifact_path="final_model")

        print(f"\nModel saved → {save_path}")


if __name__ == "__main__":
    main()