"""
Batch evaluation of a finetuned Whisper model on the IndicVoices val split.

Loads the saved model from checkpoints/final_model/, streams the val split,
runs generation, and computes WER / CER using jiwer.

Called by scripts/evaluate.py.
"""

import os
import itertools
from pathlib import Path
from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.evaluation.metrics import make_compute_metrics
from src.evaluation.error_analysis import ErrorAnalyser
from src.data.dataset import DataConfig, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from src.model.bottleneck_adapter import inject_bottleneck_adapters, load_adapter_weights


def run_evaluation(
    cfg,
    model_dir: Optional[str] = None,
    n_samples: Optional[int] = None,
) -> dict:
    """
    Stream the val split through the model and return WER/CER + error breakdown.

    Args:
        cfg       : merged OmegaConf config
        model_dir : path to saved model dir (defaults to checkpoints/final_model)
        n_samples : cap evaluation at N samples (None = full val split)

    Returns:
        dict with keys: wer, cer, substitutions, deletions, insertions,
                        per_domain, predictions (list of dicts)
    """
    model_dir = model_dir or str(Path(cfg.training.output_dir) / "final_model")
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_dir}...")

    baft_enabled = cfg.get("baft", {}).get("enabled", False)
    
    # ── Processor ALWAYS from base model ───────────────────────────────
    processor = WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
    )
    
    # ── Load model ─────────────────────────────────────────────────────
    if baft_enabled:
        print("BAFT detected → reconstructing model with adapters...")
    
        # 1. Load base model
        model = WhisperForConditionalGeneration.from_pretrained(
            cfg.model.name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    
        # 2. Inject adapters (same config as training!)
        baft_cfg = cfg.get("baft", {})
        model = inject_bottleneck_adapters(
            model,
            d=baft_cfg.get("d", 64),
            dropout=baft_cfg.get("dropout", 0.0),
        )
    
        # 3. Load adapter weights
        model = load_adapter_weights(model, model_dir)
    
    else:
        print("Full finetune model detected → loading directly...")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
    
    model = model.to(device)
    model.eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_config = DataConfig.from_omega(cfg)
    hf_token    = os.environ.get("HF_TOKEN")
    eval_ds     = build_eval_dataset(data_config, processor, token=hf_token)

    model_dtype = next(model.parameters()).dtype
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        model_dtype=model_dtype,
    )

    compute_metrics = make_compute_metrics(processor.tokenizer)

    # ── Stream + generate ─────────────────────────────────────────────────────
    stream = iter(eval_ds)
    if n_samples:
        stream = itertools.islice(stream, n_samples)

    model_keys  = {"input_features", "labels"}
    all_preds   = []
    all_refs    = []
    all_domains = []
    batch_size  = cfg.training.per_device_eval_batch_size
    batch_buf   = []

    def _process_batch(buf):
        clean = [{k: v for k, v in s.items() if k in model_keys} for s in buf]
        batch = collator(clean)
        input_features = batch["input_features"].to(device)
        labels         = batch["labels"]

        with torch.no_grad():
            pred_ids = model.generate(
                input_features,
                language=cfg.model.language,
                task=cfg.model.task,
                max_new_tokens=cfg.training.generation_max_length,
            )

        preds = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        refs  = processor.tokenizer.batch_decode(
            labels.masked_fill(labels == -100, processor.tokenizer.pad_token_id),
            skip_special_tokens=True,
        )
        return preds, refs

    print(f"Running evaluation on {'all' if not n_samples else n_samples} val samples...")
    seen = 0
    for sample in stream:
        batch_buf.append(sample)
        if len(batch_buf) >= batch_size:
            preds, refs = _process_batch(batch_buf)
            all_preds.extend(preds)
            all_refs.extend(refs)
            all_domains.extend([s.get("domain", "unknown") for s in batch_buf])
            batch_buf.clear()
            seen += batch_size
            if seen % 100 == 0:
                print(f"  Evaluated {seen} samples...")

    # Flush remainder
    if batch_buf:
        preds, refs = _process_batch(batch_buf)
        all_preds.extend(preds)
        all_refs.extend(refs)
        all_domains.extend([s.get("domain", "unknown") for s in batch_buf])

    # ── Metrics ───────────────────────────────────────────────────────────────
    from jiwer import wer, cer, process_words
    overall_wer = wer(all_refs, all_preds)
    overall_cer = cer(all_refs, all_preds)

    analyser   = ErrorAnalyser(all_refs, all_preds, all_domains)
    breakdown  = analyser.error_breakdown()
    per_domain = analyser.per_domain_wer()

    results = {
        "n_samples":     len(all_preds),
        "wer":           round(overall_wer, 4),
        "cer":           round(overall_cer, 4),
        **breakdown,
        "per_domain":    per_domain,
        "predictions":   [
            {"reference": r, "hypothesis": p, "domain": d}
            for r, p, d in zip(all_refs, all_preds, all_domains)
        ],
    }
    return results