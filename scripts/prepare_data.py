"""
Step 2 entrypoint: validate the streaming data pipeline end-to-end.

Streams samples through the full pipeline:
  open_stream → extract_audio → load_audio_manually → preprocess → collate

Run:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --n-train 100 --n-val 50
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import WhisperProcessor, WhisperConfig

from src.config_loader import load_config
from src.data.dataset import DataConfig, build_train_dataset, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--n-val",   type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()

    model_name = cfg.model.name
    print(f"\n{'='*60}")
    print(f"  Data pipeline validation")
    print(f"  Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"  Model   : {model_name}")
    print(f"{'='*60}\n")

    # ── Processor ─────────────────────────────────────────────────────────────
    print("Loading WhisperProcessor...")
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=cfg.model.language,
        task=cfg.model.task,
        feature_size=cfg.data.get("feature_size", 128),
    )

    # ── Collator ──────────────────────────────────────────────────────────────
    model_cfg = WhisperConfig.from_pretrained(model_name)
    collator  = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model_cfg.decoder_start_token_id,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    hf_token    = os.environ.get("HF_TOKEN")
    data_config = DataConfig.from_omega(cfg)

    print("Connecting to IndicVoices (streaming)...")
    train_ds = build_train_dataset(data_config, processor, token=hf_token)
    eval_ds  = build_eval_dataset(data_config,  processor, token=hf_token)
    print("Connected.\n")

    # ── Validate splits ────────────────────────────────────────────────────────
    _validate_split(train_ds, collator, processor, split="train", n=args.n_train)

    if eval_ds is not None:
        _validate_split(eval_ds, collator, processor, split="val", n=args.n_val)
    else:
        print("WARNING: eval split not available — skipping val validation.\n")

    print("Pipeline validation complete. Safe to proceed to Step 3.\n")


def _validate_split(ds, collator, processor, split: str, n: int):
    print(f"── {split.upper()}  (collecting {n} valid samples) ──────────────")
    t0 = time.time()

    samples = list(itertools.islice(iter(ds), n))
    elapsed = time.time() - t0

    if not samples:
        print(f"  ERROR: no valid samples returned for {split}.\n")
        return

    feat_shapes = [s["input_features"].shape for s in samples]
    label_lens  = [len(s["labels"]) for s in samples]

    print(f"  Collected     : {len(samples)} samples in {elapsed:.1f}s")
    print(f"  Feature shape : {feat_shapes[0]}  (expected: ({128 if 'large' in str(feat_shapes[0]) else 80}, 3000))")
    print(f"  Label lengths : min={min(label_lens)}  max={max(label_lens)}  "
          f"avg={sum(label_lens)/len(label_lens):.1f}")

    print(f"\n  Sample transcripts:")
    for i, s in enumerate(samples[:3]):
        text = processor.tokenizer.decode(s["labels"], skip_special_tokens=True)
        print(f"    [{i}] {text[:80]}")

    # Collation check — strip non-model keys first
    model_keys = {"input_features", "labels"}
    batch_samples = [{k: v for k, v in s.items() if k in model_keys} for s in samples[:8]]
    batch = collator(batch_samples)
    print(f"\n  Collated batch:")
    print(f"    input_features : {tuple(batch['input_features'].shape)}")
    print(f"    labels         : {tuple(batch['labels'].shape)}")
    print()


if __name__ == "__main__":
    main()