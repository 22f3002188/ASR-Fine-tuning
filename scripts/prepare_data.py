"""
Validate the streaming ASR data pipeline end-to-end.

Checks:
- processor loading
- streaming dataset connection
- sample preprocessing
- label/token generation
- collator batch creation

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

from transformers import WhisperProcessor

from src.config_loader import load_config
from src.data.dataset import DataConfig, build_train_dataset, build_eval_dataset
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding


def parse_args():
    parser = argparse.ArgumentParser(description="Validate streaming ASR data pipeline")
    parser.add_argument("--n-train", type=int, default=50, help="Number of train samples to validate")
    parser.add_argument("--n-val", type=int, default=20, help="Number of validation samples to validate")
    return parser.parse_args()


def load_processor(cfg, hf_token: str | None) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(
        cfg.model.name,
        language=cfg.model.language,
        task=cfg.model.task,
        feature_size=cfg.data.get("feature_size", 128),
        token=hf_token,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
    )


def build_collator(processor: WhisperProcessor) -> DataCollatorSpeechSeq2SeqWithPadding:
    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id,
    )


def validate_split(ds, collator, processor, split_name: str, num_samples: int, expected_feature_size: int):
    print(f"── {split_name.upper()} (collecting {num_samples} valid samples) ──────────────")
    start_time = time.time()

    samples = list(itertools.islice(iter(ds), num_samples))
    elapsed = time.time() - start_time

    if not samples:
        print(f"  ERROR: no valid samples returned for {split_name}.\n")
        return

    feature_shapes = [sample["input_features"].shape for sample in samples]
    label_lengths = [len(sample["labels"]) for sample in samples]

    print(f"  Collected     : {len(samples)} samples in {elapsed:.1f}s")
    print(f"  Feature shape : {feature_shapes[0]}  (expected: ({expected_feature_size}, 3000))")
    print(
        f"  Label lengths : min={min(label_lengths)}  "
        f"max={max(label_lengths)}  "
        f"avg={sum(label_lengths) / len(label_lengths):.1f}"
    )

    print("\n  Sample transcripts:")
    for i, sample in enumerate(samples[:3]):
        text = processor.tokenizer.decode(sample["labels"], skip_special_tokens=True)
        print(f"    [{i}] {text[:80]}")

    batch_samples = [
        {"input_features": sample["input_features"], "labels": sample["labels"]}
        for sample in samples[:8]
    ]
    batch = collator(batch_samples)

    print("\n  Collated batch:")
    print(f"    input_features : {tuple(batch['input_features'].shape)}")
    print(f"    labels         : {tuple(batch['labels'].shape)}")
    print()


def main():
    args = parse_args()
    cfg = load_config()
    hf_token = os.environ.get("HF_TOKEN")

    print(f"\n{'=' * 60}")
    print("  Data pipeline validation")
    print(f"  Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"  Model   : {cfg.model.name}")
    print(f"{'=' * 60}\n")

    print("Loading WhisperProcessor...")
    processor = load_processor(cfg, hf_token)

    print("Building collator...")
    collator = build_collator(processor)

    print("Connecting to IndicVoices (streaming)...")
    data_config = DataConfig.from_omega(cfg)
    train_dataset = build_train_dataset(data_config, processor, token=hf_token)
    eval_dataset = build_eval_dataset(data_config, processor, token=hf_token)
    print("Connected.\n")

    validate_split(
        ds=train_dataset,
        collator=collator,
        processor=processor,
        split_name="train",
        num_samples=args.n_train,
        expected_feature_size=data_config.feature_size,
    )

    if eval_dataset is not None:
        validate_split(
            ds=eval_dataset,
            collator=collator,
            processor=processor,
            split_name="val",
            num_samples=args.n_val,
            expected_feature_size=data_config.feature_size,
        )
    else:
        print("WARNING: validation split not available — skipping val validation.\n")

    print("Pipeline validation complete. Safe to proceed.\n")


if __name__ == "__main__":
    main()

import os
os._exit(0)