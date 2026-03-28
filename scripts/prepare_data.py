"""
Step 2 entrypoint: validate the streaming data pipeline end-to-end.

Streams a small number of samples through the full pipeline:
  load → cast_column → decode_audio → filter → augment → feature extract
  → tokenize → collate → batch shape check

Run:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --n-train 100 --n-val 50
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import WhisperProcessor, WhisperConfig

from src.config_loader import load_config
from src.data.dataset import StreamingASRDataset, DataConfig
from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--n-val", type=int, default=20)
    return p.parse_args()


def summarize_batch(batch, samples, processor):
    print(f"\n  Collated batch:")
    print(f"    input_features : {tuple(batch['input_features'].shape)}")
    print(f"    labels         : {tuple(batch['labels'].shape)}")

    n_pad = (batch["labels"] == -100).sum(dim=1).tolist()
    print(f"    padding (-100) per sample: {n_pad}")

    print(f"\n  Sample check:")
    for i, s in enumerate(samples[:3]):
        text = processor.tokenizer.decode(
            s["labels"],
            skip_special_tokens=True
        )
        print(f"    [{i}] {text[:100]}")

    print()


def main():
    args = parse_args()
    cfg = load_config()

    model_name = cfg.model.name

    print(f"\n{'='*60}")
    print(f"  Data pipeline validation")
    print(f"  Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print(f"  Model   : {model_name}")
    print(f"{'='*60}\n")

    # Processor
    print("Loading WhisperProcessor...")
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=cfg.model.language,
        task=cfg.model.task,
    )

    # Dataset
    data_config = DataConfig.from_omega(cfg)

    ds = StreamingASRDataset(
        data_config,
        processor.feature_extractor,
        processor.tokenizer,
    )

    print("Connecting to IndicVoices (streaming)...")
    ds.load()
    print("Connected.\n")

    # Collator
    model_cfg = WhisperConfig.from_pretrained(model_name)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model_cfg.decoder_start_token_id,
    )

    # Validate splits
    for split, n, shuffle in [
        ("train", args.n_train, True),
        ("val", args.n_val, False),
    ]:
        _validate_split(
            ds,
            collator,
            processor,
            split=split,
            n=n,
            shuffle=shuffle,
        )

    print("Pipeline validation complete. Safe to proceed to Step 3.\n")

    # useful cleanup borrowed from incoming branch
    gc.collect()


def _validate_split(ds, collator, processor, split, n, shuffle):
    print(f"── {split.upper()}  ({n} samples) ──────────────────────────────")

    t0 = time.time()

    samples = ds.get_split_as_list(
        split,
        n_samples=n,
        shuffle=shuffle,
    )

    elapsed = time.time() - t0

    if not samples:
        print("  ERROR: no valid samples returned.")
        print("  Check audio_column and dataset access.\n")
        return

    feat_shapes = [s["input_features"].shape for s in samples]
    label_lens = [len(s["labels"]) for s in samples]

    print(f"  Loaded        : {len(samples)} samples in {elapsed:.1f}s")
    print(f"  Feature shape : {feat_shapes[0]}")
    print(
        f"  Label lengths : "
        f"min={min(label_lens)}  "
        f"max={max(label_lens)}  "
        f"avg={sum(label_lens)/len(label_lens):.1f} tokens"
    )

    batch = collator(samples[:8])

    summarize_batch(batch, samples, processor)


if __name__ == "__main__":
    main()