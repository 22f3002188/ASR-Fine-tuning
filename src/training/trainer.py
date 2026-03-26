from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

from src.data.dataset import (
    DataConfig,
    build_train_dataset,
    build_eval_dataset,
)
from src.model.model import build_processor as _build_processor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    max_label_length: int = 256

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"][: self.max_label_length]} for f in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1),
            -100,
        )

        bos_token_id = self.processor.tokenizer.bos_token_id
        if bos_token_id is not None and labels.size(1) > 0:
            if bool((labels[:, 0] == bos_token_id).all()):
                labels = labels[:, 1:]

        batch["labels"] = labels
        batch["language"] = [f["language"] for f in features]
        batch["domain"] = [f["domain"] for f in features]
        batch["text"] = [f["text"] for f in features]
        batch["duration"] = torch.tensor([f["duration"] for f in features], dtype=torch.float32)

        return batch


def build_processor(model_name: str) -> WhisperProcessor:
    return _build_processor(model_name=model_name)


def build_data_collator(
    processor: WhisperProcessor,
    max_label_length: int = 256,
) -> DataCollatorSpeechSeq2SeqWithPadding:
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        max_label_length=max_label_length,
    )


def build_train_dataloader(
    data_config: DataConfig,
    processor: WhisperProcessor,
    collator: DataCollatorSpeechSeq2SeqWithPadding,
    batch_size: int,
    hf_token: Optional[str] = None,
) -> DataLoader:
    dataset = build_train_dataset(
        config=data_config,
        processor=processor,
        token=hf_token,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_eval_dataloader(
    data_config: DataConfig,
    processor: WhisperProcessor,
    collator: DataCollatorSpeechSeq2SeqWithPadding,
    batch_size: int,
    hf_token: Optional[str] = None,
) -> Optional[DataLoader]:
    dataset = build_eval_dataset(
        config=data_config,
        processor=processor,
        token=hf_token,
    )

    if dataset is None:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def summarize_batch(batch):
    print(f"input_features shape: {batch['input_features'].shape}")
    print(f"labels shape: {batch['labels'].shape}")

    # Language check
    if "language" in batch:
        from collections import Counter
        print("language counts:", dict(Counter(batch["language"])))

    # Domain check
    if "domain" in batch:
        from collections import Counter
        print("domain counts:", dict(Counter(batch["domain"])))

    # Duration check
    if "duration" in batch:
        durations = batch["duration"]
        print("duration stats:", {
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations)/len(durations)
        })

    #  NEW: PRINT FEW SAMPLES
    print("\n Sample check:")
    for i in range(min(3, len(batch["text"]))):
        print(f"\nSample {i+1}:")
        print(f"Domain: {batch['domain'][i]}")
        print(f"Duration: {batch['duration'][i]:.2f}s")
        print(f"Text: {batch['text'][i][:100]}")