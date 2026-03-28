"""
Data collator for Whisper Seq2Seq finetuning.
Directly matches DataCollatorSpeechSeq2SeqWithPadding from fine_tune_whisper.ipynb (Section 6).

Two-part padding:
  - input_features : already fixed-length 30s — just stack into tensor
  - labels         : pad to max length in batch, replace pad token id with -100
                     so CrossEntropyLoss ignores padding positions
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import WhisperProcessor


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Matches the collator from fine_tune_whisper.ipynb exactly.

    Args:
        processor              : WhisperProcessor (wraps feature_extractor + tokenizer)
        decoder_start_token_id : model.config.decoder_start_token_id
                                 Used to strip the leading BOS token that the
                                 Trainer appends automatically during generation.

    Usage:
        processor = WhisperProcessor.from_pretrained(model_name, language=..., task=...)
        collator  = DataCollatorSpeechSeq2SeqWithPadding(
                        processor=processor,
                        decoder_start_token_id=model.config.decoder_start_token_id,
                    )
        trainer = Seq2SeqTrainer(..., data_collator=collator)
    """
    processor: Any
    decoder_start_token_id: int
    model_dtype: Any = None    # e.g. torch.float16 — cast input_features to match model weights

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:

        # ── Audio features ────────────────────────────────────────────────────
        # Already fixed-length (30s = 3000 frames) — just stack into a tensor.
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # feature_extractor.pad() always returns float32. Cast to float16 here
        # so it matches model weight dtype under DataParallel / bf16 training.
        # The Trainer does this automatically in single-GPU mode but not always
        # through DataParallel — doing it in the collator is the safe universal fix.
        if self.model_dtype is not None:
            batch["input_features"] = batch["input_features"].to(self.model_dtype)

        # ── Labels ────────────────────────────────────────────────────────────
        # Pad to max length in this batch, then replace pad token id with -100.
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip BOS token from the start — the Trainer appends it automatically
        # during generation, so having it in labels causes a length mismatch.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch