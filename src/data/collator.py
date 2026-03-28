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


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    model_dtype: Any = None

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if not torch.isfinite(batch["input_features"]).all():
            raise ValueError("NaN/Inf detected in input_features before model forward.")

        if self.model_dtype is not None:
            batch["input_features"] = batch["input_features"].to(self.model_dtype)

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        ).long()

        if labels.shape[1] > 1 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        valid_label_counts = labels.ne(-100).sum(dim=1)
        if (valid_label_counts == 0).any().cpu().item():
            bad_rows = (valid_label_counts == 0).nonzero(as_tuple=False).view(-1).tolist()
            raise ValueError(
                "Encountered samples with zero valid label tokens after padding/masking. "
                f"Bad row indices: {bad_rows}"
            )

        batch["labels"] = labels
        return batch