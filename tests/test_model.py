"""Unit tests: model loading, collator behaviour."""

import numpy as np
import pytest
import torch


def test_collator_pads_labels():
    from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.pad = lambda features, **kw: type("B", (), {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    })()

    fe = MagicMock()
    fe.pad = lambda features, **kw: type("B", (), {
        "input_features": torch.zeros(2, 128, 3000),
    })()

    processor = MagicMock()
    processor.feature_extractor = fe
    processor.tokenizer         = tokenizer

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=1,
        model_dtype=None,
    )

    features = [
        {"input_features": np.zeros((128, 3000)), "labels": [1, 2, 3]},
        {"input_features": np.zeros((128, 3000)), "labels": [4, 5]},
    ]
    batch = collator(features)
    assert "input_features" in batch
    assert "labels"         in batch
    # Padding positions should be -100
    assert (batch["labels"] == -100).any()


def test_collator_strips_bos():
    """Labels starting with decoder_start_token_id should have it stripped."""
    from src.data.collator import DataCollatorSpeechSeq2SeqWithPadding
    from unittest.mock import MagicMock

    decoder_start_id = 50258

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.pad = lambda features, **kw: type("B", (), {
        "input_ids": torch.tensor([[decoder_start_id, 1, 2], [decoder_start_id, 3, 4]]),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    })()

    fe = MagicMock()
    fe.pad = lambda features, **kw: type("B", (), {
        "input_features": torch.zeros(2, 128, 3000),
    })()

    processor = MagicMock()
    processor.feature_extractor = fe
    processor.tokenizer         = tokenizer

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=decoder_start_id,
        model_dtype=None,
    )

    features = [
        {"input_features": np.zeros((128, 3000)), "labels": [decoder_start_id, 1, 2]},
        {"input_features": np.zeros((128, 3000)), "labels": [decoder_start_id, 3, 4]},
    ]
    batch = collator(features)
    # BOS should be stripped — first token should not be decoder_start_id
    assert batch["labels"][0, 0].item() != decoder_start_id


def test_postprocess_whitespace():
    from src.inference.postprocess import postprocess
    assert postprocess("  hello   world  ") == "hello world"


def test_postprocess_remove_punctuation():
    from src.inference.postprocess import postprocess
    result = postprocess("hello, world!", remove_punctuation=True)
    assert "," not in result
    assert "!" not in result


def test_postprocess_empty():
    from src.inference.postprocess import postprocess
    assert postprocess("") == ""