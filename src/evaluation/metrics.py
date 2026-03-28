"""
WER / CER metric computation for Whisper Seq2SeqTrainer.

make_compute_metrics() returns a callable that the Trainer calls
after every evaluation loop with (predictions, label_ids).
"""

import numpy as np
from jiwer import wer, cer
from transformers import WhisperTokenizer


def make_compute_metrics(tokenizer: WhisperTokenizer):
    """
    Build a compute_metrics function bound to the given tokenizer.

    The returned function:
      1. Replaces -100 padding in labels with pad_token_id
      2. Batch-decodes predictions and labels, skipping special tokens
      3. Computes and returns WER + CER

    Args:
        tokenizer : WhisperTokenizer (used for batch_decode)

    Returns:
        Callable[[EvalPrediction], dict] expected by Seq2SeqTrainer
    """

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids

        # Replace -100 (padding) with pad_token_id so decode doesn't crash
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # Decode — skip special tokens (language id, task id, timestamps, etc.)
        predictions = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        references  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # jiwer expects lists of strings
        wer_score = wer(references, predictions)
        cer_score = cer(references, predictions)

        return {
            "wer": round(wer_score, 4),
            "cer": round(cer_score, 4),
        }

    return compute_metrics