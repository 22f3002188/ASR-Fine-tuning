"""
Text post-processing for Whisper ASR output.

Whisper outputs raw decoded text — this module applies normalisation steps
appropriate for Punjabi (Gurmukhi script) before downstream use.
"""

import re
from typing import Optional


def postprocess(
    text: str,
    remove_punctuation: bool = False,
    normalise_whitespace: bool = True,
    strip_filler_words: bool = False,
) -> str:
    """
    Apply post-processing to a single transcript string.

    Args:
        text                : raw Whisper output string
        remove_punctuation  : strip punctuation (useful for WER computation)
        normalise_whitespace: collapse multiple spaces, strip leading/trailing
        strip_filler_words  : remove common Punjabi filler words (ਉਹ, ਮਤਲਬ)

    Returns:
        cleaned string
    """
    if not text:
        return text

    if normalise_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    if remove_punctuation:
        # Remove standard ASCII punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Remove Devanagari/Gurmukhi punctuation
        text = re.sub(r"[।॥۔،؟!]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

    if strip_filler_words:
        # Common Punjabi fillers — extend as needed
        fillers = ["ਉਹ", "ਮਤਲਬ", "ਯਾਨੀ", "ਹਾਂ", "ਜੀ"]
        pattern = r"\b(" + "|".join(re.escape(f) for f in fillers) + r")\b"
        text = re.sub(pattern, "", text)
        text = re.sub(r"\s+", " ", text).strip()

    return text


def batch_postprocess(texts: list[str], **kwargs) -> list[str]:
    """Apply postprocess() to a list of strings."""
    return [postprocess(t, **kwargs) for t in texts]