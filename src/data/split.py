"""
Split utilities for streaming datasets.

With HF streaming there are no pre-computed splits to write to disk.
This module handles two cases:
  1. Dataset already has train/val/test splits (Common Voice, etc.) → use as-is.
  2. Dataset only has a single split → carve out val/test via .take() / .skip().
"""

from datasets import IterableDataset


def split_from_single(
    ds: IterableDataset,
    val_size: int = 500,
    test_size: int = 500,
) -> dict[str, IterableDataset]:
    """
    Carve val and test out of a single streaming split using .take() / .skip().

    NOTE: This is deterministic only when the upstream dataset has a fixed order
    (i.e. shuffle=False upstream). Always shuffle AFTER splitting, not before.

    Args:
        ds        : the single IterableDataset (e.g. loaded["train"])
        val_size  : number of samples to reserve for validation
        test_size : number of samples to reserve for test

    Returns:
        dict with keys "train", "val", "test"
    """
    holdout = val_size + test_size

    return {
        "train": ds.skip(holdout),
        "val":   ds.take(val_size),
        "test":  ds.skip(val_size).take(test_size),
    }
