"""
Split utilities for streaming datasets.

IndicVoices already ships train + valid splits, so this module is mostly
a fallback for datasets that only have a single split.
"""

from datasets import IterableDataset


def split_from_single(
    ds: IterableDataset,
    val_size: int = 1000,
    test_size: int = 500,
) -> dict[str, IterableDataset]:
    """
    Carve val + test out of a single streaming split using .take() / .skip().

    IMPORTANT: .take() and .skip() are NOT shuffled — always shuffle AFTER
    splitting, not before, to avoid val/test contamination from any ordering
    bias in the source dataset (e.g. speakers sorted alphabetically).

    Args:
        ds        : source IterableDataset (unshuffled)
        val_size  : samples to reserve for validation
        test_size : samples to reserve for test

    Returns:
        dict with "train", "val", "test" IterableDatasets
    """
    holdout = val_size + test_size
    return {
        "train": ds.skip(holdout),
        "val":   ds.take(val_size),
        "test":  ds.skip(val_size).take(test_size),
    }