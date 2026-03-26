"""
Learning rate scheduler utilities.

The Seq2SeqTrainer handles scheduling internally via TrainingArguments
(lr_scheduler_type + warmup_steps). This module exposes a helper to
compute a good warmup_steps value if you want to derive it from
your estimated dataset size rather than setting it manually.
"""


def suggested_warmup_steps(
    estimated_train_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    warmup_fraction: float = 0.03,
) -> int:
    """
    Derive a warmup_steps value from dataset size.

    Standard practice: warm up for ~3% of total steps. This is a rule of
    thumb — adjust based on how stable your initial loss curve looks.

    Args:
        estimated_train_samples    : approximate number of training samples
                                     (hard to know exactly with streaming;
                                      use dataset card estimates)
        batch_size                 : per_device_train_batch_size *
                                     num_gpus * gradient_accumulation_steps
        gradient_accumulation_steps: as set in train.yaml
        warmup_fraction            : fraction of total steps to use for warmup

    Returns:
        Recommended warmup_steps (rounded to nearest 100)

    Example:
        # IndicVoices Punjabi ~200k samples, effective batch=64, 10 epochs
        steps = suggested_warmup_steps(200_000, batch_size=64)
        # → warmup for ~3% of 3125 steps ≈ 100 steps
    """
    effective_batch = batch_size * gradient_accumulation_steps
    total_steps     = estimated_train_samples // effective_batch
    warmup          = int(total_steps * warmup_fraction)
    return max(100, round(warmup / 100) * 100)   # floor at 100, snap to nearest 100