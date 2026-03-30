"""
Seq2SeqTrainer subclass for Whisper + LoRA finetuning.

For vanilla (non-PEFT) finetuning the base Seq2SeqTrainer handles
everything correctly. This subclass is kept as a thin wrapper so we
can add custom behaviour later (e.g. logging, LR monitoring) without
changing the training script.
"""

from transformers import Seq2SeqTrainer

class WhisperTrainer(Seq2SeqTrainer):
    """
    Thin subclass of Seq2SeqTrainer — no overrides for vanilla finetuning.
    Extend here when custom training step behaviour is needed.
    """
    pass