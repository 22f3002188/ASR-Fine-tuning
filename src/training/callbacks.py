"""
Custom Trainer callbacks for vanilla Whisper finetuning.

  - SaveCheckpointCallback : saves the full model + processor at each checkpoint.
  - EarlyStoppingOnWER     : stops training when WER stops improving.
"""

import os
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class SaveCheckpointCallback(TrainerCallback):
    """
    Save the full model and processor at every checkpoint step.
    The Trainer already saves optimizer state — this additionally
    saves the HuggingFace model format so the checkpoint is
    self-contained for inference.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model     = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")   # actually the feature_extractor here
        if model is None:
            return control

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        model.save_pretrained(checkpoint_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)

        print(f"[SaveCheckpointCallback] Saved → {checkpoint_dir}")
        return control


class EarlyStoppingOnWER(TrainerCallback):
    """
    Stop training if WER has not improved by at least `min_delta`
    over the last `patience` evaluations.

    Args:
        patience  : evaluations without improvement before stopping
        min_delta : minimum WER improvement to count as progress
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience    = patience
        self.min_delta   = min_delta
        self._best_wer   = float("inf")
        self._no_improve = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs,
    ):
        wer = metrics.get("eval_wer")
        if wer is None:
            return control

        if wer < self._best_wer - self.min_delta:
            self._best_wer   = wer
            self._no_improve = 0
        else:
            self._no_improve += 1
            print(
                f"[EarlyStoppingOnWER] No improvement {self._no_improve}/{self.patience}. "
                f"Best: {self._best_wer:.4f}  Current: {wer:.4f}"
            )

        if self._no_improve >= self.patience:
            print(f"[EarlyStoppingOnWER] Stopping.")
            control.should_training_stop = True

        return control


class SaveAdapterCallback(TrainerCallback):
    """
    BAFT-mode checkpoint callback.
    Saves only the bottleneck adapter weights at each checkpoint step.
    Each checkpoint is ~50MB instead of ~6GB for large-v3.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        from src.model.bottleneck_adapter import save_adapter_weights

        model = kwargs.get("model")
        if model is None:
            return control

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        save_adapter_weights(model, checkpoint_dir)
        print(f"[SaveAdapterCallback] Adapter saved → {checkpoint_dir}")
        return control