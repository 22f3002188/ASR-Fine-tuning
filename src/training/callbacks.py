"""
Custom HuggingFace Trainer callbacks.

  - SavePeftAdapterCallback : saves only the LoRA adapter weights at each
                              checkpoint, not the full model. Keeps checkpoint
                              size from ~6GB (full large-v3) to ~50MB.

  - EarlyStoppingOnWER      : stops training if WER hasn't improved by a
                              minimum delta over a patience window. The built-in
                              EarlyStoppingCallback works on loss — this one
                              works on WER directly, which is what matters.
"""

import os
import shutil

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class SavePeftAdapterCallback(TrainerCallback):
    """
    At every checkpoint save, write only the LoRA adapter weights.

    The full model weights are unchanged and don't need to be saved —
    they can be reloaded from the HuggingFace Hub at merge time.
    This keeps each checkpoint at ~50MB instead of ~6GB for large-v3.

    The adapter is saved to:
        {output_dir}/checkpoint-{step}/adapter_model/
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "save_pretrained"):
            return control

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        adapter_dir = os.path.join(checkpoint_dir, "adapter_model")
        model.save_pretrained(adapter_dir)
        print(f"[SavePeftAdapterCallback] Adapter saved → {adapter_dir}")
        return control


class EarlyStoppingOnWER(TrainerCallback):
    """
    Stop training if WER has not improved by at least `min_delta`
    over the last `patience` evaluations.

    Args:
        patience  : number of eval steps without improvement before stopping
        min_delta : minimum absolute WER improvement to count as progress
                    (e.g. 0.001 = 0.1 WER point)
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience     = patience
        self.min_delta    = min_delta
        self._best_wer    = float("inf")
        self._no_improve  = 0

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
                f"[EarlyStoppingOnWER] No improvement for {self._no_improve}/{self.patience} "
                f"evals. Best WER: {self._best_wer:.4f}, current: {wer:.4f}"
            )

        if self._no_improve >= self.patience:
            print(f"[EarlyStoppingOnWER] Stopping — no WER improvement for {self.patience} evals.")
            control.should_training_stop = True

        return control
