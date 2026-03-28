"""
Seq2SeqTrainer subclass for Whisper + LoRA finetuning.

Handles the one Whisper-specific quirk the base Trainer doesn't:
  - During evaluation with predict_with_generate=True, the PeftModel's
    generate() needs the same forced_decoder_ids that were set on
    generation_config. We pass them explicitly to avoid any config
    inheritance issues across PEFT wrapper layers.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalLoopOutput


class WhisperTrainer(Seq2SeqTrainer):
    """
    Thin subclass of Seq2SeqTrainer.

    Overrides prediction_step() to ensure forced_decoder_ids flow correctly
    through the PeftModel wrapper during generation.
    """

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        # Pull forced_decoder_ids from generation_config and inject into
        # inputs so they're passed to model.generate() explicitly.
        # This prevents the PeftModel wrapper from losing them.
        if not prediction_loss_only and hasattr(model, "generation_config"):
            forced = getattr(model.generation_config, "forced_decoder_ids", None)
            if forced is not None:
                self._gen_kwargs = getattr(self, "_gen_kwargs", {})
                self._gen_kwargs["forced_decoder_ids"] = forced

        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
