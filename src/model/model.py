"""
Whisper model loader for vanilla (full) finetuning.

No LoRA, no adapter wrapping, no layer freezing.
The full model is trained end-to-end with the optimizer
updating all parameters.
"""

from __future__ import annotations

import torch
from transformers import WhisperForConditionalGeneration


def load_model(cfg):
    """
    Load Whisper model with dtype chosen from config/runtime.
    """

    model_name = cfg.model.name

    # Read preferred dtype from config if present.
    # Supported: "float32", "float16", "bfloat16", "auto"
    requested_dtype = str(getattr(cfg.model, "torch_dtype", "auto")).lower()

    if requested_dtype == "float32":
        torch_dtype = torch.float32
    elif requested_dtype == "float16":
        torch_dtype = torch.float16
    elif requested_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )

    # Let labels drive decoder inputs during training.
    # Avoid forcing decoder ids during training forward.
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    # Suppress timestamps during normal ASR generation unless explicitly needed
    if hasattr(model.generation_config, "suppress_tokens"):
        pass

    # Recommended during training for memory
    model.config.use_cache = False

    return model