"""
Whisper model loader (full fine-tuning).
"""

from __future__ import annotations

import os
import torch
from transformers import WhisperForConditionalGeneration


def load_model(cfg):
    model_name = cfg.model.name

    torch_dtype = _get_dtype(cfg)

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        cache_dir=os.environ.get("HF_HUB_CACHE"),
    )

    # Training settings
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # important for training

    return model


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def _get_dtype(cfg):
    dtype = str(getattr(cfg.model, "torch_dtype", "auto")).lower()

    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16

    return "auto"