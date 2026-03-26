from __future__ import annotations

from typing import Optional

from huggingface_hub import login
from transformers import WhisperProcessor


def hf_login_if_needed(token: Optional[str]) -> None:
    if not token:
        raise ValueError(
            "HF_TOKEN is not set.\n"
            "Run this first:\n"
            "export HF_TOKEN='your_huggingface_token'"
        )
    login(token=token, add_to_git_credential=False)
    print("Hugging Face login successful.")


def build_processor(model_name: str) -> WhisperProcessor:
    return WhisperProcessor.from_pretrained(model_name)