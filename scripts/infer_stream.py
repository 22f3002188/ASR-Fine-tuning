"""
Inference from streaming IndicVoices using the project's existing dataset pipeline.

This avoids direct HF audio decoding inside this script, so it works with the
same preprocessing path that training already uses.

Run:
    CUDA_VISIBLE_DEVICES=1 python scripts/infer_stream.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.data.dataset import DataConfig, build_train_dataset


MODEL_PATH = "checkpoints/final_model"


def load_model_and_processor(cfg):
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)

    # Force Punjabi transcription prefix tokens
    processor.tokenizer.set_prefix_tokens(
        language="pa",
        task="transcribe",
    )

    print("Loading base model...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model.name,
        token=hf_token,
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    return model, processor, device


def get_one_processed_sample(cfg, processor):
    """
    Reuse the project's dataset pipeline so we avoid direct HF audio decoding
    in this inference script.
    """
    hf_token = os.environ.get("HF_TOKEN")
    data_cfg = DataConfig.from_omega(cfg)

    ds = build_train_dataset(
        config=data_cfg,
        processor=processor,
        token=hf_token,
    )

    for sample in ds:
        if sample is not None:
            return sample

    return None


def run_inference(model, processor, sample, device):
    input_features = torch.tensor(
        sample["input_features"],
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    # Whisper warning workaround
    attention_mask = torch.ones(
        (input_features.shape[0], input_features.shape[-1]),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            language="pa",
            task="transcribe",
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_new_tokens=64,
            do_sample=False,
        )

    prediction = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    reference = processor.tokenizer.decode(
        sample["labels"],
        skip_special_tokens=True,
    )

    return prediction, reference


def main():
    cfg = load_config()

    print("\n" + "=" * 50)
    print(" Streaming inference")
    print(f" Model   : {cfg.model.name}")
    print(f" Dataset : {cfg.data.dataset_name} / {cfg.data.language}")
    print("=" * 50 + "\n")

    model, processor, device = load_model_and_processor(cfg)

    print("Fetching one processed streaming sample...")
    sample = get_one_processed_sample(cfg, processor)

    if sample is None:
        print("No valid sample found.")
        return

    print("Running inference...\n")
    prediction, reference = run_inference(model, processor, sample, device)

    print("=" * 60)
    print("REFERENCE:")
    print(reference)
    print("\nPREDICTION:")
    print(prediction)
    print("=" * 60)


if __name__ == "__main__":
    main()