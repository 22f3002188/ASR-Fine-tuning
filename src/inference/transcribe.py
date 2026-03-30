"""
Single-file and batch transcription using the finetuned Whisper model.

Two backends supported:
  1. HuggingFace transformers (default) — loads from checkpoints/final_model/
  2. faster-whisper (CTranslate2)       — faster CPU/GPU inference, requires
     conversion via ct2-whisper-converter first:

     ct2-whisper-converter \
         --model checkpoints/final_model \
         --output_dir checkpoints/final_model_ct2 \
         --quantization float16
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


# ── HuggingFace backend ───────────────────────────────────────────────────────

class HFTranscriber:
    """
    Transcribe audio using the HuggingFace transformers backend.
    Loads the finetuned model directly from the saved checkpoint directory.
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "punjabi",
        task: str = "transcribe",
        device: Optional[str] = None,
    ):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.task     = task

        print(f"Loading model from {model_dir} on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        sampling_rate: int = 16_000,
        max_new_tokens: int = 444,
    ) -> str:
        """
        Transcribe a single audio clip.

        Args:
            audio         : numpy float32 array or path to audio file
            sampling_rate : sample rate of the audio array (ignored if path)
            max_new_tokens: max tokens to generate

        Returns:
            Transcribed string.
        """
        if isinstance(audio, (str, Path)):
            audio, sampling_rate = _load_audio(audio)

        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features.to(self.device)

        if self.device == "cuda":
            inputs = inputs.half()

        with torch.no_grad():
            pred_ids = self.model.generate(
                inputs,
                language=self.language,
                task=self.task,
                max_new_tokens=max_new_tokens,
            )

        return self.processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    def transcribe_batch(
        self,
        audio_list: list[Union[np.ndarray, str, Path]],
        sampling_rate: int = 16_000,
        max_new_tokens: int = 444,
    ) -> list[str]:
        """Transcribe a list of audio clips."""
        return [self.transcribe(a, sampling_rate, max_new_tokens) for a in audio_list]


# ── faster-whisper backend ────────────────────────────────────────────────────

class FasterWhisperTranscriber:
    """
    Transcribe using the faster-whisper (CTranslate2) backend.
    Significantly faster than the HF backend for CPU and GPU inference.
    Requires model conversion first — see module docstring.
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "pa",    # ISO 639-1 code for Punjabi
        device: str = "auto",
        compute_type: str = "float16",
    ):
        from faster_whisper import WhisperModel
        self.language = language
        print(f"Loading faster-whisper model from {model_dir}...")
        self.model = WhisperModel(model_dir, device=device, compute_type=compute_type)

    def transcribe(self, audio: Union[np.ndarray, str, Path]) -> str:
        """Transcribe a single audio clip or file path."""
        audio_input = str(audio) if isinstance(audio, Path) else audio
        segments, _ = self.model.transcribe(
            audio_input,
            language=self.language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments)


# ── Audio loading helper ──────────────────────────────────────────────────────

def _load_audio(path: Union[str, Path], target_sr: int = 16_000):
    """Load an audio file to a float32 numpy array at target_sr."""
    import librosa
    arr, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return arr, sr