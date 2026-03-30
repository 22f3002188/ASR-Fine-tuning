"""
End-to-end inference pipeline.

Accepts a file path or numpy array, applies optional VAD to split long audio
into chunks, transcribes each chunk, and concatenates the results.

For files shorter than 30s, chunking is skipped — the audio is passed directly.
For longer files, silero VAD is used to find speech boundaries before chunking.
"""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.inference.transcribe import HFTranscriber, FasterWhisperTranscriber
from src.inference.postprocess import postprocess


SAMPLE_RATE = 16_000
MAX_DURATION_SECS = 30.0


class ASRPipeline:
    """
    End-to-end ASR pipeline: load audio → VAD chunk → transcribe → postprocess.

    Args:
        model_dir   : path to finetuned model (HF format)
        language    : language name for generation forcing
        use_faster  : use faster-whisper backend instead of HF transformers
        device      : "cuda", "cpu", or None (auto-detect)
    """

    def __init__(
        self,
        model_dir: str,
        language: str = "punjabi",
        use_faster: bool = False,
        device: Optional[str] = None,
    ):
        if use_faster:
            self.transcriber = FasterWhisperTranscriber(
                model_dir=model_dir,
                language="pa",
                device=device or "auto",
            )
        else:
            self.transcriber = HFTranscriber(
                model_dir=model_dir,
                language=language,
                device=device,
            )

    def __call__(
        self,
        audio: Union[str, Path, np.ndarray],
        sampling_rate: int = SAMPLE_RATE,
        remove_punctuation: bool = False,
        strip_filler_words: bool = False,
    ) -> str:
        """
        Transcribe an audio clip end-to-end.

        Args:
            audio             : file path or float32 numpy array
            sampling_rate     : sample rate (ignored if path given)
            remove_punctuation: strip punctuation from output
            strip_filler_words: remove filler words from output

        Returns:
            Final transcribed and post-processed string.
        """
        # Load if path
        if isinstance(audio, (str, Path)):
            audio, sampling_rate = _load_audio(audio)

        # Route to chunked or direct transcription
        duration = len(audio) / sampling_rate
        if duration <= MAX_DURATION_SECS:
            text = self.transcriber.transcribe(audio, sampling_rate=sampling_rate)
        else:
            text = self._transcribe_long(audio, sampling_rate)

        # Post-process
        return postprocess(
            text,
            remove_punctuation=remove_punctuation,
            strip_filler_words=strip_filler_words,
        )

    def _transcribe_long(self, audio: np.ndarray, sampling_rate: int) -> str:
        """
        Split audio longer than 30s into VAD-aligned chunks and transcribe each.
        Falls back to fixed 25s chunks if VAD is unavailable.
        """
        chunks = _vad_chunk(audio, sampling_rate)
        texts  = []
        for chunk in chunks:
            text = self.transcriber.transcribe(chunk, sampling_rate=sampling_rate)
            texts.append(text.strip())
        return " ".join(t for t in texts if t)


# ── VAD chunking ──────────────────────────────────────────────────────────────

def _vad_chunk(
    audio: np.ndarray,
    sampling_rate: int,
    chunk_secs: float = 25.0,
) -> list[np.ndarray]:
    """
    Split audio into chunks using silero VAD if available.
    Falls back to fixed-length chunking if torch.hub load fails.

    Args:
        audio       : float32 audio array
        sampling_rate: sample rate (must be 16000 for silero)
        chunk_secs  : max chunk length in seconds

    Returns:
        List of numpy arrays, each at most chunk_secs long.
    """
    try:
        return _silero_vad_chunk(audio, sampling_rate, chunk_secs)
    except Exception:
        return _fixed_chunk(audio, sampling_rate, chunk_secs)


def _silero_vad_chunk(
    audio: np.ndarray,
    sampling_rate: int,
    max_chunk_secs: float,
) -> list[np.ndarray]:
    """Use silero VAD to find speech segment boundaries."""
    import torch

    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    tensor = torch.from_numpy(audio)
    timestamps = get_speech_timestamps(
        tensor, vad_model, sampling_rate=sampling_rate, return_seconds=False
    )

    if not timestamps:
        return [audio]

    # Merge timestamps into chunks up to max_chunk_secs
    chunks = []
    max_samples = int(max_chunk_secs * sampling_rate)
    current_start = timestamps[0]["start"]
    current_end   = timestamps[0]["end"]

    for ts in timestamps[1:]:
        if ts["end"] - current_start > max_samples:
            chunks.append(audio[current_start:current_end])
            current_start = ts["start"]
        current_end = ts["end"]

    chunks.append(audio[current_start:current_end])
    return [c for c in chunks if len(c) > 0]


def _fixed_chunk(
    audio: np.ndarray,
    sampling_rate: int,
    chunk_secs: float,
) -> list[np.ndarray]:
    """Simple fixed-length chunking without VAD."""
    chunk_samples = int(chunk_secs * sampling_rate)
    return [
        audio[i : i + chunk_samples]
        for i in range(0, len(audio), chunk_samples)
        if len(audio[i : i + chunk_samples]) > 0
    ]


def _load_audio(path: Union[str, Path], target_sr: int = SAMPLE_RATE):
    import librosa
    arr, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return arr, sr