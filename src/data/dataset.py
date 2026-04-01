"""
Streaming ASR dataset for IndicVoices + Whisper.
Simple, readable, and compatible with Seq2SeqTrainer.
"""

from __future__ import annotations

import io
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from transformers import WhisperProcessor

try:
    from src.data.augment import AudioAugmentor
except Exception:
    AudioAugmentor = None


SAMPLE_RATE = 16_000
MAX_LABEL_LENGTH = 448


@dataclass
class DataConfig:
    dataset_name: str
    language: str
    split_train: str = "train"
    split_val: str = "valid"
    audio_column: str = "audio"
    text_column: str = "normalized"
    domain_column: str = "task_name"
    sampling_rate: int = SAMPLE_RATE
    feature_size: int = 128
    min_duration_secs: float = 0.1
    max_duration_secs: float = 30.0
    shuffle_buffer_size: int = 5000
    prefetch_size: int = 8
    seed: int = 42
    augmentation: dict = field(default_factory=dict)
    normalization: str = "basic"

    @classmethod
    def from_omega(cls, cfg) -> "DataConfig":
        import dataclasses
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(cfg.data, resolve=True)
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in valid_fields})


def get_hf_datasets_cache_dir() -> str:
    base = os.environ.get("HF_HOME", str(Path.home() / "hf_cache"))
    cache_dir = os.environ.get("HF_DATASETS_CACHE", f"{base}/datasets")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_split_name(config: DataConfig, split_name: str) -> str:
    if split_name == "train":
        return config.split_train
    if split_name in {"val", "valid", "validation", "eval"}:
        return config.split_val
    return split_name


def clean_text(text: Any) -> Optional[str]:
    if text is None:
        return None

    text = str(text).strip()
    if not text:
        return None

    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.strip()

    return text if len(text) >= 2 else None


def normalize_text(text: str, mode: str = "basic") -> str:
    if mode == "basic":
        return re.sub(r"\s+", " ", text.strip())
    return text.strip()


def get_transcript(sample: dict[str, Any], text_column: str, normalization: str) -> Optional[str]:
    text = clean_text(sample.get(text_column))
    if text is None:
        return None
    return normalize_text(text, normalization)


def get_domain(sample: dict[str, Any], domain_column: str) -> str:
    value = sample.get(domain_column)
    if value is None:
        return "unknown_domain"

    value = str(value).strip()
    return value if value else "unknown_domain"


def open_stream(config: DataConfig, split_name: str, token: Optional[str] = None):
    resolved_split = resolve_split_name(config, split_name)

    ds = load_dataset(
        config.dataset_name,
        config.language,
        split=resolved_split,
        streaming=True,
        token=token,
        cache_dir="/home/harsh/hf_cache/datasets",
    )

    try:
        ds = ds.decode(False)
    except Exception:
        pass

    if resolved_split == config.split_train:
        ds = ds.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer_size)

    return ds


def load_audio_from_path(audio_path: str, target_sr: int) -> Optional[tuple[np.ndarray, int]]:
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if audio is None or len(audio) == 0:
            return None
        audio = np.asarray(audio, dtype=np.float32)
        return audio, sr
    except Exception:
        return None


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int) -> Optional[tuple[np.ndarray, int]]:
    try:
        with io.BytesIO(audio_bytes) as buffer:
            audio, sr = sf.read(buffer, dtype="float32")

        if audio is None or len(audio) == 0:
            return None

        audio = np.asarray(audio, dtype=np.float32)

        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        audio = np.asarray(audio, dtype=np.float32)
        return audio, target_sr
    except Exception:
        pass

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        return load_audio_from_path(temp_path, target_sr)
    except Exception:
        return None
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def extract_audio_field(sample: dict[str, Any], preferred_key: str) -> Optional[Any]:
    candidate_keys = [
        preferred_key,
        "audio",
        "audio_path",
        "audio_filepath",
        "audio_file",
        "audio_filename",
        "file_path",
        "filepath",
        "path",
        "file",
        "wav_path",
        "mp3_path",
        "flac_path",
    ]

    for key in candidate_keys:
        value = sample.get(key)
        if value is not None:
            return value

    return None


def load_audio(audio_obj: Any, target_sr: int) -> Optional[tuple[np.ndarray, int]]:
    if audio_obj is None:
        return None

    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            try:
                audio = np.asarray(audio_obj["array"], dtype=np.float32)
                sr = int(audio_obj.get("sampling_rate", target_sr))

                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1)

                if sr != target_sr:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

                audio = np.asarray(audio, dtype=np.float32)

                if audio is None or len(audio) == 0:
                    return None

                return audio, target_sr
            except Exception:
                pass

        if audio_obj.get("path"):
            loaded = load_audio_from_path(audio_obj["path"], target_sr)
            if loaded is not None:
                return loaded

        if audio_obj.get("bytes"):
            return load_audio_from_bytes(audio_obj["bytes"], target_sr)

        return None

    if isinstance(audio_obj, str):
        return load_audio_from_path(audio_obj, target_sr)

    try:
        decoded = audio_obj.get_all_samples()
        audio = decoded.data.squeeze(0).numpy().astype("float32")
        if audio is None or len(audio) == 0:
            return None
        return np.asarray(audio, dtype=np.float32), target_sr
    except Exception:
        return None


def preprocess_sample(
    sample: dict[str, Any],
    config: DataConfig,
    processor: WhisperProcessor,
    augmentor: Optional[Any] = None,
    augment: bool = False,
) -> Optional[dict[str, Any]]:
    transcript = get_transcript(sample, config.text_column, config.normalization)
    if transcript is None:
        return None

    audio_obj = extract_audio_field(sample, config.audio_column)
    if audio_obj is None:
        return None

    loaded = load_audio(audio_obj, config.sampling_rate)
    if loaded is None:
        return None

    audio, sr = loaded
    duration = len(audio) / sr

    if duration < config.min_duration_secs or duration > config.max_duration_secs:
        return None

    if augment and augmentor is not None:
        try:
            audio = augmentor.augment_waveform(audio)
        except Exception:
            pass

    try:
        input_features = processor.feature_extractor(
            audio,
            sampling_rate=sr,
        ).input_features[0]
    except Exception:
        return None

    if augment and augmentor is not None:
        try:
            input_features = augmentor.augment_features(input_features)
        except Exception:
            pass

    try:
        labels = processor.tokenizer(transcript).input_ids
    except Exception:
        return None

    if len(labels) > MAX_LABEL_LENGTH:
        return None

    return {
        "input_features": input_features,
        "labels": labels,
        "text": transcript,
        "duration": float(duration),
        "domain": get_domain(sample, config.domain_column),
        "language": config.language,
    }


class StreamingASRDataset(IterableDataset):
    def __init__(
        self,
        config: DataConfig,
        processor: WhisperProcessor,
        split_name: str,
        token: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.processor = processor
        self.split_name = split_name
        self.token = token
        self.is_train = resolve_split_name(config, split_name) == config.split_train
        self.augmentor = self._build_augmentor()

    def _build_augmentor(self) -> Optional[Any]:
        if AudioAugmentor is None:
            return None

        aug_cfg = self.config.augmentation or {}
        if not aug_cfg.get("enabled", False):
            return None

        return AudioAugmentor(
            noise_prob=aug_cfg.get("noise_prob", 0.3),
            speed_prob=aug_cfg.get("speed_prob", 0.3),
            speed_range=tuple(aug_cfg.get("speed_range", [0.9, 1.1])),
            do_spec_augment=aug_cfg.get("spec_augment", True),
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = self.config.seed + worker_id

        random.seed(seed)

        stream = open_stream(self.config, self.split_name, token=self.token)
        buffer: list[dict[str, Any]] = []

        for sample in stream:
            processed = preprocess_sample(
                sample=sample,
                config=self.config,
                processor=self.processor,
                augmentor=self.augmentor,
                augment=self.is_train,
            )

            if processed is None:
                continue

            buffer.append(processed)

            if len(buffer) >= self.config.prefetch_size:
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()

        if buffer:
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()


def build_train_dataset(
    config: DataConfig,
    processor: WhisperProcessor,
    token: Optional[str] = None,
) -> StreamingASRDataset:
    return StreamingASRDataset(
        config=config,
        processor=processor,
        split_name=config.split_train,
        token=token,
    )


def build_eval_dataset(
    config: DataConfig,
    processor: WhisperProcessor,
    token: Optional[str] = None,
) -> Optional[StreamingASRDataset]:
    try:
        _ = open_stream(config, config.split_val, token=token)
        return StreamingASRDataset(
            config=config,
            processor=processor,
            split_name=config.split_val,
            token=token,
        )
    except Exception:
        return None