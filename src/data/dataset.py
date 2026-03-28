"""
Streaming ASR dataset for IndicVoices + Whisper.

Key finding from dataset exploration (Section 4-5):
  - audio_filepath is a torchcodec AudioDecoder, NOT a filepath string.
  - Use stream.cast_column(col, Audio(sampling_rate=16000)) to let HF decode
    it automatically. After casting, each row[audio_col] is a dict:
      {"array": np.ndarray float32, "sampling_rate": 16000, "path": ...}
  - decode_audio() calls .get_all_samples() on the raw decoder as a fallback,
    but cast_column is the primary path.
"""

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, IterableDataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer

import io
import soundfile as sf
import librosa

from src.data.augment import AudioAugmentor


SAMPLE_RATE = 16_000


# ── Audio decoding helper ─────────────────────────────────────────────────────

def decode_audio(af) -> Optional[np.ndarray]:
    """
    Decode audio safely without torchcodec.
    
    Supports:
    - HF streaming audio dict
    - raw bytes
    - filepath string
    """
    if af is None:
        return None

    try:
        # Already decoded by dataset / dict format
        if isinstance(af, dict):
            if "array" in af:
                arr = np.asarray(af["array"], dtype=np.float32)

                if arr.ndim > 1:
                    arr = arr.mean(axis=0)

                if af.get("sampling_rate", SAMPLE_RATE) != SAMPLE_RATE:
                    arr = librosa.resample(
                        arr,
                        orig_sr=af["sampling_rate"],
                        target_sr=SAMPLE_RATE
                    )

                return arr.astype(np.float32)

            # filepath-style dict
            if "path" in af and af["path"]:
                audio, sr = sf.read(af["path"])

                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                if sr != SAMPLE_RATE:
                    audio = librosa.resample(
                        audio,
                        orig_sr=sr,
                        target_sr=SAMPLE_RATE
                    )

                return audio.astype(np.float32)

        # plain filepath string
        if isinstance(af, str):
            audio, sr = sf.read(af)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != SAMPLE_RATE:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=SAMPLE_RATE
                )

            return audio.astype(np.float32)

        # raw bytes stream
        if isinstance(af, bytes):
            audio, sr = sf.read(io.BytesIO(af))

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sr != SAMPLE_RATE:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=SAMPLE_RATE
                )

            return audio.astype(np.float32)

    except Exception as e:
        print(f"Audio decode failed: {e}")
        return None

    return None


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    dataset_name: str
    language: str
    split_train: str = "train"
    split_val: str = "valid"
    audio_column: str = "audio_filepath"
    text_column: str = "normalized"
    sampling_rate: int = SAMPLE_RATE
    feature_size: int = 128
    max_duration_secs: float = 30.0
    min_duration_secs: float = 0.1
    buffer_size: int = 5000
    seed: int = 42
    augmentation: dict = field(default_factory=dict)
    normalization: str = "basic"

    @classmethod
    def from_omega(cls, cfg) -> "DataConfig":
        import dataclasses
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(cfg.data, resolve=True)
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


# ── Dataset ───────────────────────────────────────────────────────────────────

class StreamingASRDataset:
    """
    Wraps IndicVoices as a lazy streaming IterableDataset for Whisper finetuning.
    """

    def __init__(
        self,
        config: DataConfig,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self._raw = None

        aug_cfg = config.augmentation or {}
        self._train_augmentor = AudioAugmentor(
            noise_prob=aug_cfg.get("noise_prob", 0.3),
            speed_prob=aug_cfg.get("speed_prob", 0.3),
            speed_range=tuple(aug_cfg.get("speed_range", [0.9, 1.1])),
            do_spec_augment=aug_cfg.get("spec_augment", True),
        ) if aug_cfg.get("enabled", False) else None

    def load(self) -> "StreamingASRDataset":
        raw = load_dataset(
            self.config.dataset_name,
            self.config.language,
            streaming=True,
            token=True,
        )
    
        # Disable automatic feature decoding
        self._raw = {
            split: ds.with_format(None)
            for split, ds in raw.items()
        }
    
        return self

    def get_split(self, split: str, shuffle: bool = False) -> IterableDataset:
        assert self._raw is not None, "Call .load() first."

        split_key = (
            self.config.split_train
            if split == "train"
            else self.config.split_val
        )

        ds: IterableDataset = self._raw[split_key]

        if shuffle:
            ds = ds.shuffle(
                seed=self.config.seed,
                buffer_size=self.config.buffer_size
            )

        is_train = split == "train"

        ds = ds.map(
            lambda sample: self._prepare_sample(sample, augment=is_train),
            remove_columns=self._columns_to_remove(ds),
        )

        ds = ds.filter(lambda x: x["input_features"] is not None)

        return ds

    def get_split_as_list(
        self,
        split: str,
        n_samples: int,
        shuffle: bool = False,
    ) -> list[dict]:
        """
        Buffered sample collection using incoming branch batching logic.
        """
        assert self._raw is not None, "Call .load() first."

        split_key = (
            self.config.split_train
            if split == "train"
            else self.config.split_val
        )

        stream: IterableDataset = self._raw[split_key]

        if shuffle:
            stream = stream.shuffle(
                seed=self.config.seed,
                buffer_size=self.config.buffer_size
            )

        budget = n_samples * 3
        is_train = split == "train"

        processed = []
        buffer = []

        for row in itertools.islice(stream, budget):
            sample = self._prepare_sample(row, augment=is_train)

            if sample["input_features"] is None:
                continue

            buffer.append(sample)

            # borrowed batching logic
            if len(buffer) >= 8:
                processed.extend(buffer)
                buffer = []

            if len(processed) >= n_samples:
                break

        if buffer:
            processed.extend(buffer)

        return processed[:n_samples]

    def _prepare_sample(self, sample: dict, augment: bool = False) -> dict:
        audio_array = decode_audio(sample.get(self.config.audio_column))

        if audio_array is None or len(audio_array) == 0:
            return {"input_features": None, "labels": None}

        duration = len(audio_array) / self.config.sampling_rate

        if not (
            self.config.min_duration_secs
            <= duration
            <= self.config.max_duration_secs
        ):
            return {"input_features": None, "labels": None}

        text = str(sample.get(self.config.text_column, "")).strip()

        if not text:
            return {"input_features": None, "labels": None}

        if augment and self._train_augmentor is not None:
            audio_array = self._train_augmentor.augment_waveform(audio_array)

        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.config.sampling_rate,
        ).input_features[0]

        if augment and self._train_augmentor is not None:
            input_features = self._train_augmentor.augment_features(input_features)

        if self.config.normalization == "basic":
            text = _basic_normalize(text)

        labels = self.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels
        }

    def _columns_to_remove(self, ds: IterableDataset) -> list[str]:
        keep = {"input_features", "labels"}
        return [c for c in ds.column_names if c not in keep]


def _basic_normalize(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.strip())