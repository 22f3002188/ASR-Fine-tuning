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
from datasets import load_dataset, IterableDataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from src.data.augment import AudioAugmentor


SAMPLE_RATE = 16_000


# ── Audio decoding helper (Section 4) ─────────────────────────────────────────

def decode_audio(af) -> Optional[np.ndarray]:
    """
    Decode an IndicVoices AudioDecoder (torchcodec backend) to numpy float32.
    Returns shape (num_samples,) at 16kHz, or None on failure.

    After cast_column, `af` is already a dict {"array": ..., "sampling_rate": ...}.
    This function handles both the cast_column output and the raw AudioDecoder
    as a fallback, making it robust to dataset version differences.
    """
    if af is None:
        return None
    try:
        # Primary path: cast_column already decoded to dict
        if isinstance(af, dict):
            arr = np.array(af["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)   # stereo → mono
            return arr
        # Fallback: raw torchcodec AudioDecoder
        decoded = af.get_all_samples()
        # data: Tensor (1, num_samples) — already mono, float32, 16kHz
        return decoded.data.squeeze(0).numpy().astype(np.float32)
    except Exception:
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
    feature_size: int = 128      # mel bins: 128 for large-v3, 80 for small/medium
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
        # Drop any keys not present in DataConfig to handle config/code drift
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)


# ── Dataset ───────────────────────────────────────────────────────────────────

class StreamingASRDataset:
    """
    Wraps IndicVoices as a lazy streaming IterableDataset for Whisper finetuning.

    Usage:
        ds = StreamingASRDataset(config, feature_extractor, tokenizer)
        ds.load()
        train_ds = ds.get_split("train", shuffle=True)
        val_ds   = ds.get_split("val",   shuffle=False)
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

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "StreamingASRDataset":
        """
        Load both splits in streaming mode and cast audio column.
        cast_column(Audio(sampling_rate=16000)) triggers torchcodec decoding
        on-the-fly — no audio is loaded until iteration begins.
        trust_remote_code=True is required for IndicVoices' custom loader.
        """
        raw = load_dataset(
            self.config.dataset_name,
            self.config.language,
            streaming=True,
            trust_remote_code=True,
            token=True,
        )
        # Cast audio column so HF decodes AudioDecoder → float32 array at 16kHz
        self._raw = {
            split: ds.cast_column(self.config.audio_column, Audio(sampling_rate=SAMPLE_RATE))
            for split, ds in raw.items()
        }
        return self

    def get_split(self, split: str, shuffle: bool = False) -> IterableDataset:
        """
        Return a fully processed IterableDataset for the requested split.

        Args:
            split   : "train" or "val"
            shuffle : True for train, False for val/test
        """
        assert self._raw is not None, "Call .load() first."

        split_key = self.config.split_train if split == "train" else self.config.split_val
        ds: IterableDataset = self._raw[split_key]

        if shuffle:
            ds = ds.shuffle(seed=self.config.seed, buffer_size=self.config.buffer_size)

        is_train = (split == "train")
        ds = ds.map(
            lambda sample: self._prepare_sample(sample, augment=is_train),
            remove_columns=self._columns_to_remove(ds),
        )
        # Drop samples that failed decoding or duration check
        ds = ds.filter(lambda x: x["input_features"] is not None)
        return ds

    def get_split_as_list(
        self,
        split: str,
        n_samples: int,
        shuffle: bool = False,
    ) -> list[dict]:
        """
        Stream exactly n_samples valid samples from a split into a list.
        Uses a 3x oversampling buffer to account for invalid/filtered rows,
        matching the approach from Section 5 of the dataset exploration notebook.

        Useful for: prepare_data.py validation, quick eval, collator testing.
        NOT for training — use get_split() (lazy IterableDataset) instead.

        Args:
            split     : "train" or "val"
            n_samples : exact number of valid samples to collect
            shuffle   : whether to shuffle the stream before sampling
        """
        assert self._raw is not None, "Call .load() first."

        split_key = self.config.split_train if split == "train" else self.config.split_val
        stream: IterableDataset = self._raw[split_key]

        if shuffle:
            stream = stream.shuffle(seed=self.config.seed, buffer_size=self.config.buffer_size)

        # 3x budget to account for filtered rows (corrupt audio, bad duration, empty text)
        budget = n_samples * 3
        is_train = (split == "train")

        processed = []
        for row in itertools.islice(stream, budget):
            sample = self._prepare_sample(row, augment=is_train)
            if sample["input_features"] is None:
                continue
            processed.append(sample)
            if len(processed) >= n_samples:
                break

        return processed

    # ── Core transform ────────────────────────────────────────────────────────

    def _prepare_sample(self, sample: dict, augment: bool = False) -> dict:
        """
        Decode audio, validate duration, extract log-mel features, tokenize text.
        Returns {"input_features": None, "labels": None} for invalid samples.

        After cast_column, sample[audio_column] is:
            {"array": np.ndarray, "sampling_rate": 16000, "path": str | None}
        """
        # ── Decode ────────────────────────────────────────────────────────────
        audio_array = decode_audio(sample.get(self.config.audio_column))
        if audio_array is None or len(audio_array) == 0:
            return {"input_features": None, "labels": None}

        # ── Duration filter ───────────────────────────────────────────────────
        duration = len(audio_array) / self.config.sampling_rate
        if not (self.config.min_duration_secs <= duration <= self.config.max_duration_secs):
            return {"input_features": None, "labels": None}

        # ── Text ──────────────────────────────────────────────────────────────
        text = str(sample.get(self.config.text_column, "")).strip()
        if not text:
            return {"input_features": None, "labels": None}

        # ── Waveform augmentation (train only) ────────────────────────────────
        if augment and self._train_augmentor is not None:
            audio_array = self._train_augmentor.augment_waveform(audio_array)

        # ── Log-mel feature extraction ────────────────────────────────────────
        input_features = self.feature_extractor(
            audio_array,
            sampling_rate=self.config.sampling_rate,
        ).input_features[0]   # (80, 3000) — already a numpy array

        # ── SpecAugment on features (train only) ──────────────────────────────
        if augment and self._train_augmentor is not None:
            input_features = self._train_augmentor.augment_features(input_features)

        # ── Text normalisation + tokenisation ─────────────────────────────────
        if self.config.normalization == "basic":
            text = _basic_normalize(text)

        labels = self.tokenizer(text).input_ids

        return {"input_features": input_features, "labels": labels}

    def _columns_to_remove(self, ds: IterableDataset) -> list[str]:
        keep = {"input_features", "labels"}
        return [c for c in ds.column_names if c not in keep]


# ── Text normalisation ─────────────────────────────────────────────────────────

def _basic_normalize(text: str) -> str:
    """
    Strip whitespace and collapse internal spaces.
    Gurmukhi (Punjabi script) is unicase — do NOT lowercase.
    Extend here for numeral normalisation, punctuation stripping, etc.
    """
    import re
    return re.sub(r"\s+", " ", text.strip())