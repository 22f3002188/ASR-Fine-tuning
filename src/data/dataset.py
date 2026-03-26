"""HuggingFace streaming Dataset wrapper for ASR finetuning."""

from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset, IterableDataset, IterableDatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer


@dataclass
class DataConfig:
    dataset_name: str           # e.g. "mozilla-foundation/common_voice_17_0"
    language: str               # e.g. "kn" for Kannada
    split_train: str = "train"
    split_val: str = "validation"
    split_test: str = "test"
    audio_column: str = "audio"
    text_column: str = "sentence"
    sampling_rate: int = 16_000
    max_duration_secs: float = 30.0
    min_duration_secs: float = 0.1
    buffer_size: int = 500      # shuffle buffer — keep low to avoid OOM
    seed: int = 42


class StreamingASRDataset:
    """
    Wraps a HuggingFace streaming IterableDataset for Whisper finetuning.

    Usage:
        ds = StreamingASRDataset(config, feature_extractor, tokenizer)
        train_ds = ds.get_split("train")   # returns an IterableDataset
    """

    def __init__(
        self,
        config: DataConfig,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: WhisperTokenizer,
        trust_remote_code: bool = False,
    ):
        self.config = config
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.trust_remote_code = trust_remote_code
        self._raw: Optional[IterableDatasetDict] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "StreamingASRDataset":
        """Load all splits in streaming mode — no data is downloaded yet."""
        self._raw = load_dataset(
            self.config.dataset_name,
            self.config.language,
            streaming=True,
            trust_remote_code=self.trust_remote_code,
        )
        return self

    def get_split(self, split: str, shuffle: bool = False) -> IterableDataset:
        """
        Return a processed, optionally shuffled IterableDataset for a split.
        All transforms are lazy — they run sample-by-sample during training.
        """
        assert self._raw is not None, "Call .load() first."

        split_key = getattr(self.config, f"split_{split}")
        ds: IterableDataset = self._raw[split_key]

        if shuffle:
            # buffer_size controls memory pressure — tune based on RAM
            ds = ds.shuffle(
                seed=self.config.seed,
                buffer_size=self.config.buffer_size,
            )

        ds = ds.filter(self._duration_filter)
        ds = ds.map(self._prepare_sample, remove_columns=self._columns_to_remove(ds))

        return ds

    # ------------------------------------------------------------------
    # Transforms (all lazy — no data read until iteration)
    # ------------------------------------------------------------------

    def _duration_filter(self, sample: dict) -> bool:
        """Drop samples that are too short or exceed Whisper's 30-second window."""
        duration = len(sample[self.config.audio_column]["array"]) / self.config.sampling_rate
        return self.config.min_duration_secs <= duration <= self.config.max_duration_secs

    def _prepare_sample(self, sample: dict) -> dict:
        """
        Convert raw audio + transcript into Whisper model inputs.
        - input_features : log-mel spectrogram  [80, 3000]
        - labels          : tokenized transcript  [seq_len]
        """
        audio = sample[self.config.audio_column]

        # Resample if the dataset's native rate differs
        if audio["sampling_rate"] != self.config.sampling_rate:
            import librosa
            audio["array"] = librosa.resample(
                audio["array"],
                orig_sr=audio["sampling_rate"],
                target_sr=self.config.sampling_rate,
            )

        input_features = self.feature_extractor(
            audio["array"],
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
        ).input_features[0]

        labels = self.tokenizer(sample[self.config.text_column]).input_ids

        return {"input_features": input_features, "labels": labels}

    def _columns_to_remove(self, ds: IterableDataset) -> list[str]:
        """Drop all columns except the ones the model needs."""
        keep = {"input_features", "labels"}
        return [c for c in ds.column_names if c not in keep]
