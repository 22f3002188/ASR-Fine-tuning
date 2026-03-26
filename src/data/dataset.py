from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
import io
import os
import random
import re
import tempfile

import librosa
import soundfile as sf
from datasets import load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import WhisperProcessor


# ============================================================
# DEBUG SETTINGS
# ============================================================

DEBUG_PROGRESS_EVERY = 50
DEBUG_PRINT_RAW_DOMAINS = True
DEBUG_RAW_DOMAIN_PRINT_LIMIT = 20
DEBUG_PRINT_SKIP_REASONS = True
DEBUG_SKIP_REASON_PRINT_LIMIT = 30
DEBUG_PRINT_SUCCESS_SAMPLES = True
DEBUG_SUCCESS_PRINT_LIMIT = 20
DEBUG_PRINT_SAMPLE_KEYS_FOR_MISSING_AUDIO = True
DEBUG_SAMPLE_KEYS_PRINT_LIMIT = 10


# ============================================================
# CONFIG
# ============================================================

@dataclass
class DataConfig:
    dataset_name: str
    language: str
    split_train: str = "train"
    split_val_candidates: list[str] = field(default_factory=lambda: ["validation", "valid", "test"])
    audio_column: str = "audio"
    text_candidates: list[str] = field(default_factory=lambda: ["normalized", "text", "verbatim"])
    domain_column: str = "task_name"
    sampling_rate: int = 16000
    min_duration_secs: float = 1.0
    max_duration_secs: float = 30.0
    shuffle_buffer_size: int = 500
    prefetch_size: int = 8
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DataConfig":
        return cls(**raw)


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text: Any) -> Optional[str]:
    if text is None:
        return None

    text = str(text).strip()
    if not text:
        return None

    text = re.sub(r"\s+", " ", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.strip()

    if len(text) < 2:
        return None

    return text


def choose_transcript(sample: dict[str, Any], text_candidates: list[str]) -> Optional[str]:
    for key in text_candidates:
        value = clean_text(sample.get(key))
        if value:
            return value
    return None


# ============================================================
# DOMAIN HELPERS
# ============================================================

def safe_domain(sample: dict[str, Any], domain_column: str) -> str:
    value = sample.get(domain_column)
    if value is None:
        return "unknown_domain"

    value = str(value).strip()
    return value if value else "unknown_domain"


# ============================================================
# DATASET LOADING
# ============================================================

def open_stream(
    config: DataConfig,
    split_name: str,
    token: Optional[str] = None,
):
    ds = load_dataset(
        config.dataset_name,
        config.language,
        split=split_name,
        streaming=True,
        token=token,
    )

    # Disable dataset-side media decoding to avoid torchcodec/ffmpeg issues.
    ds = ds.decode(False)

    if split_name == config.split_train:
        ds = ds.shuffle(
            seed=config.seed,
            buffer_size=config.shuffle_buffer_size,
        )

    return ds


# ============================================================
# AUDIO HELPERS
# ============================================================

def _load_from_path(audio_path: str, target_sr: int) -> Optional[tuple[Any, int]]:
    try:
        speech_array, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if speech_array is None or len(speech_array) == 0:
            return None
        return speech_array, sr
    except Exception:
        return None


def _load_from_bytes(audio_bytes: bytes, target_sr: int) -> Optional[tuple[Any, int]]:
    try:
        with io.BytesIO(audio_bytes) as bio:
            speech_array, sr = sf.read(bio, dtype="float32")

        if speech_array is None or len(speech_array) == 0:
            return None

        if getattr(speech_array, "ndim", 1) > 1:
            speech_array = speech_array.mean(axis=1)

        if sr != target_sr:
            speech_array = librosa.resample(
                speech_array,
                orig_sr=sr,
                target_sr=target_sr,
            )
            sr = target_sr

        return speech_array, sr
    except Exception:
        pass

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        speech_array, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        if speech_array is None or len(speech_array) == 0:
            return None
        return speech_array, sr
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def extract_audio_candidate(sample: dict[str, Any], preferred_key: str) -> Optional[Any]:
    """
    Try multiple possible audio field names instead of assuming only one.
    """
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

    seen = set()
    ordered_keys = []
    for key in candidate_keys:
        if key not in seen:
            ordered_keys.append(key)
            seen.add(key)

    for key in ordered_keys:
        value = sample.get(key)
        if value is not None:
            return value

    return None


def load_audio_manually(audio_obj: Any, target_sr: int) -> Optional[tuple[Any, int]]:
    if audio_obj is None:
        return None

    if isinstance(audio_obj, dict):
        audio_path = audio_obj.get("path")
        audio_bytes = audio_obj.get("bytes")

        if audio_path:
            loaded = _load_from_path(audio_path, target_sr)
            if loaded is not None:
                return loaded

        if audio_bytes:
            loaded = _load_from_bytes(audio_bytes, target_sr)
            if loaded is not None:
                return loaded

        return None

    if isinstance(audio_obj, str):
        return _load_from_path(audio_obj, target_sr)

    return None


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_sample(
    sample: dict[str, Any],
    config: DataConfig,
    processor: WhisperProcessor,
) -> tuple[Optional[dict[str, Any]], str]:
    domain = safe_domain(sample, config.domain_column)

    transcript = choose_transcript(sample, config.text_candidates)
    if transcript is None:
        return None, "missing_text"

    audio_obj = extract_audio_candidate(sample, config.audio_column)
    if audio_obj is None:
        return None, "missing_audio_field"

    loaded = load_audio_manually(audio_obj, config.sampling_rate)
    if loaded is None:
        return None, "audio_load_failed"

    speech_array, sr = loaded

    duration = float(len(speech_array)) / float(sr)
    if duration < config.min_duration_secs:
        return None, "too_short"
    if duration > config.max_duration_secs:
        return None, "too_long"

    try:
        input_features = processor.feature_extractor(
            speech_array,
            sampling_rate=sr,
        ).input_features[0]
    except Exception:
        return None, "feature_extraction_failed"

    try:
        labels = processor.tokenizer(transcript).input_ids
    except Exception:
        return None, "tokenization_failed"

    return {
        "input_features": input_features,
        "labels": labels,
        "text": transcript,
        "duration": duration,
        "domain": domain,
        "language": config.language,
    }, "kept"


# ============================================================
# ITERABLE DATASET
# ============================================================

class PunjabiStreamingDataset(TorchIterableDataset):
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

    def _stream(self):
        return open_stream(self.config, self.split_name, token=self.token)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        random.seed(self.config.seed)

        ds = self._stream()
        buffer: list[dict[str, Any]] = []

        total_seen = 0
        total_kept = 0
        total_skipped = 0
        raw_domain_printed = 0
        skip_reason_printed = 0
        success_printed = 0
        sample_keys_printed = 0

        reason_counts = {
            "missing_text": 0,
            "missing_audio_field": 0,
            "audio_load_failed": 0,
            "too_short": 0,
            "too_long": 0,
            "feature_extraction_failed": 0,
            "tokenization_failed": 0,
            "exception": 0,
            "kept": 0,
        }

        for sample in ds:
            total_seen += 1

            raw_domain = safe_domain(sample, self.config.domain_column)

            if DEBUG_PRINT_RAW_DOMAINS and raw_domain_printed < DEBUG_RAW_DOMAIN_PRINT_LIMIT:
                print(f"[RAW DOMAIN] {repr(raw_domain)}", flush=True)
                raw_domain_printed += 1

            try:
                processed, reason = preprocess_sample(sample, self.config, self.processor)
            except Exception as e:
                processed = None
                reason = "exception"
                if DEBUG_PRINT_SKIP_REASONS and skip_reason_printed < DEBUG_SKIP_REASON_PRINT_LIMIT:
                    print(f"[SKIP EXCEPTION] {type(e).__name__}: {e}", flush=True)
                    skip_reason_printed += 1

            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if processed is None:
                total_skipped += 1

                if DEBUG_PRINT_SKIP_REASONS and skip_reason_printed < DEBUG_SKIP_REASON_PRINT_LIMIT:
                    audio_candidate = extract_audio_candidate(sample, self.config.audio_column)
                    print(
                        f"[SKIP] reason={reason} | domain={repr(raw_domain)} | "
                        f"text_keys_present={[k for k in self.config.text_candidates if sample.get(k) is not None]} | "
                        f"audio_candidate_type={type(audio_candidate).__name__}",
                        flush=True,
                    )
                    skip_reason_printed += 1

                if (
                    reason == "missing_audio_field"
                    and DEBUG_PRINT_SAMPLE_KEYS_FOR_MISSING_AUDIO
                    and sample_keys_printed < DEBUG_SAMPLE_KEYS_PRINT_LIMIT
                ):
                    print(
                        f"[MISSING AUDIO DEBUG] sample keys = {sorted(list(sample.keys()))}",
                        flush=True,
                    )
                    sample_keys_printed += 1

            else:
                total_kept += 1
                buffer.append(processed)

                if DEBUG_PRINT_SUCCESS_SAMPLES and success_printed < DEBUG_SUCCESS_PRINT_LIMIT:
                    print(
                        f"[KEPT] domain={repr(processed['domain'])} | "
                        f"duration={processed['duration']:.2f}s | "
                        f"text_len={len(processed['text'])}",
                        flush=True,
                    )
                    success_printed += 1

            if total_seen % DEBUG_PROGRESS_EVERY == 0:
                print(
                    f"[DEBUG] Seen: {total_seen} | Kept: {total_kept} | Skipped: {total_skipped} | "
                    f"Reasons: {reason_counts}",
                    flush=True,
                )

            if len(buffer) >= self.config.prefetch_size:
                print(f"[DEBUG] Buffer ready -> yielding samples | Buffer size: {len(buffer)}", flush=True)
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()

        if buffer:
            print(f"[DEBUG] Final buffer flush | Size: {len(buffer)}", flush=True)
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()

        print(
            f"[DEBUG] Stream ended | Total seen: {total_seen} | Total kept: {total_kept} | "
            f"Total skipped: {total_skipped} | Final reason counts: {reason_counts}",
            flush=True,
        )


# ============================================================
# BUILDERS
# ============================================================

def build_train_dataset(
    config: DataConfig,
    processor: WhisperProcessor,
    token: Optional[str] = None,
) -> PunjabiStreamingDataset:
    return PunjabiStreamingDataset(
        config=config,
        processor=processor,
        split_name=config.split_train,
        token=token,
    )


def build_eval_dataset(
    config: DataConfig,
    processor: WhisperProcessor,
    token: Optional[str] = None,
) -> Optional[PunjabiStreamingDataset]:
    for split_name in config.split_val_candidates:
        try:
            _ = open_stream(config, split_name, token=token)
            return PunjabiStreamingDataset(
                config=config,
                processor=processor,
                split_name=split_name,
                token=token,
            )
        except Exception:
            continue

    return None