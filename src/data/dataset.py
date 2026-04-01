"""
Streaming ASR dataset for IndicVoices + Whisper.

Audio loading strategy (from dataset exploration):
  - audio column may be a dict {"array":..., "sampling_rate":...},
    a filepath string, raw bytes, or a torchcodec AudioDecoder.
  - load_audio_manually() handles all cases with fallback chains.
  - open_stream() does NOT cast_column — raw access is more robust
    across IndicVoices versions and avoids torchcodec dependency issues.

Inherits torch.utils.data.IterableDataset directly — compatible with
Seq2SeqTrainer without any HuggingFace IterableDataset wrapper.
"""

from __future__ import annotations

import io
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import librosa
import soundfile as sf
from datasets import load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import WhisperProcessor

try:
    from src.data.augment import AudioAugmentor
except Exception:
    AudioAugmentor = None


# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000

# Debug verbosity — set to 0 in production to silence
DEBUG_PROGRESS_EVERY = 50
DEBUG_PRINT_RAW_DOMAINS = True
DEBUG_RAW_DOMAIN_PRINT_LIMIT = 20
DEBUG_PRINT_SKIP_REASONS = True
DEBUG_SKIP_REASON_PRINT_LIMIT = 30
DEBUG_PRINT_SUCCESS_SAMPLES = True
DEBUG_SUCCESS_PRINT_LIMIT = 20
DEBUG_PRINT_SAMPLE_KEYS_FOR_MISSING_AUDIO = True
DEBUG_SAMPLE_KEYS_PRINT_LIMIT = 10


# ── Config ────────────────────────────────────────────────────────────────────

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
    min_duration_secs: float = 1.0
    max_duration_secs: float = 30.0
    buffer_size: int = 500
    shuffle_buffer_size: int = 500
    prefetch_size: int = 8
    seed: int = 42
    augmentation: dict = field(default_factory=dict)
    normalization: str = "basic"

    @classmethod
    def from_omega(cls, cfg) -> "DataConfig":
        import dataclasses
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(cfg.data, resolve=True)

        if "buffer_size" in raw and "shuffle_buffer_size" not in raw:
            raw["shuffle_buffer_size"] = raw["buffer_size"]

        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in valid})


# ── Text helpers ──────────────────────────────────────────────────────────────

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


def _basic_normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def choose_transcript(
    sample: dict[str, Any],
    text_column: str,
    normalization: str = "basic",
) -> Optional[str]:
    value = clean_text(sample.get(text_column))
    if value and normalization == "basic":
        value = _basic_normalize(value)
    return value


# ── Domain helpers ────────────────────────────────────────────────────────────

def safe_domain(sample: dict[str, Any], domain_column: str) -> str:
    value = sample.get(domain_column)
    if value is None:
        return "unknown_domain"
    value = str(value).strip()
    return value if value else "unknown_domain"


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _ensure_hf_cache_env() -> None:
    """
    Force all Hugging Face caches to writable user paths.
    This avoids accidental writes to read-only system paths like /models/huggingface.
    """
    hf_home = os.getenv("HF_HOME", "/home/harsh/hf_cache")
    hf_hub_cache = os.getenv("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    hf_datasets_cache = os.getenv("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
    hf_transformers_cache = os.getenv("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    hf_assets_cache = os.getenv("HUGGINGFACE_ASSETS_CACHE", os.path.join(hf_home, "assets"))

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_hub_cache
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_hub_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_transformers_cache
    os.environ["HUGGINGFACE_ASSETS_CACHE"] = hf_assets_cache

    for path in [
        hf_home,
        hf_hub_cache,
        hf_datasets_cache,
        hf_transformers_cache,
        hf_assets_cache,
    ]:
        os.makedirs(path, exist_ok=True)


# ── Dataset loading ───────────────────────────────────────────────────────────

def open_stream(
    config: DataConfig,
    split_name: str,
    token: Optional[str] = None,
):
    """
    Open a raw streaming split. No cast_column — audio is decoded manually.
    """
    _ensure_hf_cache_env()
    token = token or os.getenv("HF_TOKEN")

    ds = load_dataset(
        config.dataset_name,
        config.language,
        split=split_name,
        streaming=True,
        token=token,
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )

    try:
        ds = ds.decode(False)
    except Exception:
        pass

    if split_name == config.split_train:
        ds = ds.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer_size)

    return ds


# ── Audio loading ─────────────────────────────────────────────────────────────

def _load_from_path(audio_path: str, target_sr: int) -> Optional[tuple]:
    try:
        arr, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return (arr, sr) if arr is not None and len(arr) > 0 else None
    except Exception:
        return None


def _load_from_bytes(audio_bytes: bytes, target_sr: int) -> Optional[tuple]:
    try:
        with io.BytesIO(audio_bytes) as bio:
            arr, sr = sf.read(bio, dtype="float32")
        if arr is None or len(arr) == 0:
            return None
        if getattr(arr, "ndim", 1) > 1:
            arr = arr.mean(axis=1)
        if sr != target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        return arr, target_sr
    except Exception:
        pass

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        arr, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        return (arr, sr) if arr is not None and len(arr) > 0 else None
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def extract_audio_candidate(sample: dict[str, Any], preferred_key: str) -> Optional[Any]:
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
    for key in candidate_keys:
        if key not in seen:
            seen.add(key)
            value = sample.get(key)
            if value is not None:
                return value
    return None


def load_audio_manually(audio_obj: Any, target_sr: int) -> Optional[tuple]:
    """
    Decode audio from any format IndicVoices may provide:
      - dict {"array": ndarray, "sampling_rate": int}
      - dict {"path": str, "bytes": bytes}
      - str filepath
      - torchcodec AudioDecoder
    """
    if audio_obj is None:
        return None

    if isinstance(audio_obj, dict):
        if "array" in audio_obj and audio_obj["array"] is not None:
            try:
                arr = audio_obj["array"]
                sr = int(audio_obj.get("sampling_rate", target_sr))
                if getattr(arr, "ndim", 1) > 1:
                    arr = arr.mean(axis=1)
                if sr != target_sr:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
                return (arr, target_sr) if arr is not None and len(arr) > 0 else None
            except Exception:
                pass

        path = audio_obj.get("path")
        audio_bytes = audio_obj.get("bytes")
        if path:
            loaded = _load_from_path(path, target_sr)
            if loaded is not None:
                return loaded
        if audio_bytes:
            return _load_from_bytes(audio_bytes, target_sr)
        return None

    if isinstance(audio_obj, str):
        return _load_from_path(audio_obj, target_sr)

    try:
        decoded = audio_obj.get_all_samples()
        arr = decoded.data.squeeze(0).numpy().astype("float32")
        return (arr, target_sr) if arr is not None and len(arr) > 0 else None
    except Exception:
        return None


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_sample(
    sample: dict[str, Any],
    config: DataConfig,
    processor: WhisperProcessor,
    augmentor: Optional[Any] = None,
    augment: bool = False,
) -> tuple[Optional[dict], str]:
    """
    Full preprocessing pipeline for one sample.
    Returns (processed_dict, reason) where reason is "kept" or a skip reason string.
    """
    transcript = choose_transcript(sample, config.text_column, config.normalization)
    if transcript is None:
        return None, "missing_text"

    audio_obj = extract_audio_candidate(sample, config.audio_column)
    if audio_obj is None:
        return None, "missing_audio_field"

    loaded = load_audio_manually(audio_obj, config.sampling_rate)
    if loaded is None:
        return None, "audio_load_failed"

    arr, sr = loaded
    duration = float(len(arr)) / float(sr)

    if duration < config.min_duration_secs:
        return None, "too_short"
    if duration > config.max_duration_secs:
        return None, "too_long"

    if augment and augmentor is not None:
        try:
            arr = augmentor.augment_waveform(arr)
        except Exception:
            pass

    try:
        input_features = processor.feature_extractor(
            arr,
            sampling_rate=sr,
        ).input_features[0]
    except Exception:
        return None, "feature_extraction_failed"

    if augment and augmentor is not None:
        try:
            input_features = augmentor.augment_features(input_features)
        except Exception:
            pass

    try:
        labels = processor.tokenizer(transcript).input_ids
    except Exception:
        return None, "tokenization_failed"

    if len(labels) > 448:
        return None, "labels_too_long"

    return {
        "input_features": input_features,
        "labels": labels,
        "text": transcript,
        "duration": duration,
        "domain": safe_domain(sample, config.domain_column),
        "language": config.language,
    }, "kept"


# ── Iterable dataset ──────────────────────────────────────────────────────────

class StreamingASRDataset(TorchIterableDataset):
    """
    PyTorch IterableDataset wrapping IndicVoices streaming splits.
    Compatible with Seq2SeqTrainer directly.
    """

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

        aug_cfg = config.augmentation or {}
        self.augmentor = None
        if AudioAugmentor is not None and aug_cfg.get("enabled", False):
            self.augmentor = AudioAugmentor(
                noise_prob=aug_cfg.get("noise_prob", 0.3),
                speed_prob=aug_cfg.get("speed_prob", 0.3),
                speed_range=tuple(aug_cfg.get("speed_range", [0.9, 1.1])),
                do_spec_augment=aug_cfg.get("spec_augment", True),
            )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        random.seed(self.config.seed)
        ds = open_stream(self.config, self.split_name, token=self.token)

        buffer: list[dict] = []
        total_seen = total_kept = total_skipped = 0
        raw_domain_printed = skip_reason_printed = success_printed = sample_keys_printed = 0
        reason_counts: dict[str, int] = {}
        is_train = self.split_name == self.config.split_train

        for sample in ds:
            total_seen += 1
            raw_domain = safe_domain(sample, self.config.domain_column)

            if DEBUG_PRINT_RAW_DOMAINS and raw_domain_printed < DEBUG_RAW_DOMAIN_PRINT_LIMIT:
                # print(f"[RAW DOMAIN] {repr(raw_domain)}", flush=True)
                raw_domain_printed += 1

            try:
                processed, reason = preprocess_sample(
                    sample,
                    self.config,
                    self.processor,
                    augmentor=self.augmentor,
                    augment=is_train,
                )
            except Exception as e:
                processed, reason = None, "exception"
                if DEBUG_PRINT_SKIP_REASONS and skip_reason_printed < DEBUG_SKIP_REASON_PRINT_LIMIT:
                    print(f"[SKIP EXCEPTION] {type(e).__name__}: {e}", flush=True)
                    skip_reason_printed += 1

            reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if processed is None:
                total_skipped += 1
                if DEBUG_PRINT_SKIP_REASONS and skip_reason_printed < DEBUG_SKIP_REASON_PRINT_LIMIT:
                    audio_candidate = extract_audio_candidate(sample, self.config.audio_column)
                    # print(
                    #     f"[SKIP] reason={reason} | domain={repr(raw_domain)} | "
                    #     f"text_present={sample.get(self.config.text_column) is not None} | "
                    #     f"audio_type={type(audio_candidate).__name__}",
                    #     flush=True,
                    # )
                    skip_reason_printed += 1

                if (
                    reason == "missing_audio_field"
                    and DEBUG_PRINT_SAMPLE_KEYS_FOR_MISSING_AUDIO
                    and sample_keys_printed < DEBUG_SAMPLE_KEYS_PRINT_LIMIT
                ):
                    print(f"[MISSING AUDIO DEBUG] keys={sorted(sample.keys())}", flush=True)
                    sample_keys_printed += 1
            else:
                total_kept += 1
                buffer.append(processed)

                if DEBUG_PRINT_SUCCESS_SAMPLES and success_printed < DEBUG_SUCCESS_PRINT_LIMIT:
                    # print(
                    #     f"[KEPT] domain={repr(processed['domain'])} | "
                    #     f"duration={processed['duration']:.2f}s | "
                    #     f"text_len={len(processed['text'])}",
                    #     flush=True,
                    # )
                    success_printed += 1

            # if total_seen % DEBUG_PROGRESS_EVERY == 0:
            #     print(
            #         f"[DEBUG] Seen:{total_seen} Kept:{total_kept} "
            #         f"Skipped:{total_skipped} Reasons:{reason_counts}",
            #         flush=True,
            #     )

            if len(buffer) >= self.config.prefetch_size:
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()

        if buffer:
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()

        print(
            f"[DEBUG] Stream ended | Seen:{total_seen} Kept:{total_kept} "
            f"Skipped:{total_skipped} | {reason_counts}",
            flush=True,
        )


# ── Builders ──────────────────────────────────────────────────────────────────

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
        open_stream(config, config.split_val, token=token)
        return StreamingASRDataset(
            config=config,
            processor=processor,
            split_name=config.split_val,
            token=token,
        )
    except Exception:
        return None