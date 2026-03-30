"""Unit tests: data loading, augmentation, collation."""

import numpy as np
import pytest


def test_clean_text_basic():
    from src.data.dataset import clean_text
    assert clean_text("  hello  ") == "hello"
    assert clean_text("") is None
    assert clean_text(None) is None
    assert clean_text("a") is None          # too short
    assert clean_text("hello  world") == "hello world"


def test_basic_normalize():
    from src.data.dataset import _basic_normalize
    assert _basic_normalize("  ਸਤ  ਸ੍ਰੀ  ਅਕਾਲ  ") == "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"


def test_extract_audio_candidate_preferred_key():
    from src.data.dataset import extract_audio_candidate
    sample = {"audio": "a.wav", "audio_filepath": "b.wav"}
    assert extract_audio_candidate(sample, "audio_filepath") == "b.wav"
    assert extract_audio_candidate(sample, "audio") == "a.wav"
    assert extract_audio_candidate({}, "audio") is None


def test_load_audio_manually_dict_with_array():
    from src.data.dataset import load_audio_manually
    arr = np.ones(16000, dtype=np.float32)
    result = load_audio_manually({"array": arr, "sampling_rate": 16000}, 16000)
    assert result is not None
    loaded_arr, sr = result
    assert sr == 16000
    assert len(loaded_arr) == 16000


def test_load_audio_manually_resample():
    from src.data.dataset import load_audio_manually
    # 8kHz array should be resampled to 16kHz
    arr = np.zeros(8000, dtype=np.float32)
    result = load_audio_manually({"array": arr, "sampling_rate": 8000}, 16000)
    assert result is not None
    _, sr = result
    assert sr == 16000


def test_dataconfig_from_omega_ignores_unknown_keys():
    """from_omega should silently drop keys not in DataConfig fields."""
    from src.data.dataset import DataConfig
    from unittest.mock import MagicMock
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "data": {
            "dataset_name": "test/ds",
            "language": "punjabi",
            "unknown_future_key": "should_be_ignored",
        }
    })
    dc = DataConfig.from_omega(cfg)
    assert dc.dataset_name == "test/ds"
    assert dc.language == "punjabi"


def test_augmentor_waveform():
    from src.data.augment import AudioAugmentor
    aug = AudioAugmentor(noise_prob=1.0, speed_prob=0.0, do_spec_augment=False)
    audio = np.zeros(16000, dtype=np.float32)
    result = aug.augment_waveform(audio)
    assert result.shape == audio.shape
    assert result.dtype == np.float32


def test_augmentor_spec_augment():
    from src.data.augment import AudioAugmentor
    aug = AudioAugmentor(noise_prob=0.0, speed_prob=0.0, do_spec_augment=True)
    features = np.ones((128, 3000), dtype=np.float32)
    result = aug.augment_features(features)
    assert result.shape == (128, 3000)
    # Some values should be masked (set to mean)
    assert not np.all(result == 1.0)