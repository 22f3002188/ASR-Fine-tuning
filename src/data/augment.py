"""
Audio augmentation for streaming ASR finetuning.

Two augmentation stages:
  1. augment_waveform()  — operates on raw float32 numpy array before feature extraction
  2. augment_features()  — SpecAugment on log-mel spectrogram [n_mels, time] after extraction

Both are called inside the streaming .map() — no data is materialised to disk.
"""

import random
import numpy as np


class AudioAugmentor:
    """
    Stateless callable that applies a stochastic augmentation chain.
    Instantiate once, call per sample inside dataset._prepare_sample().
    """

    def __init__(
        self,
        noise_prob: float = 0.3,
        speed_prob: float = 0.3,
        speed_range: tuple = (0.9, 1.1),
        do_spec_augment: bool = True,
        # SpecAugment params
        num_time_masks: int = 2,
        time_mask_param: int = 50,
        num_freq_masks: int = 2,
        freq_mask_param: int = 10,
    ):
        self.noise_prob = noise_prob
        self.speed_prob = speed_prob
        self.speed_range = speed_range
        self.do_spec_augment = do_spec_augment
        self.num_time_masks = num_time_masks
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.freq_mask_param = freq_mask_param

    # ── Stage 1: waveform ─────────────────────────────────────────────────────

    def augment_waveform(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise and speed perturbation to a raw float32 waveform."""
        if random.random() < self.noise_prob:
            audio = _add_gaussian_noise(audio, snr_db=random.uniform(15, 35))
        if random.random() < self.speed_prob:
            rate = random.uniform(*self.speed_range)
            audio = _speed_perturb(audio, rate=rate)
        return audio

    # ── Stage 2: spectrogram ──────────────────────────────────────────────────

    def augment_features(self, features: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to a log-mel spectrogram [n_mels, time]."""
        if self.do_spec_augment:
            return _spec_augment(
                features,
                num_time_masks=self.num_time_masks,
                time_mask_param=self.time_mask_param,
                num_freq_masks=self.num_freq_masks,
                freq_mask_param=self.freq_mask_param,
            )
        return features


# ── Individual transforms ──────────────────────────────────────────────────────

def _add_gaussian_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Additive white Gaussian noise at a given signal-to-noise ratio."""
    signal_power = np.mean(audio ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), audio.shape).astype(np.float32)
    return audio + noise


def _speed_perturb(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Speed perturbation via linear index resampling.
    rate > 1.0 → faster (shorter), rate < 1.0 → slower (longer).

    Note: this changes duration. Samples that become > 30s after perturbation
    will be caught by the duration filter in dataset._prepare_sample().
    For production, prefer audiomentations.TimeStretch which preserves pitch.
    """
    if rate == 1.0:
        return audio
    indices = np.round(np.arange(0, len(audio), rate)).astype(np.int64)
    indices = indices[indices < len(audio)]
    return audio[indices].astype(np.float32)


def _spec_augment(
    features: np.ndarray,
    num_time_masks: int = 2,
    time_mask_param: int = 50,
    num_freq_masks: int = 2,
    freq_mask_param: int = 10,
) -> np.ndarray:
    """
    SpecAugment on a log-mel spectrogram [n_mels, time].
    Masked regions are filled with the feature mean — less destructive than zero-fill.
    """
    features = features.copy()
    mean_val = features.mean()
    _, T = features.shape

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, T))
        t0 = random.randint(0, max(T - t, 1))
        features[:, t0 : t0 + t] = mean_val

    for _ in range(num_freq_masks):
        n_mels = features.shape[0]
        f = random.randint(0, min(freq_mask_param, n_mels))
        f0 = random.randint(0, max(n_mels - f, 1))
        features[f0 : f0 + f, :] = mean_val

    return features
