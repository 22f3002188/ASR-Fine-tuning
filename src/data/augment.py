"""
Audio augmentation for streaming ASR finetuning.

All transforms operate on raw numpy arrays (float32, mono)
so they can be composed inside a streaming .map() call
without breaking the lazy evaluation chain.

Augmentation is intentionally kept off for val/test splits —
pass augment=False when calling get_split("val").
"""

import random
import numpy as np


# ------------------------------------------------------------------
# Individual transforms
# ------------------------------------------------------------------

def spec_augment(
    input_features: np.ndarray,
    num_time_masks: int = 2,
    time_mask_param: int = 50,
    num_freq_masks: int = 2,
    freq_mask_param: int = 10,
) -> np.ndarray:
    """
    SpecAugment on a log-mel spectrogram [n_mels, time].
    Masks are filled with the feature mean (less destructive than 0).
    """
    features = input_features.copy()
    mean_val = features.mean()

    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(features.shape[1] - t, 1))
        features[:, t0 : t0 + t] = mean_val

    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(features.shape[0] - f, 1))
        features[f0 : f0 + f, :] = mean_val

    return features


def add_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Additive white Gaussian noise at a target SNR."""
    signal_power = np.mean(audio ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    return (audio + noise).astype(np.float32)


def speed_perturb(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """
    Simple speed perturbation via linear interpolation.
    rate < 1.0 → slower, rate > 1.0 → faster.
    Prefer audiomentations.TimeStretch for production use.
    """
    if rate == 1.0:
        return audio
    indices = np.round(np.arange(0, len(audio), rate)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices].astype(np.float32)


# ------------------------------------------------------------------
# Composed augmentation callable (used inside .map())
# ------------------------------------------------------------------

class AudioAugmentor:
    """
    Stateless callable that applies a random augmentation chain.

    Usage inside StreamingASRDataset._prepare_sample:
        if self.augmentor:
            audio["array"] = self.augmentor(audio["array"])
    """

    def __init__(
        self,
        noise_prob: float = 0.3,
        speed_prob: float = 0.3,
        speed_range: tuple[float, float] = (0.9, 1.1),
        spec_augment: bool = True,
    ):
        self.noise_prob   = noise_prob
        self.speed_prob   = speed_prob
        self.speed_range  = speed_range
        self.do_spec      = spec_augment

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if random.random() < self.noise_prob:
            snr = random.uniform(15, 35)
            audio = add_noise(audio, snr_db=snr)
        if random.random() < self.speed_prob:
            rate = random.uniform(*self.speed_range)
            audio = speed_perturb(audio, rate=rate)
        return audio

    def apply_spec_augment(self, features: np.ndarray) -> np.ndarray:
        """Call this on input_features after feature extraction."""
        if self.do_spec:
            return spec_augment(features)
        return features
