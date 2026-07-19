"""Fidelity-metric configuration: composite weights and every metric's internal parameters.

Nested so each group stays small: the composite is ``weights`` (name -> weight) plus one config per
component metric, and ``preprocess`` carries the loudness-normalization target, the single
dynamic-range floor shared by the log-spectral metrics, and the segmental-SNR framing. Weights and
resolutions are the calibration surface (P7); they are provisional but now live in one YAML file.
"""

from __future__ import annotations

from optisample.config.base import ConfigModel
from optisample.config.dsp import MelParams, StftParams


class MrstftConfig(ConfigModel):
    """Multi-resolution STFT distance: the set of ``(n_fft, hop)`` resolutions to average over."""

    resolutions: tuple[StftParams, ...]


class LogMelConfig(ConfigModel):
    """Log-mel L1 distance sizing."""

    mel: MelParams


class McdConfig(ConfigModel):
    """Mel-cepstral distortion: coefficient count and the mel sizing behind it."""

    n_mfcc: int
    mel: MelParams


class SpectralShapeConfig(ConfigModel):
    """Brightness/rolloff/flatness/flux-variance term: analysis sizing, sub-weights, rolloff fraction."""

    stft: StftParams
    weights: tuple[float, float, float, float]
    rolloff_percent: float


class SegmentalSnrConfig(ConfigModel):
    """Segmental-SNR diagnostic: per-frame framing and the dB clip range."""

    frame_length: int
    hop_length: int
    clip_low_db: float
    clip_high_db: float


class PreprocessConfig(ConfigModel):
    """Shared comparison settings: loudness target, log dynamic-range floor, segmental-SNR framing."""

    target_lufs: float
    dynamic_range_db: float
    segmental: SegmentalSnrConfig


class MetricsConfig(ConfigModel):
    """The composite fidelity: term weights plus each component metric's configuration."""

    weights: dict[str, float]
    mrstft: MrstftConfig
    logmel: LogMelConfig
    mcd: McdConfig
    spectral_shape: SpectralShapeConfig
    preprocess: PreprocessConfig
