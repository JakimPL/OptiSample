"""DSP-layer configuration: spectral-analysis sizing, loop detection, and quantization.

``StftParams``/``MelParams`` were the ``dsp/spectral.py`` parameter bundles; they now live here so the
analysis resolution is a tunable, not a signature default. ``LoopConfig`` holds the loop detector's
band, periodicity gate, placement and seam-crossfade knobs; ``EncodeConfig`` is the derived bundle the
surrogate encoder needs (loop detection + the normalization peak).
"""

from __future__ import annotations

from optisample.config.base import ConfigModel


class StftParams(ConfigModel):
    """Short-time Fourier transform sizing."""

    n_fft: int
    hop_length: int


class MelParams(ConfigModel):
    """Mel-spectrogram sizing (carries its own STFT sizing)."""

    n_fft: int
    hop_length: int
    n_mels: int
    fmin: float
    fmax: float | None

    def stft(self) -> StftParams:
        return StftParams(n_fft=self.n_fft, hop_length=self.hop_length)


class SpectralConfig(ConfigModel):
    """General analysis params for the standalone spectral primitives (centroid/rolloff/flux/...)."""

    stft: StftParams
    mel: MelParams


class LoopConfig(ConfigModel):  # pylint: disable=too-many-instance-attributes
    """Loop-point detection: search band, periodicity/sustain gates, placement, and seam crossfade."""

    min_hz: float
    max_hz: float
    min_correlation: float
    min_periods: int
    min_loop_s: float
    attack_skip_s: float
    tail_skip_s: float
    max_estimation_s: float
    sustain_decay_ratio: float
    crossfade_s: float


class QuantizeConfig(ConfigModel):
    """Requantization shaping: the peak a sample is normalized to before storing ("store hot")."""

    target_peak: float


class EncodeConfig(ConfigModel):
    """What :func:`optisample.dsp.surrogate.encode` needs beyond one swept ``EncodingParams`` point."""

    loop: LoopConfig
    target_peak: float
