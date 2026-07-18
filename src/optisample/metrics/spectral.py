"""Spectral fidelity metrics: multi-resolution STFT and log-mel L1.

The multi-resolution STFT distance is the workhorse — it captures quantization noise,
bandlimiting/HF loss and coarse envelope shape at once, and is invariant to nothing it
shouldn't be (level is handled by the normalize-before-compare harness).

The log terms clamp magnitudes to a fixed dynamic range below the *reference peak* before the
logarithm. Without this, a plain ``log(mag + tiny_eps)`` blows up in near-silent bins: it would
rate a transparent 16-bit requantization (whose −90 dB noise fills otherwise-empty HF bins)
as *worse* than an audible bandwidth cut. The floor makes inaudible noise invisible while
still penalizing audible noise (e.g. 8-bit at ~−49 dB).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisample.dsp.spectral import MelParams, StftParams, melspectrogram, stft_magnitude
from optisample.metrics.base import MetricContext, Signal

_TINY = 1e-12
_STFT_REL_FLOOR = 1e-4  # 80 dB below the reference peak, amplitude domain
_MEL_REL_FLOOR = 1e-8  # 80 dB below the reference peak, power domain

DEFAULT_RESOLUTIONS: tuple[StftParams, ...] = (
    StftParams(n_fft=256, hop_length=64),
    StftParams(n_fft=1024, hop_length=256),
    StftParams(n_fft=2048, hop_length=512),
)


def _min_frames(first: Signal, second: Signal) -> int:
    return int(min(first.shape[0], second.shape[0]))


def _floored_log(values: Signal, reference_peak: float, rel_floor: float) -> Signal:
    """Natural log after clamping ``values`` to ``rel_floor`` of the reference peak."""
    floor = max(reference_peak * rel_floor, _TINY)
    return np.log(np.maximum(values, floor))


@dataclass(frozen=True)
class MultiResolutionStft:
    """Mean over resolutions of (spectral convergence + dynamic-range-floored log-magnitude L1)."""

    resolutions: tuple[StftParams, ...] = DEFAULT_RESOLUTIONS
    name: str = "mrstft"

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        del ctx  # resolution set is fixed; sample rate does not change the distance
        total = 0.0
        for params in self.resolutions:
            ref_mag = stft_magnitude(reference, params)
            cand_mag = stft_magnitude(candidate, params)
            frames = _min_frames(ref_mag, cand_mag)
            ref_mag, cand_mag = ref_mag[:frames], cand_mag[:frames]
            convergence = float(np.linalg.norm(ref_mag - cand_mag) / (np.linalg.norm(ref_mag) + _TINY))
            peak = float(np.max(ref_mag)) if ref_mag.size else 0.0
            log_l1 = float(
                np.mean(
                    np.abs(_floored_log(ref_mag, peak, _STFT_REL_FLOOR) - _floored_log(cand_mag, peak, _STFT_REL_FLOOR))
                )
            )
            total += convergence + log_l1
        return total / len(self.resolutions)


@dataclass(frozen=True)
class LogMelL1:
    """Mean absolute difference of dynamic-range-floored log-mel spectrograms."""

    params: MelParams = field(default_factory=MelParams)
    name: str = "logmel_l1"

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        ref_mel = melspectrogram(reference, ctx.sample_rate, self.params)
        cand_mel = melspectrogram(candidate, ctx.sample_rate, self.params)
        frames = _min_frames(ref_mel, cand_mel)
        ref_mel, cand_mel = ref_mel[:frames], cand_mel[:frames]
        peak = float(np.max(ref_mel)) if ref_mel.size else 0.0
        ref_log = _floored_log(ref_mel, peak, _MEL_REL_FLOOR)
        cand_log = _floored_log(cand_mel, peak, _MEL_REL_FLOOR)
        return float(np.mean(np.abs(ref_log - cand_log)))
