"""Timbre metrics: mel-cepstral distortion and a spectral-shape composite.

The spectral-shape metric folds in a flux-variance mismatch term: a too-short loop is
spectrally *static* compared with an evolving real sustain, which brightness/rolloff deltas
alone would not catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optisample.dsp.spectral import (
    MelParams,
    StftParams,
    mfcc,
    spectral_centroid,
    spectral_flatness,
    spectral_flux,
    spectral_rolloff,
)
from optisample.metrics.base import MetricContext, Signal

_EPS = 1e-9
_MCD_CONSTANT = 10.0 / np.log(10.0) * np.sqrt(2.0)


@dataclass(frozen=True)
class MelCepstralDistortion:
    """Frame-averaged mel-cepstral distortion in dB (c0 excluded), assuming time-aligned input."""

    n_mfcc: int = 13
    params: MelParams = field(default_factory=MelParams)
    name: str = "mcd"

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        ref_mfcc = mfcc(reference, ctx.sample_rate, self.n_mfcc, self.params)[:, 1:]
        cand_mfcc = mfcc(candidate, ctx.sample_rate, self.n_mfcc, self.params)[:, 1:]
        frames = int(min(ref_mfcc.shape[0], cand_mfcc.shape[0]))
        if frames == 0:
            return 0.0
        diff = ref_mfcc[:frames] - cand_mfcc[:frames]
        per_frame = _MCD_CONSTANT * np.sqrt(np.sum(diff**2, axis=1))
        return float(np.mean(per_frame))


def _relative_delta(reference: float, candidate: float) -> float:
    return abs(reference - candidate) / (abs(reference) + _EPS)


@dataclass(frozen=True)
class SpectralShape:
    """Weighted brightness/rolloff/flatness deltas plus a flux-variance (static-loop) mismatch."""

    params: StftParams = field(default_factory=StftParams)
    weights: tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)
    name: str = "spectral_shape"

    def components(self, reference: Signal, candidate: Signal, sample_rate: int) -> dict[str, float]:
        centroid = _relative_delta(
            spectral_centroid(reference, sample_rate, self.params),
            spectral_centroid(candidate, sample_rate, self.params),
        )
        rolloff = _relative_delta(
            spectral_rolloff(reference, sample_rate, params=self.params),
            spectral_rolloff(candidate, sample_rate, params=self.params),
        )
        flatness = abs(spectral_flatness(reference, self.params) - spectral_flatness(candidate, self.params))
        flux_variance = _relative_delta(
            float(np.std(spectral_flux(reference, self.params))),
            float(np.std(spectral_flux(candidate, self.params))),
        )
        return {"centroid": centroid, "rolloff": rolloff, "flatness": flatness, "flux_variance": flux_variance}

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        parts = self.components(reference, candidate, ctx.sample_rate)
        keys = ("centroid", "rolloff", "flatness", "flux_variance")
        return float(sum(weight * parts[key] for weight, key in zip(self.weights, keys)))
