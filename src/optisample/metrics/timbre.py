from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.metrics import SpectralWeights
from optisample.config.spectral import MelParams, StftParams
from optisample.dsp.spectral import (
    mfcc,
    spectral_centroid,
    spectral_flatness,
    spectral_rolloff,
)
from optisample.metrics.base import MetricContext, Signal, register_metric
from optisample.metrics.diagnostics import flux_variance

_EPS: Final = 1e-9
_MCD_CONSTANT: Final = 10.0 / np.log(10.0) * np.sqrt(2.0)


@dataclass(frozen=True)
class MelCepstralDistortion:
    """Frame-averaged mel-cepstral distortion in dB (c0 excluded), assuming time-aligned input."""

    n_mfcc: int
    params: MelParams
    dynamic_range_db: float
    name: str = "mcd"

    def distance(self, reference: Signal, candidate: Signal, context: MetricContext) -> float:
        ref_mfcc = mfcc(reference, context.sample_rate, self.params, self.n_mfcc, self.dynamic_range_db)[:, 1:]
        cand_mfcc = mfcc(candidate, context.sample_rate, self.params, self.n_mfcc, self.dynamic_range_db)[:, 1:]
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

    params: StftParams
    weights: SpectralWeights
    rolloff_percent: float
    name: str = "spectral_shape"

    def components(
        self,
        reference: Signal,
        candidate: Signal,
        sample_rate: int,
    ) -> dict[str, float]:
        centroid = _relative_delta(
            spectral_centroid(reference, sample_rate, self.params),
            spectral_centroid(candidate, sample_rate, self.params),
        )
        rolloff = _relative_delta(
            spectral_rolloff(reference, sample_rate, self.params, self.rolloff_percent),
            spectral_rolloff(candidate, sample_rate, self.params, self.rolloff_percent),
        )
        flatness = abs(spectral_flatness(reference, self.params) - spectral_flatness(candidate, self.params))
        flux_variance_delta = _relative_delta(
            flux_variance(reference, self.params),
            flux_variance(candidate, self.params),
        )
        return {"centroid": centroid, "rolloff": rolloff, "flatness": flatness, "flux_variance": flux_variance_delta}

    def distance(
        self,
        reference: Signal,
        candidate: Signal,
        context: MetricContext,
    ) -> float:
        parts = self.components(reference, candidate, context.sample_rate)
        weighted = (
            self.weights.centroid * parts["centroid"],
            self.weights.rolloff * parts["rolloff"],
            self.weights.flatness * parts["flatness"],
            self.weights.flux_variance * parts["flux_variance"],
        )
        # sum(), not a ``+`` chain: CPython's compensated float summation fixes the exact score.
        return float(sum(weighted))


register_metric(
    "mcd",
    lambda config: MelCepstralDistortion(
        n_mfcc=config.mcd.n_mfcc,
        params=config.mcd.mel,
        dynamic_range_db=config.preprocess.dynamic_range_db,
    ),
)
register_metric(
    "spectral_shape",
    lambda config: SpectralShape(
        params=config.spectral_shape.stft,
        weights=config.spectral_shape.weights,
        rolloff_percent=config.spectral_shape.rolloff_percent,
    ),
)
