"""Weighted composite fidelity + a one-call quality report.

Weights here are provisional and deliberately transparent; P7 calibrates them against
research-grade perceptual metrics (ViSQOL/Zimtohrli). The composite compares loudness-matched
signals, while the report also surfaces the raw level gap and SNR-style diagnostics separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from optisample.metrics.base import Metric, MetricContext, Signal, get, register
from optisample.metrics.diagnostics import loudness_delta, segmental_snr, si_sdr, snr
from optisample.metrics.preprocess import match_length, prepare
from optisample.metrics.spectral import LogMelL1, MultiResolutionStft
from optisample.metrics.timbre import MelCepstralDistortion, SpectralShape

# Register the lightweight metrics once, on import.
register(MultiResolutionStft())
register(LogMelL1())
register(MelCepstralDistortion())
register(SpectralShape())

# Provisional weights (calibrated in P7). Chosen so that on a clearly-audible degradation no
# single term dominates: MCD lives on a large dB scale, so it gets a small multiplier; the
# amplitude/log terms and the shape term are brought into the same rough range.
DEFAULT_WEIGHTS: dict[str, float] = {
    "mrstft": 1.0,
    "logmel_l1": 0.35,
    "spectral_shape": 1.0,
    "mcd": 0.01,
}


@dataclass(frozen=True)
class WeightedMetric:
    metric: Metric
    weight: float


@dataclass(frozen=True)
class CompositeFidelity:
    """Non-negative weighted sum of component distances (0 = identical)."""

    components: tuple[WeightedMetric, ...]
    name: str = "composite"

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        return float(sum(item.weight * item.metric.distance(reference, candidate, ctx) for item in self.components))

    def breakdown(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> dict[str, float]:
        """Raw (unweighted) per-metric distances, for interpretability."""
        return {item.metric.name: item.metric.distance(reference, candidate, ctx) for item in self.components}


def default_composite(weights: dict[str, float] | None = None) -> CompositeFidelity:
    """Build the lightweight composite from the registry using ``weights`` (defaults provisional)."""
    chosen = weights if weights is not None else DEFAULT_WEIGHTS
    return CompositeFidelity(tuple(WeightedMetric(get(name), weight) for name, weight in chosen.items()))


@dataclass(frozen=True)
class QualityReport:
    """Fused fidelity plus the interpretable breakdown and level/SNR diagnostics."""

    fidelity: float
    breakdown: dict[str, float]
    diagnostics: dict[str, float]


def evaluate(
    reference: Signal,
    candidate: Signal,
    sample_rate: int,
    composite: CompositeFidelity | None = None,
    *,
    normalize: bool = True,
) -> QualityReport:
    """Compare two signals: composite fidelity (loudness-matched) + interpretable diagnostics."""
    composite = composite if composite is not None else default_composite()
    raw_ref, raw_cand = match_length(reference, candidate)
    norm_ref, norm_cand, ctx = prepare(reference, candidate, sample_rate, normalize=normalize)
    return QualityReport(
        fidelity=composite.distance(norm_ref, norm_cand, ctx),
        breakdown=composite.breakdown(norm_ref, norm_cand, ctx),
        diagnostics={
            "loudness_delta_lu": loudness_delta(raw_ref, raw_cand, sample_rate),
            "si_sdr_db": si_sdr(raw_ref, raw_cand),
            "snr_db": snr(norm_ref, norm_cand),
            "segmental_snr_db": segmental_snr(norm_ref, norm_cand),
        },
    )
