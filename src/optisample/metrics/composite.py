"""Weighted composite fidelity + a one-call quality report.

Weights and per-metric parameters are loaded from config (:mod:`optisample.config.metrics`);
:func:`build_composite` assembles the weighted set once per run from a :class:`MetricsConfig` and
is threaded through the search rather than rebuilt per comparison. The composite compares
loudness-matched signals, while the report also surfaces the raw level gap and SNR-style
diagnostics separately. Weights remain provisional and are calibrated in P7.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from optisample.config import load_config  # transitional: only ``default_composite`` needs it (removed in phase 7)
from optisample.config.metrics import MetricsConfig, SegmentalSnrConfig
from optisample.metrics.base import Metric, MetricContext, Signal, build_metric
from optisample.metrics.diagnostics import loudness_delta, segmental_snr, si_sdr, snr
from optisample.metrics.preprocess import match_length, prepare


@dataclass(frozen=True)
class WeightedMetric:
    metric: Metric
    weight: float


@dataclass(frozen=True)
class CompositeFidelity:
    """Non-negative weighted sum of component distances (0 = identical).

    Carries the loudness-normalization target and segmental-SNR framing so :func:`evaluate` needs
    only the composite (not the whole metrics config) to run both fidelity and diagnostics.
    """

    components: tuple[WeightedMetric, ...]
    target_lufs: float
    segmental: SegmentalSnrConfig
    name: str = "composite"

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        return float(sum(item.weight * item.metric.distance(reference, candidate, ctx) for item in self.components))

    def breakdown(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> dict[str, float]:
        """Raw (unweighted) per-metric distances, for interpretability."""
        return {item.metric.name: item.metric.distance(reference, candidate, ctx) for item in self.components}


def build_composite(config: MetricsConfig) -> CompositeFidelity:
    """Assemble the weighted composite from config: one registry-built metric per weight entry."""
    components = tuple(WeightedMetric(build_metric(name, config), weight) for name, weight in config.weights.items())
    return CompositeFidelity(
        components=components, target_lufs=config.preprocess.target_lufs, segmental=config.preprocess.segmental
    )


@lru_cache(maxsize=1)
def default_composite() -> CompositeFidelity:
    """Transitional: the composite built from the bundled config.

    The optimize layer now threads a config-built composite through ``EvalContext``/``OptimizeSettings``
    (phase 6); this bridge remains only for the calibrator's fallback and is removed once phase 7
    threads a composite through it.
    """
    return build_composite(load_config().metrics)


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
    composite: CompositeFidelity,
    *,
    normalize: bool = True,
) -> QualityReport:
    """Compare two signals: composite fidelity (loudness-matched) + interpretable diagnostics."""
    raw_ref, raw_cand = match_length(reference, candidate)
    norm_ref, norm_cand, ctx = prepare(reference, candidate, sample_rate, composite.target_lufs, normalize=normalize)
    return QualityReport(
        fidelity=composite.distance(norm_ref, norm_cand, ctx),
        breakdown=composite.breakdown(norm_ref, norm_cand, ctx),
        diagnostics={
            "loudness_delta_lu": loudness_delta(raw_ref, raw_cand, sample_rate),
            "si_sdr_db": si_sdr(raw_ref, raw_cand),
            "snr_db": snr(norm_ref, norm_cand),
            "segmental_snr_db": segmental_snr(norm_ref, norm_cand, composite.segmental),
        },
    )
