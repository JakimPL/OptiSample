"""Weighted composite fidelity + a one-call quality report.

Weights and per-metric parameters are loaded from config (:mod:`optisample.config.metrics`);
:func:`build_composite` assembles the weighted set once per run from a :class:`MetricsConfig` and
is threaded through the search rather than rebuilt per comparison. The composite compares
loudness-matched signals, while the report also surfaces the raw level gap and SNR-style
diagnostics separately. Weights remain provisional and are calibrated in P7.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def score(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> tuple[float, dict[str, float]]:
        """Weighted fidelity and the raw per-metric breakdown, from a single pass over the components.

        Equivalent to :meth:`distance` paired with :meth:`breakdown`, but evaluates each metric only
        once -- the two are always needed together in :func:`evaluate`, which dominates the run, and
        each metric.distance is an STFT-heavy computation worth not repeating.
        """
        raw = [(item, item.metric.distance(reference, candidate, ctx)) for item in self.components]
        fidelity = float(sum(item.weight * dist for item, dist in raw))
        return fidelity, {item.metric.name: dist for item, dist in raw}

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        """Weighted composite distance (0 = identical)."""
        return self.score(reference, candidate, ctx)[0]

    def breakdown(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> dict[str, float]:
        """Raw (unweighted) per-metric distances, for interpretability."""
        return self.score(reference, candidate, ctx)[1]


def build_composite(config: MetricsConfig) -> CompositeFidelity:
    """Assemble the weighted composite from config: one registry-built metric per weight entry."""
    components = tuple(WeightedMetric(build_metric(name, config), weight) for name, weight in config.weights.items())
    return CompositeFidelity(
        components=components, target_lufs=config.preprocess.target_lufs, segmental=config.preprocess.segmental
    )


@dataclass(frozen=True)
class QualityReport:
    """Fused fidelity plus the interpretable breakdown and level/SNR diagnostics."""

    fidelity: float
    breakdown: dict[str, float]
    diagnostics: dict[str, float]


def _diagnostics(
    raw: tuple[Signal, Signal], norm: tuple[Signal, Signal], sample_rate: int, segmental: SegmentalSnrConfig
) -> dict[str, float]:
    """Level/SNR diagnostics surfaced next to the composite: raw signals for level, normalized for SNR."""
    raw_ref, raw_cand = raw
    norm_ref, norm_cand = norm
    return {
        "loudness_delta_lu": loudness_delta(raw_ref, raw_cand, sample_rate),
        "si_sdr_db": si_sdr(raw_ref, raw_cand),
        "snr_db": snr(norm_ref, norm_cand),
        "segmental_snr_db": segmental_snr(norm_ref, norm_cand, segmental),
    }


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
    fidelity, breakdown = composite.score(norm_ref, norm_cand, ctx)
    diagnostics = _diagnostics((raw_ref, raw_cand), (norm_ref, norm_cand), sample_rate, composite.segmental)
    return QualityReport(fidelity=fidelity, breakdown=breakdown, diagnostics=diagnostics)
