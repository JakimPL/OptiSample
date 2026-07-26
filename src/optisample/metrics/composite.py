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

    def score(
        self,
        reference: Signal,
        candidate: Signal,
        context: MetricContext,
    ) -> tuple[float, dict[str, float]]:
        """Weighted fidelity and the raw per-metric breakdown, from a single pass over the components.

        Equivalent to :meth:`distance` paired with :meth:`breakdown`, but evaluates each metric only
        once -- the two are always needed together in :func:`evaluate`, which dominates the run, and
        each metric.distance is an STFT-heavy computation worth not repeating.
        """
        raw = [(item, item.metric.distance(reference, candidate, context)) for item in self.components]
        fidelity = float(sum(item.weight * dist for item, dist in raw))
        return fidelity, {item.metric.name: dist for item, dist in raw}

    def distance(self, reference: Signal, candidate: Signal, context: MetricContext) -> float:
        """Weighted composite distance (0 = identical)."""
        return self.score(reference, candidate, context)[0]

    def breakdown(self, reference: Signal, candidate: Signal, context: MetricContext) -> dict[str, float]:
        """Raw (unweighted) per-metric distances, for interpretability."""
        return self.score(reference, candidate, context)[1]


def build_composite(config: MetricsConfig) -> CompositeFidelity:
    """Assemble the weighted composite from config: one registry-built metric per weight entry."""
    components = tuple(WeightedMetric(build_metric(name, config), weight) for name, weight in config.weights.items())
    return CompositeFidelity(
        components=components,
        target_lufs=config.preprocess.target_lufs,
        segmental=config.preprocess.segmental,
    )


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
    """Compare two signals: composite fidelity (loudness-matched) + interpretable diagnostics.

    The level diagnostics read the raw (length-matched) signals; the SNR diagnostics read the
    loudness-normalized pair, so each is measured on the signals it is meaningful for.
    """
    raw_ref, raw_cand = match_length(reference, candidate)
    norm_ref, norm_cand, context = prepare(
        reference,
        candidate,
        sample_rate,
        composite.target_lufs,
        normalize=normalize,
    )
    fidelity, breakdown = composite.score(norm_ref, norm_cand, context)
    diagnostics = {
        "loudness_delta_lu": loudness_delta(raw_ref, raw_cand, sample_rate),
        "si_sdr_db": si_sdr(raw_ref, raw_cand),
        "snr_db": snr(norm_ref, norm_cand),
        "segmental_snr_db": segmental_snr(
            norm_ref,
            norm_cand,
            composite.segmental,
        ),
    }
    return QualityReport(
        fidelity=fidelity,
        breakdown=breakdown,
        diagnostics=diagnostics,
    )
