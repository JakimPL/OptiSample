from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Final

from pydantic import BaseModel, ConfigDict

from optisample.artifacts.documents.loops import LoopsDocument, RecordingLoopsRecord
from optisample.artifacts.documents.metrics import MetricsDocument
from optisample.artifacts.documents.plan import EncodingRecord, PlanDocument
from optisample.artifacts.documents.reduction import ReducedDocument, ReductionDocument

_NOTHING: Final = 0.0
_UNLOOPED: Final = None
_ONE_KEY: Final = 1


class Readings(BaseModel):
    """The shape every reading of an artifact shares: frozen, and serialized straight into the record store."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _median(values: Sequence[float]) -> float:
    return float(median(values)) if values else _NOTHING


def _mean(values: Sequence[float]) -> float:
    return float(fmean(values)) if values else _NOTHING


def _share(part: Iterable[bool]) -> float:
    counted = list(part)
    return sum(counted) / len(counted) if counted else _NOTHING


def _weighted(values: Sequence[float | None], weights: Sequence[float]) -> float:
    """The mean of ``values`` under ``weights``, over the readings that came back."""
    paired = [(value, weight) for value, weight in zip(values, weights) if value is not None]
    total = sum(weight for _, weight in paired)
    if total <= _NOTHING:
        return _NOTHING

    return sum(value * weight for value, weight in paired) / total


def _span_ratios(recordings: Sequence[RecordingLoopsRecord]) -> list[float]:
    """How far each recording's dearest offer stands above its cheapest, which is the length axis on offer."""
    return [
        max(offer.end_s for offer in recording.offered) / min(offer.end_s for offer in recording.offered)
        for recording in recordings
        if recording.offered
    ]


class LoopReadings(Readings):
    """What the loop stage settled: how much material loops, what it offers, and what it turned down."""

    recordings: int
    looped_recordings: int
    looped_share: float
    offers: int
    offers_per_recording: float
    rejected: int
    rejected_by_gate: dict[str, int]
    lacking: dict[str, int]
    median_offer_s: float
    shortest_offer_s: float
    longest_offer_s: float
    median_span_ratio: float
    median_seam_step: float
    median_spectral_distance_db: float
    median_level_drift_db: float

    @classmethod
    def read(cls, loops_json: Path) -> LoopReadings:
        """The readings ``loops_json`` states, over every recording the stage measured."""
        document = LoopsDocument.model_validate_json(loops_json.read_text(encoding="utf-8"))
        recordings = document.recordings
        offered = [offer for recording in recordings for offer in recording.offered]
        rejected = [turned for recording in recordings for turned in recording.rejected]
        return cls(
            recordings=len(recordings),
            looped_recordings=sum(bool(recording.offered) for recording in recordings),
            looped_share=_share(bool(recording.offered) for recording in recordings),
            offers=len(offered),
            offers_per_recording=_mean([len(recording.offered) for recording in recordings]),
            rejected=len(rejected),
            rejected_by_gate=dict(Counter(str(turned.gate) for turned in rejected)),
            lacking=dict(Counter(str(recording.lacking) for recording in recordings if recording.lacking)),
            median_offer_s=_median([offer.end_s for offer in offered]),
            shortest_offer_s=min((offer.end_s for offer in offered), default=_NOTHING),
            longest_offer_s=max((offer.end_s for offer in offered), default=_NOTHING),
            median_span_ratio=_median(_span_ratios(recordings)),
            median_seam_step=_median([offer.quality.seam_step for offer in offered]),
            median_spectral_distance_db=_median([offer.quality.spectral_distance for offer in offered]),
            median_level_drift_db=_median([abs(offer.quality.level_drift_db) for offer in offered]),
        )


class ReductionReadings(Readings):
    """What the pre-optimization stage left: how many recordings, how long, and the rung each is stored at."""

    listed_recordings: int
    kept_recordings: int
    played_notes: int
    scored_classes: int
    covering_share: float
    median_duration_s: float
    stored_seconds: float
    median_required_s: float
    rungs: dict[str, int]
    median_useful_rate_hz: float

    @classmethod
    def of(cls, reduction: ReductionDocument) -> ReductionReadings:
        """The readings ``reduction`` states."""
        kept = reduction.recordings
        return cls(
            listed_recordings=reduction.listed_recordings,
            kept_recordings=reduction.kept_recordings,
            played_notes=reduction.played_notes,
            scored_classes=reduction.scored_classes,
            covering_share=_share(record.covers_material for record in kept),
            median_duration_s=_median([record.duration_s for record in kept]),
            stored_seconds=float(sum(record.duration_s for record in kept)),
            median_required_s=_median([record.required_duration_s for record in kept]),
            rungs=dict(Counter(str(grid.stored.target_rate) for grid in reduction.grids)),
            median_useful_rate_hz=_median([grid.useful_rate_hz for grid in reduction.grids]),
        )

    @classmethod
    def read(cls, reduction_json: Path) -> ReductionReadings:
        """The readings the reduced dataset's own ``reduction.json`` states."""
        document = ReducedDocument.model_validate_json(reduction_json.read_text(encoding="utf-8"))
        return cls.of(document.reduction)


class PlanReadings(Readings):
    """What one allocation decided: its objective, what it stored, and what the budget went on."""

    strategy: str
    objective: float
    energy_exponent: float
    samples: int
    instruments: int
    layers: int
    budget_bytes: int
    used_bytes: int
    fill: float
    keys_played: int
    keys_answered: int
    looped_share: float
    compressed_share: float
    stored_seconds: float
    median_zone_keys: float
    widest_zone_keys: int
    rungs: dict[str, int]
    depths: dict[str, int]
    median_hull_size: float
    reserve_bytes_per_sample: int | None
    objective_uncapped: float | None
    widest_envelope_drift_db: float | None

    @classmethod
    def read(cls, plan_json: Path) -> PlanReadings:
        """The readings ``plan_json`` states, whichever strategy wrote it.

        A grouped plan states the keys each zone covers and an ungrouped one keeps a sample per key, so the
        zone widths read alike and a run of either shape lands one record in the store.
        """
        plan = PlanDocument.model_validate_json(plan_json.read_text(encoding="utf-8"))
        zones = plan.zones or ()
        pitches = plan.pitches or ()
        items: list[EncodingRecord] = [*zones, *pitches]
        widths = [len(zone.keys) for zone in zones] or [_ONE_KEY] * len(pitches)
        return cls(
            strategy=plan.strategy,
            objective=plan.objective,
            energy_exponent=plan.energy_exponent,
            samples=len(items),
            instruments=len(plan.instruments),
            layers=len({instrument.layer for instrument in plan.instruments}),
            budget_bytes=plan.budget.sample_budget_bytes,
            used_bytes=plan.budget.used_bytes,
            fill=plan.budget.used_bytes / plan.budget.sample_budget_bytes,
            keys_played=plan.keyboard.played,
            keys_answered=plan.keyboard.answered,
            looped_share=_share(item.loop_index is not _UNLOOPED for item in items),
            compressed_share=_share(item.compress for item in items),
            stored_seconds=float(sum(item.frames / item.target_rate for item in items)),
            median_zone_keys=_median(widths),
            widest_zone_keys=max(widths, default=0),
            rungs=dict(Counter(str(item.target_rate) for item in items)),
            depths=dict(Counter(str(item.depth_bits) for item in items)),
            median_hull_size=_median([item.hull_size for item in items]),
            reserve_bytes_per_sample=plan.reserve.bytes_per_sample if plan.reserve else None,
            objective_uncapped=plan.reserve.objective_uncapped if plan.reserve else None,
            widest_envelope_drift_db=max(
                (instrument.envelope_drift_db for instrument in plan.instruments), default=None
            ),
        )


class MeasuredReadings(Readings):
    """What a plan measures once written: the objective on the samples the module carries, and its terms."""

    objective: float
    plan_objective: float
    written_gap: float
    breakdown: dict[str, float]
    segmental_snr_db: float
    si_sdr_db: float
    loudness_delta_lu: float

    @classmethod
    def read(cls, metrics_json: Path) -> MeasuredReadings:
        """The readings ``metrics_json`` states.

        The per-term breakdown and the diagnostics are weighted by playing time, which is the share of the
        material each scored class stands for; the objective beside them carries the objective's own
        weighting, so the two read together say what moved and how much of the plan it moved.
        """
        document = MetricsDocument.model_validate_json(metrics_json.read_text(encoding="utf-8"))
        events = [event for note in document.notes for event in note.events]
        weights = [event.weight for event in events]
        return cls(
            objective=document.objective,
            plan_objective=document.plan_objective,
            written_gap=document.objective - document.plan_objective,
            breakdown={
                term: _weighted([event.breakdown.get(term) for event in events], weights) for term in _terms(document)
            },
            segmental_snr_db=_weighted([event.diagnostics.get("segmental_snr_db") for event in events], weights),
            si_sdr_db=_weighted([event.diagnostics.get("si_sdr_db") for event in events], weights),
            loudness_delta_lu=_weighted([event.diagnostics.get("loudness_delta_lu") for event in events], weights),
        )


def _terms(document: MetricsDocument) -> tuple[str, ...]:
    """The component metrics the run measured, named in the order the first scored class states them."""
    for note in document.notes:
        for event in note.events:
            return tuple(event.breakdown)

    return ()
