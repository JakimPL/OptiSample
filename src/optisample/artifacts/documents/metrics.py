from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from optisample.artifacts.serialize import Frozen
from optisample.dsp.surrogate import StoredSample
from optisample.music import note_name
from optisample.optimize.tasks import EvalContext, PitchTask, score_events


class RepresentativeEventRecord(Frozen):
    """The single event chosen as a pitch's audible representative (the A/B render)."""

    velocity: int
    duration_s: float


class EventMetricRecord(Frozen):
    """One scored note class of a pitch: its stored volume and the surrogate fidelity + sub-scores it earned.

    ``velocity`` names the loudest note the class covers and ``duration_s`` the length it was scored
    over; ``weight`` is the playing time of every note the class stands for. A reading with no finite
    value behind it -- the loudness of a note too quiet to gate, say -- stands as ``None``, which is how
    :func:`~optisample.artifacts.serialize.write_json` states it and therefore how the written document
    reads back.
    """

    velocity: int
    duration_s: float
    weight: float
    volume: int
    fidelity: float
    breakdown: Mapping[str, float | None]
    diagnostics: Mapping[str, float | None]


class NoteMetricRecord(Frozen):
    """Every scored event of one covered pitch, plus which sample served it and its objective share.

    ``layer`` names the velocity band these events fall in, so a key played across several dynamics
    holds one record per layer, each stating the share of the objective that band's own notes carry. It
    is also the folder the pair's A/B WAVs were written under.
    """

    pitch: int
    note: str
    layer: str
    served_by: str
    representative: int
    weight: float
    mean_distortion: float
    objective_contribution: float
    render_source: str
    render_rate: int
    representative_event: RepresentativeEventRecord
    events: list[EventMetricRecord]


class MetricsDocument(Frozen):
    """Per-note surrogate fidelity for one strategy; ``objective`` reproduces the plan's objective."""

    strategy: str
    instrument_id: str
    sample_rate: int
    objective: float
    plan_objective: float
    notes: list[NoteMetricRecord]


@dataclass(frozen=True)
class RenderedNote:
    """The A/B-render outcome and provenance for one note: which sample served it and what it rendered."""

    served_by: str
    layer: str
    representative: int
    source: str
    rate: int
    event: RepresentativeEventRecord


def event_records(stored: StoredSample, task: PitchTask, context: EvalContext) -> tuple[list[EventMetricRecord], float]:
    """Score every event of one pitch; return per-event records and their weighted-fidelity sum.

    Consumes the same :func:`~optisample.optimize.tasks.score_events` stream the optimizer sums into its
    objective, so the returned contribution is exactly this pitch's share of ``plan.objective``.
    """
    records: list[EventMetricRecord] = []
    contribution = 0.0
    for score in score_events(stored, task, context):
        contribution += score.weighted_fidelity
        records.append(
            EventMetricRecord(
                velocity=score.event.velocity,
                duration_s=score.event.duration_s,
                weight=score.event.weight,
                volume=score.event.volume,
                fidelity=score.report.fidelity,
                breakdown=score.report.breakdown,
                diagnostics=score.report.diagnostics,
            )
        )
    return records, contribution


def note_record(
    task: PitchTask, events: Sequence[EventMetricRecord], contribution: float, rendered: RenderedNote
) -> NoteMetricRecord:
    """Assemble one pitch's metric record from its scored events and the render that was written."""
    return NoteMetricRecord(
        pitch=task.pitch,
        note=note_name(task.pitch),
        layer=rendered.layer,
        served_by=rendered.served_by,
        representative=rendered.representative,
        weight=task.weight,
        mean_distortion=contribution / task.objective_weight if task.objective_weight > 0.0 else 0.0,
        objective_contribution=contribution,
        render_source=rendered.source,
        render_rate=rendered.rate,
        representative_event=rendered.event,
        events=list(events),
    )


def metrics_document(
    strategy: str,
    instrument_id: str,
    sample_rate: int,
    plan_objective: float,
    notes: Sequence[NoteMetricRecord],
) -> MetricsDocument:
    """The per-note metrics document; its ``objective`` sums the notes back to ``plan.objective``."""
    return MetricsDocument(
        strategy=strategy,
        instrument_id=instrument_id,
        sample_rate=sample_rate,
        objective=sum(note.objective_contribution for note in notes),
        plan_objective=plan_objective,
        notes=list(notes),
    )
