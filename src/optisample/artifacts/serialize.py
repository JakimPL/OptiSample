import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from optisample.config.reduce import DedupeKey
from optisample.dsp.loop import Loop
from optisample.dsp.surrogate import StoredSample
from optisample.music import note_name
from optisample.optimize.plans import (
    GroupedInstrumentPlan,
    InstrumentPlan,
    SampleUnit,
    StrategyPlan,
)
from optisample.optimize.reduce.summary import ReductionSummary
from optisample.optimize.tasks import EvalContext, PitchTask, score_events
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.module.size import SizeReport

_OPTIONAL_HEAD: Final = ("method", "pitches", "zones")


class _Frozen(BaseModel):
    """Base for every artifact document: immutable and rejecting unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class LoopRecord(_Frozen):
    """The loop actually stored after re-encoding: a half-open ``[start, end)`` frame range."""

    start: int
    end: int


class AnchorRecord(_Frozen):
    """One measured velocity anchor of the loudness-matched velocity->volume map."""

    velocity: int
    loudness_lufs: float
    volume: int


class VelocityMapDocument(_Frozen):
    """The velocity->volume map: its anchors and the full 0..127 lookup table."""

    reference_volume: int
    anchors: list[AnchorRecord]
    volumes: list[int]


class BudgetRecord(_Frozen):
    """The byte accounting for one plan: the budgets it was given and what it spent."""

    module_budget_bytes: int
    sample_budget_bytes: int
    used_bytes: int
    module_bytes: int


class ModuleSizeRecord(_Frozen):
    """What the written module occupies, split by what spends the bytes.

    The budget accounts for the instrument's footprint alone, so these totals also carry the audition
    material the module plays -- which is why they exceed :attr:`BudgetRecord.module_bytes`.
    """

    total_bytes: int
    header_bytes: int
    pcm_bytes: int
    pattern_bytes: int


class EncodingRecord(_Frozen):
    """The stored-encoding block both strategies share: chosen params, geometry, cost and hull size.

    ``loop`` is the loop *actually stored* after re-encoding (not merely the one the sweep requested).
    The plan items (:class:`PitchItemRecord`, :class:`ZoneItemRecord`) inherit these fields so the block
    appears once per item, flattened alongside the item's own leading fields.
    """

    target_rate: int
    depth_bits: int
    trim_s: float | None
    loop: LoopRecord | None
    frames: int
    stored_bytes: int
    distortion: float
    hull_size: int


class _PitchHead(_Frozen):
    """The leading fields of an ungrouped item: one kept key and how much it is played."""

    pitch: int
    note: str
    weight: float
    representative_velocity: int


class _ZoneHead(_Frozen):
    """The leading fields of a grouped item: the key span the zone covers and its representative."""

    keys: list[int]
    pitches: list[int]
    representative: int
    representative_velocity: int
    weight: float


# Listing the head base last puts its fields first and EncodingRecord's after them (MRO field order),
# reproducing the flat ``{head..., encoding...}`` layout the dump tree writes.
class PitchItemRecord(EncodingRecord, _PitchHead):
    """One kept pitch: its identity and the encoding chosen for its sample."""


class ZoneItemRecord(EncodingRecord, _ZoneHead):
    """One pitch zone: the keys it serves and the encoding chosen for its representative sample."""


class KeptRecordingRecord(_Frozen):
    """One recording that survived deduplication, measured against the material its pitch plays.

    ``covers_material`` is false when the survivor runs shorter than its pitch's longest note, in which
    case the objective scores it over as much of the note as was recorded.
    """

    key: str
    pitch: int
    velocity: int
    duration_s: float
    required_duration_s: float
    covers_material: bool


class ShortlistedEncodingRecord(_Frozen):
    """One encoding the bandwidth pre-pass left in the running for a pitch's stored sample."""

    target_rate: int
    depth_bits: int
    loop: bool


class NarrowedGridRecord(_Frozen):
    """What the pre-pass left of one pitch's encoding grid, and the stored band that bounds it."""

    pitch: int
    note: str
    useful_rate_hz: float
    shortlist: list[ShortlistedEncodingRecord]


class ReductionDocument(_Frozen):
    """The pre-optimization stage's decisions: what survived ingest and how small the search space got.

    The three ``*_recordings``/``*_notes``/``grid_size`` counts are the before side of each reduction
    axis; ``recordings`` and ``grids`` are the after side, per identity and per played pitch.
    """

    listed_recordings: int
    kept_recordings: int
    played_notes: int
    scored_classes: int
    grid_size: int
    recordings: list[KeptRecordingRecord]
    grids: list[NarrowedGridRecord]


class WrittenSampleRecord(_Frozen):
    """One survivor as a reduced dataset holds it: the file written and the index its notes join on.

    ``index`` leads the filename, which is what a later ingest reads to route each note back to this
    recording. ``frames`` and ``duration_s`` measure what was written, so a dataset trimmed to what the
    material asks for states the length it kept.
    """

    index: int
    key: str
    file: str
    frames: int
    duration_s: float


class ReducedDocument(_Frozen):
    """What one reduce run produced: the dataset it wrote and the decisions that shaped it.

    ``dedupe_key`` is the identity the survivors were kept under, so a run reading this dataset back
    states the projection it already stands at. ``sample_rate`` is the analysis rate every survivor was
    written at, which is the rate the shortlist in ``reduction`` was measured over.
    """

    instrument_id: str
    dedupe_key: DedupeKey
    sample_rate: int
    samples: list[WrittenSampleRecord]
    reduction: ReductionDocument


class PlanDocument(_Frozen):
    """One optimized plan for either strategy: budgets, the velocity map, and the kept items.

    ``method`` is recorded only for the ungrouped strategy; ``pitches`` and ``zones`` are mutually
    exclusive. The optional-head fields a strategy leaves unset are dropped at serialization, so each
    strategy writes exactly the keys that apply to it.
    """

    strategy: str
    instrument_id: str
    method: str | None = None
    objective: float
    budget: BudgetRecord
    module: ModuleSizeRecord
    reduction: ReductionDocument
    velocity_map: VelocityMapDocument
    pitches: list[PitchItemRecord] | None = None
    zones: list[ZoneItemRecord] | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_head(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        return {key: value for key, value in data.items() if not (key in _OPTIONAL_HEAD and value is None)}


class RepresentativeEventRecord(_Frozen):
    """The single event chosen as a pitch's audible representative (the A/B render)."""

    velocity: int
    duration_s: float


class EventMetricRecord(_Frozen):
    """One scored note class of a pitch: its stored volume and the surrogate fidelity + sub-scores it earned.

    ``velocity`` names the loudest note the class covers and ``duration_s`` the length it was scored
    over; ``weight`` is the playing time of every note the class stands for.
    """

    velocity: int
    duration_s: float
    weight: float
    volume: int
    fidelity: float
    breakdown: dict[str, float]
    diagnostics: dict[str, float]


class NoteMetricRecord(_Frozen):
    """Every scored event of one covered pitch, plus which sample served it and its objective share."""

    pitch: int
    note: str
    served_by: str
    representative: int
    weight: float
    mean_distortion: float
    objective_contribution: float
    render_source: str
    render_rate: int
    representative_event: RepresentativeEventRecord
    events: list[EventMetricRecord]


class MetricsDocument(_Frozen):
    """Per-note surrogate fidelity for one strategy; ``objective`` reproduces the plan's objective."""

    strategy: str
    instrument_id: str
    sample_rate: int
    objective: float
    plan_objective: float
    notes: list[NoteMetricRecord]


def _json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars to Python and non-finite floats (silence -> -inf) to null."""
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, document: BaseModel) -> None:
    """Serialize a document to pretty JSON, coercing numpy/non-finite values to JSON-safe Python."""
    path.write_text(json.dumps(_json_safe(document.model_dump()), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write ``text`` verbatim as UTF-8 (the human-readable report and the infeasibility note)."""
    path.write_text(text, encoding="utf-8")


def _velocity_map_document(velocity_map: VelocityVolumeMap) -> VelocityMapDocument:
    reference_volume = max((anchor.volume for anchor in velocity_map.anchors), default=0)
    return VelocityMapDocument(
        reference_volume=reference_volume,
        anchors=[
            AnchorRecord(velocity=anchor.velocity, loudness_lufs=anchor.loudness_lufs, volume=anchor.volume)
            for anchor in velocity_map.anchors
        ],
        volumes=list(velocity_map.volumes),
    )


def reduction_document(reduction: ReductionSummary) -> ReductionDocument:
    """The pre-optimization stage's own outcome as a document, for the plan tree and the reduced dataset.

    Both consumers state the same reduction, so a dataset written by ``reduce`` and a plan produced from
    it report their shared search space in one shape.
    """
    return ReductionDocument(
        listed_recordings=reduction.listed_recordings,
        kept_recordings=reduction.kept_recordings,
        played_notes=reduction.played_notes,
        scored_classes=reduction.scored_classes,
        grid_size=reduction.grid_size,
        recordings=[
            KeptRecordingRecord(
                key=recording.key.label,
                pitch=recording.key.pitch,
                velocity=recording.key.velocity,
                duration_s=recording.duration_s,
                required_duration_s=recording.required_duration_s,
                covers_material=recording.covers_material,
            )
            for recording in reduction.recordings
        ],
        grids=[
            NarrowedGridRecord(
                pitch=grid.pitch,
                note=note_name(grid.pitch),
                useful_rate_hz=grid.useful_rate_hz,
                shortlist=[
                    ShortlistedEncodingRecord(
                        target_rate=params.target_rate,
                        depth_bits=params.depth_bits,
                        loop=params.loop,
                    )
                    for params in grid.shortlist
                ],
            )
            for grid in reduction.grids
        ],
    )


def _loop_record(loop: Loop | None) -> LoopRecord | None:
    """The loop actually stored (``{start, end}``), or ``None`` when the sample was not looped."""
    return None if loop is None else LoopRecord(start=loop.start, end=loop.end)


def _budget_record(plan: StrategyPlan) -> BudgetRecord:
    return BudgetRecord(
        module_budget_bytes=plan.module_budget_bytes,
        sample_budget_bytes=plan.sample_budget_bytes,
        used_bytes=plan.used_bytes,
        module_bytes=plan.module_bytes,
    )


def _module_size_record(size: SizeReport) -> ModuleSizeRecord:
    return ModuleSizeRecord(
        total_bytes=size.total,
        header_bytes=size.headers,
        pcm_bytes=size.pcm,
        pattern_bytes=size.patterns,
    )


def _encoding_record(unit: SampleUnit, loop: Loop | None) -> EncodingRecord:
    return EncodingRecord(
        target_rate=unit.params.target_rate,
        depth_bits=unit.params.depth_bits,
        trim_s=unit.params.trim_s,
        loop=_loop_record(loop),
        frames=unit.frames,
        stored_bytes=unit.stored_bytes,
        distortion=unit.distortion,
        hull_size=unit.hull_size,
    )


def _pitch_item(unit: SampleUnit, loop: Loop | None) -> PitchItemRecord:
    return PitchItemRecord(
        pitch=unit.representative,
        note=note_name(unit.representative),
        weight=unit.weight,
        representative_velocity=unit.representative_key.velocity,
        **_encoding_record(unit, loop).model_dump(),
    )


def _zone_item(unit: SampleUnit, loop: Loop | None) -> ZoneItemRecord:
    return ZoneItemRecord(
        keys=[unit.keys[0], unit.keys[-1]],
        pitches=list(unit.keys),
        representative=unit.representative,
        representative_velocity=unit.representative_key.velocity,
        weight=unit.weight,
        **_encoding_record(unit, loop).model_dump(),
    )


def plan_document(
    plan: InstrumentPlan | GroupedInstrumentPlan,
    loops: Sequence[Loop | None],
    size: SizeReport,
) -> PlanDocument:
    """One plan document for either strategy; ``loops`` are the per-item *stored* loops, in plan order.

    The plan's :meth:`~optisample.optimize.plans.StrategyPlan.sample_units` supplies the shared encoding
    block for every item; only the leading fields (a pitch vs. a zone, and whether a ``method`` is
    recorded) differ, selected by narrowing on the plan's strategy. ``size`` is what the module the plan
    exports to actually occupies.
    """
    units = plan.sample_units()
    budget = _budget_record(plan)
    module = _module_size_record(size)
    reduction = reduction_document(plan.reduction)
    velocity_map = _velocity_map_document(plan.velocity_map)
    if plan.strategy == "grouped":
        return PlanDocument(
            strategy="grouped",
            instrument_id=plan.instrument_id,
            objective=plan.objective,
            budget=budget,
            module=module,
            reduction=reduction,
            velocity_map=velocity_map,
            zones=[_zone_item(unit, loop) for unit, loop in zip(units, loops)],
        )
    return PlanDocument(
        strategy="ungrouped",
        instrument_id=plan.instrument_id,
        method=plan.method,
        objective=plan.objective,
        budget=budget,
        module=module,
        reduction=reduction,
        velocity_map=velocity_map,
        pitches=[_pitch_item(unit, loop) for unit, loop in zip(units, loops)],
    )


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


@dataclass(frozen=True)
class RenderedNote:
    """The A/B-render outcome and provenance for one note: which sample served it and what it rendered."""

    served_by: str
    representative: int
    source: str
    rate: int
    event: RepresentativeEventRecord


def note_record(
    task: PitchTask, events: Sequence[EventMetricRecord], contribution: float, rendered: RenderedNote
) -> NoteMetricRecord:
    """Assemble one pitch's metric record from its scored events and the render that was written."""
    return NoteMetricRecord(
        pitch=task.pitch,
        note=note_name(task.pitch),
        served_by=rendered.served_by,
        representative=rendered.representative,
        weight=task.weight,
        mean_distortion=contribution / task.weight if task.weight > 0.0 else 0.0,
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
