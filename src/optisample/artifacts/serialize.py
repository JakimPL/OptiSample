import json
import math
from collections.abc import Mapping, Sequence
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
from optisample.dsp.decay import NO_DECAY, LinearDecay
from optisample.dsp.loop import Loop, LoopQuality
from optisample.dsp.surrogate import NO_LOOP, SettledLoop, StoredSample
from optisample.keys import SampleKey
from optisample.loop.settle import RejectedLoop, Settlement, StoredLoop
from optisample.music import note_name
from optisample.optimize.export.build import instrument_name
from optisample.optimize.export.coverage import KeyCoverage
from optisample.optimize.export.envelope import decay_dispersion, shared_decay
from optisample.optimize.layers.slots import InstrumentSlot, SlotLayout
from optisample.optimize.plans import (
    GroupedInstrumentPlan,
    InstrumentPlan,
    SampleReserve,
    SampleUnit,
    StrategyPlan,
)
from optisample.optimize.reduce.summary import ReductionSummary
from optisample.optimize.reduce.trim import RecordingScreen
from optisample.optimize.tasks import EvalContext, PitchTask, score_events
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.module.size import SizeReport

_OPTIONAL_HEAD: Final = ("method", "pitches", "reserve", "zones")
NO_DRIFT: Final = 0.0  # what one envelope costs an instrument whose samples state no decline to share


class Frozen(BaseModel):
    """Base for every artifact document: immutable and rejecting unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class LoopRecord(Frozen):
    """The loop actually stored after re-encoding: a half-open ``[start, end)`` frame range."""

    start: int
    end: int


class DecayRecord(Frozen):
    """The ramp a looped sample is played down by: when it falls, and how far.

    Seconds run from the note's onset, ``start_s`` sitting where the stored material ends, and
    ``final_gain`` is the share of the loop's own level a note still sounds at once the ramp is through.
    """

    start_s: float
    end_s: float
    final_gain: float


class AnchorRecord(Frozen):
    """One measured velocity anchor of the loudness-matched velocity->volume map."""

    velocity: int
    loudness_lufs: float
    volume: int


class VelocityMapDocument(Frozen):
    """The velocity->volume map: its anchors and the full 0..127 lookup table."""

    reference_volume: int
    anchors: list[AnchorRecord]
    volumes: list[int]


class BudgetRecord(Frozen):
    """The byte accounting for one plan: the budgets it was given and what it spent."""

    module_budget_bytes: int
    sample_budget_bytes: int
    used_bytes: int
    module_bytes: int


class ModuleSizeRecord(Frozen):
    """What the written module occupies, split by what spends the bytes.

    The budget accounts for the instrument's footprint alone, so these totals also carry the audition
    material the module plays -- which is why they exceed :attr:`BudgetRecord.module_bytes`.
    """

    total_bytes: int
    header_bytes: int
    pcm_bytes: int
    pattern_bytes: int


class EncodingRecord(Frozen):
    """The stored-encoding block both strategies share: chosen params, geometry, cost and hull size.

    ``loop`` and ``decay`` are read off the sample as re-encoding actually stored it, so they state the
    loop a player wraps on and the ramp it is brought down by rather than what the sweep asked for. The
    plan items (:class:`PitchItemRecord`, :class:`ZoneItemRecord`) inherit these fields so the block
    appears once per item, flattened alongside the item's own leading fields.
    """

    target_rate: int
    depth_bits: int
    compress: bool
    trim_s: float | None
    loop: LoopRecord | None
    decay: DecayRecord | None
    frames: int
    stored_bytes: int
    distortion: float
    hull_size: int


class _PitchHead(Frozen):
    """The leading fields of an ungrouped item: one kept key and how much it is played."""

    pitch: int
    note: str
    weight: float
    representative_velocity: int


class KeyboardRecord(Frozen):
    """What of the format's keyboard the written instruments answer.

    ``numbered`` is the keys the format offers, ``played`` the keys the material reached, and
    ``answered`` how many resolve to a sample once the rest are filled from the recording nearest them,
    so a reader sees the stretch an instrument plays over beside the stretch it was recorded over.
    """

    numbered: int
    played: int
    answered: int


class InstrumentRecord(Frozen):
    """One instrument the plan is written as, summed over the samples it owns.

    ``index`` is the instrument number the written pattern names for a note this record answers, and
    ``layer`` the position in the plan's velocity split a zone's own ``layer`` states. A band a format
    writes as several instruments states each one's own stretch of keyboard in ``lowest_pitch`` and
    ``highest_pitch``, which are the keys its samples were stored for. ``keys`` counts those keys and
    ``objective_share`` what their notes carry of the plan's objective, so the records add up to what the
    whole plan covers, stores and scores.

    ``envelope_drift_db`` is the widest decibel gap the one volume envelope this instrument carries leaves
    a sample of its own by the end of that sample's ramp. A format gives the envelope to the instrument
    rather than the sample, so keys declining at different rates share one curve, and this states what
    that costs the worst of them -- the reading that says whether the instrument is worth splitting.
    """

    index: int
    name: str
    layer: int
    band: str
    lowest_velocity: int
    highest_velocity: int
    lowest_pitch: int | None
    highest_pitch: int | None
    keys: int
    samples: int
    stored_bytes: int
    weight: float
    objective_share: float
    envelope_drift_db: float


class ReserveRecord(Frozen):
    """The sample cap a grouped plan was held to, and the charge per stored sample that held it there.

    ``bytes_per_sample`` is what each stored sample was priced above the bytes it occupies, so a plan
    already inside its cap records nothing charged. ``objective_uncapped`` is what the same budget scored
    at that price, which states what meeting the cap was worth.
    """

    cap: int
    bytes_per_sample: int
    objective_uncapped: float


class _ZoneHead(Frozen):
    """The leading fields of a grouped item: the velocity band it answers for and the keys it covers."""

    layer: int
    keys: list[int]
    pitches: list[int]
    representative: int
    representative_velocity: int
    weight: float


class PitchItemRecord(EncodingRecord, _PitchHead):
    """One kept pitch: its identity and the encoding chosen for its sample."""


class ZoneItemRecord(EncodingRecord, _ZoneHead):
    """One pitch zone: the keys it serves and the encoding chosen for its representative sample."""


class KeptRecordingRecord(Frozen):
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


class StoredFormatRecord(Frozen):
    """The format the reduction settled for a pitch's stored sample, from the recording's own content.

    ``compress`` states whether the sample runs through dynamics on the way to the quantizer, which a
    shallow depth asks for and a deep one leaves alone.
    """

    target_rate: int
    depth_bits: int
    compress: bool


class LoopQualityRecord(Frozen):
    """What measuring a loop said about it: the wrap it makes, the level it holds, the timbre it holds on to.

    ``seam_step`` reads the wrap in units of the loop region's own frame-to-frame motion,
    ``level_drift_db`` the fall across the region that holding it at one level had to flatten, and
    ``spectral_distance`` the decibel distance between the loop's timbre and the material past it.
    """

    seam_step: float
    level_drift_db: float
    spectral_distance: float


class SettledLoopRecord(Frozen):
    """The loop one recording is stored around, in frames of the recording the stage wrote beside this.

    Frames are what a player wraps between and seconds are where a listener hears it, so both are stated.
    ``decay`` is the ramp a note held past the stored span falls on, where the recording states one to make.
    """

    start: int
    end: int
    start_s: float
    end_s: float
    quality: LoopQualityRecord
    decay: DecayRecord | None


class RejectedLoopRecord(Frozen):
    """A candidate the ladder climbed past, and the gate it fell outside of.

    Reading these says why a recording ended up stored around a later loop, or around none: each entry is
    a cheaper loop that was measured and found wanting on ``gate``.
    """

    start_s: float
    end_s: float
    quality: LoopQualityRecord
    gate: str


class RecordingLoopsRecord(Frozen):
    """What the loop stage decided for one recording: the loop it keeps, and the ones it climbed past.

    ``cc`` carries the controller buckets its identity was keyed under, so a reader rebuilds the same
    :class:`~optisample.keys.SampleKey` the audio is held under. ``search_s`` is the stretch candidates were
    measured over, which is the longest note the material plays at this pitch.
    """

    key: str
    pitch: int
    note: str
    velocity: int
    cc: list[tuple[int, int]]
    search_s: float
    stored: SettledLoopRecord | None
    rejected: list[RejectedLoopRecord]


class LoopsDocument(Frozen):
    """Every loop one instrument's recordings were settled around, beside the dataset they were read from.

    ``sample_rate`` is the rate the recordings were analysed at, which the frames in every
    :class:`SettledLoopRecord` are counted in. Reading this back is what lets a later stage store the loops
    a run already settled, hand-tuned or as they came.
    """

    instrument_id: str
    sample_rate: int
    recordings: list[RecordingLoopsRecord]


class NarrowedGridRecord(Frozen):
    """What the pre-pass settled for one pitch: the band it read and the format it stores at.

    ``useful_rate_hz`` is the rate the recording's own content asks for and ``stored`` the ladder rung
    reaching it, so a reader sees both the measurement and the format it named. ``swept`` counts the
    encodings the sweep then runs for this pitch, which is that one format over the stored spans it offers.
    """

    pitch: int
    note: str
    useful_rate_hz: float
    stored: StoredFormatRecord
    swept: int


class ReductionDocument(Frozen):
    """The pre-optimization stage's decisions: what survived ingest and how small the search space got.

    The ``*_recordings``/``*_notes`` counts are the before side of each reduction axis; ``recordings`` and
    ``grids`` are the after side, per identity and per played pitch.
    """

    listed_recordings: int
    kept_recordings: int
    played_notes: int
    scored_classes: int
    recordings: list[KeptRecordingRecord]
    grids: list[NarrowedGridRecord]


class WrittenSampleRecord(Frozen):
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


class ScreenRecord(Frozen):
    """What admitting the recordings cost: the ones left out, and the material that left unplayable.

    ``silenced`` names each recording whose peak stayed under the configured silence floor, so a reader
    sees which slots the dataset holds no audio for. ``unplayable`` are the pitches those losses stripped
    of every recording, and ``dropped_notes`` how many played notes went with them.
    """

    silenced: list[str]
    unplayable: list[int]
    dropped_notes: int


class ReducedDocument(Frozen):
    """What one reduce run produced: the dataset it wrote and the decisions that shaped it.

    ``dedupe_key`` is the identity the survivors were kept under, so a run reading this dataset back
    states the projection it already stands at. ``sample_rate`` is the analysis rate every survivor was
    written at, which is the rate the bands in ``reduction`` were measured over. ``screen`` states
    what the dataset leaves out, beside the ``samples`` it holds.
    """

    instrument_id: str
    dedupe_key: DedupeKey
    sample_rate: int
    samples: list[WrittenSampleRecord]
    screen: ScreenRecord
    reduction: ReductionDocument


class PlanDocument(Frozen):
    """One optimized plan for either strategy: budgets, the velocity map, the layers and the kept items.

    ``method`` is recorded only for the ungrouped strategy, ``reserve`` only for the grouped one, and
    ``pitches`` and ``zones`` are mutually exclusive. The optional-head fields a strategy leaves unset are
    dropped at serialization, so each strategy writes exactly the keys that apply to it. A grouped plan's
    ``reserve`` states the sample cap it was held to and what holding it there cost.
    ``instruments`` is what the plan is written as, one
    entry per instrument the module numbers, so a plan keeping one recording per key in one band reports
    its single instrument. ``energy_exponent`` states how steeply each note's own energy scaled its
    distortion, which is what settles whether two documents' objectives may be compared.
    """

    strategy: str
    instrument_id: str
    method: str | None = None
    objective: float
    energy_exponent: float
    budget: BudgetRecord
    module: ModuleSizeRecord
    keyboard: KeyboardRecord
    reduction: ReductionDocument
    velocity_map: VelocityMapDocument
    instruments: list[InstrumentRecord]
    pitches: list[PitchItemRecord] | None = None
    reserve: ReserveRecord | None = None
    zones: list[ZoneItemRecord] | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_head(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        return {key: value for key, value in data.items() if not (key in _OPTIONAL_HEAD and value is None)}


class RepresentativeEventRecord(Frozen):
    """The single event chosen as a pitch's audible representative (the A/B render)."""

    velocity: int
    duration_s: float


class EventMetricRecord(Frozen):
    """One scored note class of a pitch: its stored volume and the surrogate fidelity + sub-scores it earned.

    ``velocity`` names the loudest note the class covers and ``duration_s`` the length it was scored
    over; ``weight`` is the playing time of every note the class stands for. A reading with no finite
    value behind it -- the loudness of a note too quiet to gate, say -- stands as ``None``, which is how
    :func:`write_json` states it and therefore how the written document reads back.
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


def json_text(document: BaseModel) -> str:
    """A document as pretty JSON, coercing numpy/non-finite values to JSON-safe Python.

    A document stored inside an archive is written from its text rather than from a file, so a manifest
    travelling with the instruments it names reads exactly as one written beside them.
    """
    return json.dumps(_json_safe(document.model_dump()), indent=2, allow_nan=False) + "\n"


def write_json(path: Path, document: BaseModel) -> None:
    """Serialize a document to pretty JSON at ``path`` (see :func:`json_text`)."""
    path.write_text(json_text(document), encoding="utf-8")


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


def screen_record(screen: RecordingScreen) -> ScreenRecord:
    """The load-time screen as a document, so a dataset states what it left behind as well as what it holds."""
    return ScreenRecord(
        silenced=[key.label for key in screen.silenced],
        unplayable=list(screen.unplayable),
        dropped_notes=screen.dropped_notes,
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
                stored=StoredFormatRecord(
                    target_rate=grid.stored.target_rate,
                    depth_bits=grid.stored.depth_bits,
                    compress=grid.stored.compress,
                ),
                swept=len(grid.encodings),
            )
            for grid in reduction.grids
        ],
    )


def _loop_record(loop: Loop | None) -> LoopRecord | None:
    """The loop a stored sample wraps on (``{start, end}``), where it holds one."""
    return None if loop is None else LoopRecord(start=loop.start, end=loop.end)


def _decay_record(decay: LinearDecay | None) -> DecayRecord | None:
    """The ramp a stored sample is played down by, where the recording states one to make."""
    if decay is None:
        return None

    return DecayRecord(start_s=decay.start_s, end_s=decay.end_s, final_gain=decay.final_gain)


def _loop_quality_record(quality: LoopQuality) -> LoopQualityRecord:
    """What measuring one candidate said about it, read out into the document's own fields."""
    return LoopQualityRecord(
        seam_step=quality.seam_step,
        level_drift_db=quality.level_drift_db,
        spectral_distance=quality.spectral_distance,
    )


def _settled_loop_record(stored: StoredLoop, sample_rate: int) -> SettledLoopRecord:
    """The loop a recording is stored around, stated in frames and in seconds alike."""
    return SettledLoopRecord(
        start=stored.loop.start,
        end=stored.loop.end,
        start_s=stored.loop.start / sample_rate,
        end_s=stored.loop.end / sample_rate,
        quality=_loop_quality_record(stored.quality),
        decay=_decay_record(stored.decay),
    )


def _rejected_loop_record(rejected: RejectedLoop, sample_rate: int) -> RejectedLoopRecord:
    """One candidate the ladder climbed past, placed in the note by the second."""
    return RejectedLoopRecord(
        start_s=rejected.loop.start / sample_rate,
        end_s=rejected.loop.end / sample_rate,
        quality=_loop_quality_record(rejected.quality),
        gate=rejected.gate.value,
    )


def loops_document(
    instrument_id: str,
    sample_rate: int,
    settlements: Mapping[SampleKey, Settlement],
    searched: Mapping[SampleKey, float],
) -> LoopsDocument:
    """What the loop stage decided for one instrument, as the document written beside its dataset.

    Recordings come out in key order, so the document a run writes reads the same way twice over the same
    dataset. ``searched`` states the stretch each recording's candidates were measured over.
    """
    return LoopsDocument(
        instrument_id=instrument_id,
        sample_rate=sample_rate,
        recordings=[
            RecordingLoopsRecord(
                key=key.label,
                pitch=key.pitch,
                note=note_name(key.pitch),
                velocity=key.velocity,
                cc=list(key.cc),
                search_s=searched[key],
                stored=(
                    None
                    if settlements[key].stored is None
                    else _settled_loop_record(_stored_of(settlements[key]), sample_rate)
                ),
                rejected=[_rejected_loop_record(rejected, sample_rate) for rejected in settlements[key].rejected],
            )
            for key in sorted(settlements)
        ],
    )


def _stored_of(settlement: Settlement) -> StoredLoop:
    """The loop ``settlement`` kept, for a caller that has already established it kept one.

    Raises:
        ValueError: if the settlement kept none, which a caller reaching here has already ruled out.
    """
    if settlement.stored is None:
        raise ValueError("settlement stores no loop")

    return settlement.stored


def _settled_key(record: RecordingLoopsRecord) -> SampleKey:
    """The identity one document entry names, rebuilt as the audio map holds it."""
    return SampleKey(pitch=record.pitch, velocity=record.velocity, cc=tuple(record.cc))


def read_loops(path: Path) -> LoopsDocument:
    """The loops document at ``path``, validated against its own shape."""
    return LoopsDocument.model_validate_json(path.read_text(encoding="utf-8"))


def settled_loops(document: LoopsDocument) -> dict[SampleKey, SettledLoop | None]:
    """The loops a document states, in the shape every encode reads them through.

    Frames come back as they were written, so a document read beside the dataset it was measured over names
    the same stretches of the same recordings.
    """
    return {
        _settled_key(record): (
            NO_LOOP
            if record.stored is None
            else SettledLoop(
                loop=Loop(start=record.stored.start, end=record.stored.end),
                decay=_read_decay(record.stored.decay),
            )
        )
        for record in document.recordings
    }


def _read_decay(record: DecayRecord | None) -> LinearDecay | None:
    """The ramp one document entry states, where it states one."""
    if record is None:
        return NO_DECAY

    return LinearDecay(start_s=record.start_s, end_s=record.end_s, final_gain=record.final_gain)


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


def _keyboard_record(coverage: KeyCoverage) -> KeyboardRecord:
    return KeyboardRecord(
        numbered=coverage.numbered,
        played=coverage.played,
        answered=coverage.answered,
    )


def _encoding_record(unit: SampleUnit, stored: StoredSample) -> EncodingRecord:
    return EncodingRecord(
        target_rate=unit.params.target_rate,
        depth_bits=unit.params.depth_bits,
        compress=unit.params.compress,
        trim_s=unit.params.trim_s,
        loop=_loop_record(stored.loop),
        decay=_decay_record(stored.decay),
        frames=unit.frames,
        stored_bytes=unit.stored_bytes,
        distortion=unit.distortion,
        hull_size=unit.hull_size,
    )


def _pitch_item(unit: SampleUnit, stored: StoredSample) -> PitchItemRecord:
    return PitchItemRecord(
        pitch=unit.representative,
        note=note_name(unit.representative),
        weight=unit.weight,
        representative_velocity=unit.representative_key.velocity,
        **_encoding_record(unit, stored).model_dump(),
    )


def _envelope_drift_db(slot: InstrumentSlot, encoded: Sequence[StoredSample]) -> float:
    """What the one envelope this slot carries costs the sample of its own it suits worst."""
    decays = [encoded[index].decay for index in slot.samples]
    shared = shared_decay(decays)
    return NO_DRIFT if shared is None else decay_dispersion(decays, shared)


def _instrument_record(
    index: int,
    slot: InstrumentSlot,
    name: str,
    encoded: Sequence[StoredSample],
) -> InstrumentRecord:
    pitches = slot.pitches
    return InstrumentRecord(
        index=index,
        name=name,
        layer=slot.layer,
        band=slot.band.label,
        lowest_velocity=slot.band.lowest,
        highest_velocity=slot.band.highest,
        lowest_pitch=pitches[0] if pitches else None,
        highest_pitch=pitches[-1] if pitches else None,
        keys=slot.keys,
        samples=len(slot.samples),
        stored_bytes=slot.stored_bytes,
        weight=slot.weight,
        objective_share=slot.objective_share,
        envelope_drift_db=_envelope_drift_db(slot, encoded),
    )


def _instrument_records(
    plan: StrategyPlan,
    layout: SlotLayout,
    encoded: Sequence[StoredSample],
) -> list[InstrumentRecord]:
    """One record per written instrument, named exactly as the module's own instrument list names it."""
    return [
        _instrument_record(index, slot, instrument_name(plan.instrument_id, layout, index), encoded)
        for index, slot in enumerate(layout.slots)
    ]


def _reserve_record(reserve: SampleReserve) -> ReserveRecord:
    return ReserveRecord(
        cap=reserve.cap,
        bytes_per_sample=reserve.bytes_per_sample,
        objective_uncapped=reserve.objective_uncapped,
    )


def _zone_item(unit: SampleUnit, stored: StoredSample) -> ZoneItemRecord:
    return ZoneItemRecord(
        layer=unit.layer,
        keys=[unit.keys[0], unit.keys[-1]],
        pitches=list(unit.keys),
        representative=unit.representative,
        representative_velocity=unit.representative_key.velocity,
        weight=unit.weight,
        **_encoding_record(unit, stored).model_dump(),
    )


def plan_document(
    plan: InstrumentPlan | GroupedInstrumentPlan,
    encoded: Sequence[StoredSample],
    size: SizeReport,
    coverage: KeyCoverage,
    layout: SlotLayout,
) -> PlanDocument:
    """One plan document for either strategy; ``encoded`` holds the re-encoded samples, in plan order.

    The plan's :meth:`~optisample.optimize.plans.StrategyPlan.sample_units` supplies the shared encoding
    block for every item; only the leading fields (a pitch vs. a zone, and whether a ``method`` is
    recorded) differ, selected by narrowing on the plan's strategy. ``size`` is what the module the plan
    exports to actually occupies, ``coverage`` what its keymaps answer of the format's keyboard, and
    ``layout`` the instruments it was written as.
    """
    units = plan.sample_units()
    budget = _budget_record(plan)
    module = _module_size_record(size)
    keyboard = _keyboard_record(coverage)
    reduction = reduction_document(plan.reduction)
    velocity_map = _velocity_map_document(plan.velocity_map)
    instruments = _instrument_records(plan, layout, encoded)
    if plan.strategy == "grouped":
        return PlanDocument(
            strategy="grouped",
            instrument_id=plan.instrument_id,
            objective=plan.objective,
            energy_exponent=plan.energy_exponent,
            budget=budget,
            module=module,
            keyboard=keyboard,
            reduction=reduction,
            velocity_map=velocity_map,
            instruments=instruments,
            reserve=_reserve_record(plan.reserve),
            zones=[_zone_item(unit, stored) for unit, stored in zip(units, encoded)],
        )
    return PlanDocument(
        strategy="ungrouped",
        instrument_id=plan.instrument_id,
        method=plan.method,
        objective=plan.objective,
        energy_exponent=plan.energy_exponent,
        budget=budget,
        module=module,
        keyboard=keyboard,
        reduction=reduction,
        velocity_map=velocity_map,
        instruments=instruments,
        pitches=[_pitch_item(unit, stored) for unit, stored in zip(units, encoded)],
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
    layer: str
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
