from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from pydantic import SerializerFunctionWrapHandler, model_serializer

from optisample.artifacts.documents.loops import LevelRecord, LoopRecord, level_record, loop_record
from optisample.artifacts.documents.reduction import ReductionDocument, reduction_document
from optisample.artifacts.documents.velocity import VelocityMapDocument, velocity_map_document
from optisample.artifacts.serialize import Frozen
from optisample.dsp.surrogate import StoredSample
from optisample.music import note_name
from optisample.optimize.export.build import instrument_name
from optisample.optimize.export.coverage import KeyCoverage
from optisample.optimize.export.voices import WrittenInstruments
from optisample.optimize.layers.slots import InstrumentSlot
from optisample.optimize.plans import (
    GroupedInstrumentPlan,
    InstrumentPlan,
    SampleReserve,
    SampleUnit,
    StrategyPlan,
)
from trackmod.module.size import SizeReport

_OPTIONAL_HEAD: Final = ("method", "pitches", "reserve", "zones")


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

    ``loop_index`` names which of the loops the stage offered for this recording the allocation bought,
    counting from the cheapest stored span, and is absent for an item storing the span it plays -- so how
    much of a note the budget paid to keep is readable beside what that cost. ``loop`` and ``level`` are
    read off the sample as re-encoding actually stored it, so they state the loop a player wraps on and the
    curve it is brought down by rather than what the sweep asked for. ``carrier`` says the waveform came out
    holding timbre alone, its level travelling on the curve beside it. It is read off the stored sample
    rather than off what the encoding asked for, so a recording whose attack a written curve has no room
    to state (:func:`~optisample.optimize.carrier.clip_envelope`) reads as the recording it was stored as
    however the sweep reached it. The plan items
    (:class:`PitchItemRecord`, :class:`ZoneItemRecord`) inherit these fields so the block appears once per
    item, flattened alongside the item's own leading fields.
    """

    target_rate: int
    depth_bits: int
    compress: bool
    carrier: bool
    trim_s: float | None
    loop_index: int | None
    loop: LoopRecord | None
    level: LevelRecord
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
    a key of its own, over the whole stretch that key is held for. A format gives the envelope to the
    instrument rather than the sample, so keys declining at different rates share one curve, and this
    states what that costs the worst of them -- the reading that says whether the instrument is worth
    splitting (:attr:`~optisample.dsp.trajectory.SharedTrajectory.dispersion_db`).
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
        carrier=not stored.level.transparent,
        trim_s=unit.params.trim_s,
        loop_index=unit.params.loop_index,
        loop=loop_record(stored.loop),
        level=level_record(stored.level),
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


def _instrument_record(
    index: int,
    slot: InstrumentSlot,
    name: str,
    drift_db: float,
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
        envelope_drift_db=drift_db,
    )


def _instrument_records(plan: StrategyPlan, written: WrittenInstruments) -> list[InstrumentRecord]:
    """One record per written instrument, named exactly as the module's own instrument list names it."""
    layout = written.layout
    return [
        _instrument_record(index, slot, instrument_name(plan.instrument_id, layout, index), written.drifts[index])
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
    written: WrittenInstruments,
) -> PlanDocument:
    """One plan document for either strategy; ``encoded`` holds the re-encoded samples, in plan order.

    The plan's :meth:`~optisample.optimize.plans.StrategyPlan.sample_units` supplies the shared encoding
    block for every item; only the leading fields (a pitch vs. a zone, and whether a ``method`` is
    recorded) differ, selected by narrowing on the plan's strategy. ``size`` is what the module the plan
    exports to actually occupies, ``coverage`` what its keymaps answer of the format's keyboard, and
    and ``written`` the instruments it was written as, beside what each of their volume envelopes leaves
    the key of its own it suits worst.
    """
    units = plan.sample_units()
    budget = _budget_record(plan)
    module = _module_size_record(size)
    keyboard = _keyboard_record(coverage)
    reduction = reduction_document(plan.reduction)
    velocity_map = velocity_map_document(plan.velocity_map)
    instruments = _instrument_records(plan, written)
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
