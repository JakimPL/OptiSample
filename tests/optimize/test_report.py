from __future__ import annotations

from dataclasses import replace
from typing import Final

from optisample.dsp.surrogate import EncodingParams
from optisample.keys import SampleKey
from optisample.metrics.size import bytes_to_kib
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.export.coverage import KeyCoverage
from optisample.optimize.knapsack import Allocation, RDCurvePoint, Selection
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import SlotLayout, pack_slots
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans import (
    FIRST_LAYER,
    NO_RESERVE,
    SINGLE_LAYER,
    GroupedInstrumentPlan,
    InstrumentPlan,
    PitchPlan,
    SampleReserve,
    StrategyPlan,
    Zone,
    ZoneOption,
    split_budget,
)
from optisample.optimize.plans.budget import populated_instrument_bytes
from optisample.optimize.reduce.bandwidth import StoredFormat
from optisample.optimize.reduce.summary import (
    KeptRecording,
    NarrowedGrid,
    ReductionSummary,
)
from optisample.optimize.report import (
    format_grouping_report,
    format_reduction_block,
    format_report,
)
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from trackmod.module.size import SizeReport
from trackmod.module.storage import Storage

_MODULE_BYTES = 64 * 1024
_SIZE = SizeReport(patterns=120, pcm=9000, headers=800, largest_pattern=90)
_COVERAGE = KeyCoverage(numbered=120, played=3, answered=120)  # a keyboard answered in full from 3 recordings
_WHOLE_AXIS = VelocityLayers((VelocityBand(0, MIDI_MAX_VELOCITY),))  # one layer answering every dynamic
_ENERGY_EXPONENT = 0.5  # the weighting these fixtures state; the report only ever echoes it back
_WHOLE_TABLE: Final = 255  # samples one instrument reaches in a format numbering them freely
_ONE_SAMPLE_EACH: Final = 1  # the narrowest format imaginable, which writes every stored zone on its own
_RESERVED: Final = 4  # instruments a budget held room for, above the two a plan came out as
_UNCHARGED: Final = SampleReserve(cap=_WHOLE_TABLE, bytes_per_sample=NO_RESERVE, objective_uncapped=0.0)
_CHARGED: Final = SampleReserve(cap=1, bytes_per_sample=512, objective_uncapped=0.05)  # a cap that decided the plan


def _layout(plan: StrategyPlan, per_instrument: int = _WHOLE_TABLE) -> SlotLayout:
    """The instruments the plan is written as, which the report prices one row each."""
    return pack_slots(plan.sample_units(), plan.layers, per_instrument)


def _point(rate: int = 22_050, depth: int = 16, size: int = 5000, distortion: float = 0.1) -> OperatingPoint:
    return OperatingPoint(EncodingParams(target_rate=rate, depth_bits=depth), size, distortion, 2400)


def _ungrouped_plan(
    pitches: tuple[PitchPlan, ...],
    curve: tuple[RDCurvePoint, ...],
    storage: Storage,
    reduction: ReductionSummary,
) -> InstrumentPlan:
    return InstrumentPlan(
        instrument_id="piano",
        budget=split_budget(_MODULE_BYTES / 1024.0, storage, SINGLE_LAYER),
        velocity_map=VelocityVolumeMap(
            tuple(64 for _ in range(128)),
            (VelocityAnchor(50, -20.0, 40), VelocityAnchor(100, -10.0, 64)),  # only the loudest is the reference
        ),
        pitches=pitches,
        allocation=Allocation(tuple(Selection(str(p.pitch), p.weight, p.chosen) for p in pitches), 5000, 0.5),
        curve=curve,
        method="lagrangian",
        energy_exponent=_ENERGY_EXPONENT,
        reduction=reduction,
    )


def test_format_report_has_all_sections(storage: Storage, reduction: ReductionSummary) -> None:
    point = _point()
    plan = _ungrouped_plan(
        pitches=(
            PitchPlan(60, 5.0, 5.0, SampleKey(60, 100), point, (point,)),
            PitchPlan(67, 4.0, 4.0, SampleKey(67, 100), point, (point,)),
        ),
        # one curve point that overflows the sample budget and one that fits, to exercise both the
        # "<= budget" marker and its absence.
        curve=(
            RDCurvePoint(lam=2.0, total_bytes=70_000, objective=9.0, indices=(0,)),
            RDCurvePoint(lam=1.0, total_bytes=60_000, objective=8.0, indices=(0,)),
        ),
        storage=storage,
        reduction=reduction,
    )
    report = format_report(plan, _SIZE, _COVERAGE, _layout(plan))
    assert "Instrument 'piano'" in report
    assert "Budget:" in report and "Objective:" in report
    assert f"{bytes_to_kib(_SIZE.total):7.1f} KiB module" in report  # the written size sits beside the budget
    assert "Per-pitch allocation" in report
    assert "C4" in report and "G4" in report  # MIDI 60 / 67 note names
    assert "Velocity->volume map" in report
    assert "[reference]" in report  # the loudest anchor is flagged, the quieter one is not
    assert "Budget->quality curve" in report
    assert report.endswith("\n")


def test_a_report_states_the_weighting_its_objective_was_measured_under(
    storage: Storage, reduction: ReductionSummary
) -> None:
    """Two objectives only compare under one weighting, so each report says which one produced it."""
    point = _point()
    ungrouped = _ungrouped_plan(
        pitches=(PitchPlan(60, 5.0, 5.0, SampleKey(60, 100), point, (point,)),),
        curve=(RDCurvePoint(lam=1.0, total_bytes=6000, objective=8.0, indices=(0,)),),
        storage=storage,
        reduction=reduction,
    )
    assert f"energy^{_ENERGY_EXPONENT:g}-weighted" in format_report(ungrouped, _SIZE, _COVERAGE, _layout(ungrouped))
    layered = _layered_plan(storage, reduction)
    assert f"energy^{_ENERGY_EXPONENT:g}-weighted" in format_grouping_report(
        layered, _SIZE, _COVERAGE, _layout(layered)
    )


def test_report_curve_always_includes_the_final_point(storage: Storage, reduction: ReductionSummary) -> None:
    # A long curve whose display stride (every other vertex) would otherwise skip the last one.
    point = _point()
    curve = tuple(
        RDCurvePoint(lam=float(12 - i), total_bytes=1000 + 400 * i, objective=20.0 - i, indices=(0,)) for i in range(12)
    )
    plan = _ungrouped_plan(
        pitches=(PitchPlan(60, 5.0, 5.0, SampleKey(60, 100), point, (point,)),),
        curve=curve,
        storage=storage,
        reduction=reduction,
    )
    report = format_report(plan, _SIZE, _COVERAGE, _layout(plan))
    assert f"{bytes_to_kib(curve[-1].total_bytes):7.1f} KiB" in report  # final vertex shown despite the stride


def _grouped_plan(
    zones: tuple[Zone, ...],
    layers: VelocityLayers,
    storage: Storage,
    reduction: ReductionSummary,
    reserve: SampleReserve = _UNCHARGED,
) -> GroupedInstrumentPlan:
    return GroupedInstrumentPlan(
        instrument_id="piano",
        budget=split_budget(_MODULE_BYTES / 1024.0, storage, layers.count),
        velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
        layers=layers,
        zones=zones,
        total_bytes=sum(zone.chosen.stored_bytes for zone in zones),
        objective=sum(zone.chosen.distortion for zone in zones),
        reserve=reserve,
        energy_exponent=_ENERGY_EXPONENT,
        reduction=reduction,
    )


def test_grouping_report_has_the_expected_sections(storage: Storage, reduction: ReductionSummary) -> None:
    multi = ZoneOption(61, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    single = ZoneOption(67, EncodingParams(target_rate=11_025, depth_bits=8), 3000, 0.3, 1200)
    plan = _grouped_plan(
        zones=(
            Zone((60, 61, 62), FIRST_LAYER, SampleKey(61, 100), 3.0, multi, (multi,)),  # a merged, multi-key zone
            Zone((67,), FIRST_LAYER, SampleKey(67, 90), 1.0, single, (single,)),  # a lone single-key zone
        ),
        layers=_WHOLE_AXIS,
        storage=storage,
        reduction=reduction,
    )
    report = format_grouping_report(plan, _SIZE, _COVERAGE, _layout(plan))
    assert "pitch-zone grouping" in report
    assert "Budget:" in report and "Zones" in report
    assert "60-62" in report and " 67 " in report  # multi-key span and single-key span
    assert "2 zones over 4 keys and 1 velocity layer" in report
    assert report.endswith("\n")


def _layered_plan(storage: Storage, reduction: ReductionSummary) -> GroupedInstrumentPlan:
    """Two keys stored twice over: once for the dynamics under v50 and once for those above it."""
    quiet = ZoneOption(60, EncodingParams(target_rate=11_025, depth_bits=8), 3000, 0.4, 1200)
    loud = ZoneOption(61, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    return _grouped_plan(
        zones=(
            Zone((60, 61), 0, SampleKey(60, 50), 2.0, quiet, (quiet,)),
            Zone((60, 61), 1, SampleKey(61, 100), 4.0, loud, (loud,)),
        ),
        layers=VelocityLayers((VelocityBand(0, 50), VelocityBand(51, MIDI_MAX_VELOCITY))),
        storage=storage,
        reduction=reduction,
    )


def test_a_layered_report_prices_the_split_band_by_band(storage: Storage, reduction: ReductionSummary) -> None:
    plan = _layered_plan(storage, reduction)
    report = format_grouping_report(plan, _SIZE, _COVERAGE, _layout(plan))
    assert "Instruments (" in report
    assert "v000-v050" in report and "v051-v127" in report
    assert f"{bytes_to_kib(3000):9.1f}" in report and f"{bytes_to_kib(6000):9.1f}" in report


def test_a_band_written_as_several_instruments_prices_each_one(storage: Storage, reduction: ReductionSummary) -> None:
    """A format numbering few samples per instrument cuts a band, and each row states the keys it owns."""
    multi = ZoneOption(61, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    single = ZoneOption(67, EncodingParams(target_rate=11_025, depth_bits=8), 3000, 0.3, 1200)
    plan = _grouped_plan(
        zones=(
            Zone((60, 61, 62), FIRST_LAYER, SampleKey(61, 100), 3.0, multi, (multi,)),
            Zone((67,), FIRST_LAYER, SampleKey(67, 90), 1.0, single, (single,)),
        ),
        layers=_WHOLE_AXIS,
        storage=storage,
        reduction=reduction,
    )
    report = format_grouping_report(plan, _SIZE, _COVERAGE, _layout(plan, _ONE_SAMPLE_EACH))
    assert "C4-D4" in report and "G4-G4" in report
    assert f"{bytes_to_kib(6000):9.1f}" in report and f"{bytes_to_kib(3000):9.1f}" in report


def test_the_header_states_the_instruments_reserved_against_those_written(
    storage: Storage, reduction: ReductionSummary
) -> None:
    """The reserve is taken at the worst case, so a plan coming out smaller says what stayed free."""
    plan = replace(
        _layered_plan(storage, reduction),
        budget=split_budget(_MODULE_BYTES / 1024.0, storage, _RESERVED),
    )
    report = format_grouping_report(plan, _SIZE, _COVERAGE, _layout(plan))
    assert f"Instruments: {_RESERVED:>5} reserved  ->  2 written" in report
    assert f"{bytes_to_kib(2 * populated_instrument_bytes(storage)):.1f} KiB of the reserve left free" in report


def test_a_layered_report_counts_each_key_once_however_many_bands_store_it(
    storage: Storage, reduction: ReductionSummary
) -> None:
    """Two layers over the same two keys is a two-key instrument, so the summary line says two."""
    plan = _layered_plan(storage, reduction)
    report = format_grouping_report(plan, _SIZE, _COVERAGE, _layout(plan))
    assert "2 zones over 2 keys and 2 velocity layers" in report


def test_every_zone_states_the_layer_it_answers_for(storage: Storage, reduction: ReductionSummary) -> None:
    """A key served twice appears once per band, so the zone table names which one each row belongs to."""
    plan = _layered_plan(storage, reduction)
    rows = format_grouping_report(plan, _SIZE, _COVERAGE, _layout(plan)).splitlines()
    zone_rows = [line for line in rows if "60-61 (2)" in line]
    assert [line.split()[0] for line in zone_rows] == ["0", "1"]


# --- the reduction block --------------------------------------------------------------------------


_COVERED: Final = (KeptRecording(SampleKey(60, 100), duration_s=1.0, required_duration_s=0.8),)
_STORED: Final = StoredFormat(target_rate=11_025, depths=(16,), carriers=(False,))
_ONE_GRID: Final = (NarrowedGrid(60, 10_500.0, _STORED, (EncodingParams(target_rate=11_025, depth_bits=16),)),)


def _summary(
    *,
    recordings: tuple[KeptRecording, ...] = _COVERED,
    grids: tuple[NarrowedGrid, ...] = _ONE_GRID,
) -> ReductionSummary:
    """A summary over fixed counts, varying only the two collections a block renders differently."""
    return ReductionSummary(
        listed_recordings=9,
        played_notes=8,
        scored_classes=3,
        recordings=recordings,
        grids=grids,
    )


def test_the_reduction_block_states_both_sides_of_every_axis() -> None:
    block = format_reduction_block(_summary())
    assert "9 listed" in block and "1 kept" in block
    assert "8 notes" in block and "3 scored classes" in block
    assert "10.5 kHz useful" in block and "11.0 kHz stored" in block
    assert "1.0 swept per key" in block


def test_a_recording_short_of_its_material_is_called_out() -> None:
    short = KeptRecording(SampleKey(67, 90), duration_s=0.40, required_duration_s=1.25)
    block = format_reduction_block(_summary(recordings=(short,)))
    assert "1 of 1 kept recordings hold less than asked" in block
    assert short.key.label in block and " 0.40s of  1.25s" in block


def test_a_run_whose_recordings_all_cover_their_material_says_nothing_of_shortfalls() -> None:
    assert "hold less than asked" not in format_reduction_block(_summary())


def test_only_the_first_few_shortfalls_are_named_and_the_rest_counted() -> None:
    shortfalls = tuple(
        KeptRecording(SampleKey(60 + index, 100), duration_s=0.1, required_duration_s=1.0) for index in range(5)
    )
    block = format_reduction_block(_summary(recordings=shortfalls))
    assert "5 of 5 kept recordings hold less than asked" in block
    assert shortfalls[2].key.label in block and shortfalls[3].key.label not in block
    assert "(and 2 more)" in block


def test_an_instrument_playing_nothing_says_its_format_is_yet_to_be_settled() -> None:
    block = format_reduction_block(_summary(grids=()))
    assert "settled once the material plays a pitch" in block and "kHz" not in block


def test_a_run_storing_every_key_at_one_rate_states_that_rate_once() -> None:
    """Two keys settling on the same rung read as one figure, which is what the run actually stored."""
    second = NarrowedGrid(67, 10_500.0, _STORED, (EncodingParams(target_rate=11_025, depth_bits=16),))
    block = format_reduction_block(_summary(grids=(*_ONE_GRID, second)))
    assert "11.0 kHz stored" in block and "11.0-11.0" not in block


def test_both_strategies_report_the_reduction(storage: Storage, reduction: ReductionSummary) -> None:
    """The stage runs once per instrument, so either plan's report states the same search space."""
    point = _point()
    plan = _ungrouped_plan(
        pitches=(PitchPlan(60, 5.0, 5.0, SampleKey(60, 100), point, (point,)),),
        curve=(RDCurvePoint(lam=1.0, total_bytes=6000, objective=8.0, indices=(0,)),),
        storage=storage,
        reduction=reduction,
    )
    ungrouped = format_report(plan, _SIZE, _COVERAGE, _layout(plan))
    option = ZoneOption(60, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    zoned = _grouped_plan(
        zones=(Zone((60,), FIRST_LAYER, SampleKey(60, 100), 1.0, option, (option,)),),
        layers=_WHOLE_AXIS,
        storage=storage,
        reduction=reduction,
    )
    grouped = format_grouping_report(zoned, _SIZE, _COVERAGE, _layout(zoned))
    block = format_reduction_block(reduction)
    assert block in ungrouped and block in grouped


def test_a_grouped_report_states_the_sample_cap_it_was_held_to(storage: Storage, reduction: ReductionSummary) -> None:
    """A cap the plan already meets is stated as met, so a reader sees the room the format still has."""
    option = ZoneOption(60, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    zones = (Zone((60,), FIRST_LAYER, SampleKey(60, 100), 1.0, option, (option,)),)
    free = _grouped_plan(zones=zones, layers=_WHOLE_AXIS, storage=storage, reduction=reduction)
    report = format_grouping_report(free, _SIZE, _COVERAGE, _layout(free))
    assert f"1 stored of {_WHOLE_TABLE} allowed" in report
    assert "priced at the bytes it stores" in report


def test_a_grouped_report_prices_the_cap_that_decided_the_plan(storage: Storage, reduction: ReductionSummary) -> None:
    """A cap met by charging states the charge and the objective the same budget reached without it."""
    option = ZoneOption(60, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    zones = (Zone((60, 61), FIRST_LAYER, SampleKey(60, 100), 1.0, option, (option,)),)
    capped = _grouped_plan(zones=zones, layers=_WHOLE_AXIS, storage=storage, reduction=reduction, reserve=_CHARGED)
    report = format_grouping_report(capped, _SIZE, _COVERAGE, _layout(capped))
    assert f"1 stored of {_CHARGED.cap} allowed" in report
    assert f"{_CHARGED.bytes_per_sample} B charged per sample" in report
    assert f"{_CHARGED.objective_uncapped:.4f} uncapped" in report
