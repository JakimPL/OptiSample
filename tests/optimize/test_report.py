from __future__ import annotations

from typing import Final

from optisample.dsp.surrogate import EncodingParams
from optisample.metrics.size import bytes_to_kib
from optisample.optimize.knapsack import Allocation, RDCurvePoint, Selection
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans import (
    GroupedInstrumentPlan,
    InstrumentPlan,
    PitchPlan,
    Zone,
    ZoneOption,
    split_budget,
)
from optisample.optimize.reduce.keys import SampleKey
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
        budget=split_budget(_MODULE_BYTES / 1024.0, storage),
        velocity_map=VelocityVolumeMap(
            tuple(64 for _ in range(128)),
            (VelocityAnchor(50, -20.0, 40), VelocityAnchor(100, -10.0, 64)),  # only the loudest is the reference
        ),
        pitches=pitches,
        allocation=Allocation(tuple(Selection(str(p.pitch), p.weight, p.chosen) for p in pitches), 5000, 0.5),
        curve=curve,
        method="lagrangian",
        reduction=reduction,
    )


def test_format_report_has_all_sections(storage: Storage, reduction: ReductionSummary) -> None:
    point = _point()
    plan = _ungrouped_plan(
        pitches=(
            PitchPlan(60, 5.0, SampleKey(60, 100), point, (point,)),
            PitchPlan(67, 4.0, SampleKey(67, 100), point, (point,)),
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
    report = format_report(plan, _SIZE)
    assert "Instrument 'piano'" in report
    assert "Budget:" in report and "Objective:" in report
    assert f"{bytes_to_kib(_SIZE.total):7.1f} KiB module" in report  # the written size sits beside the budget
    assert "Per-pitch allocation" in report
    assert "C4" in report and "G4" in report  # MIDI 60 / 67 note names
    assert "Velocity->volume map" in report
    assert "[reference]" in report  # the loudest anchor is flagged, the quieter one is not
    assert "Budget->quality curve" in report
    assert report.endswith("\n")


def test_report_curve_always_includes_the_final_point(storage: Storage, reduction: ReductionSummary) -> None:
    # A long curve whose display stride (every other vertex) would otherwise skip the last one.
    point = _point()
    curve = tuple(
        RDCurvePoint(lam=float(12 - i), total_bytes=1000 + 400 * i, objective=20.0 - i, indices=(0,)) for i in range(12)
    )
    plan = _ungrouped_plan(
        pitches=(PitchPlan(60, 5.0, SampleKey(60, 100), point, (point,)),),
        curve=curve,
        storage=storage,
        reduction=reduction,
    )
    report = format_report(plan, _SIZE)
    assert f"{bytes_to_kib(curve[-1].total_bytes):7.1f} KiB" in report  # final vertex shown despite the stride


def test_grouping_report_has_the_expected_sections(storage: Storage, reduction: ReductionSummary) -> None:
    multi = ZoneOption(61, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    single = ZoneOption(67, EncodingParams(target_rate=11_025, depth_bits=8), 3000, 0.3, 1200)
    plan = GroupedInstrumentPlan(
        instrument_id="piano",
        budget=split_budget(_MODULE_BYTES / 1024.0, storage),
        velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
        zones=(
            Zone((60, 61, 62), SampleKey(61, 100), 3.0, multi, (multi,)),  # a merged, multi-key zone
            Zone((67,), SampleKey(67, 90), 1.0, single, (single,)),  # a lone single-key zone
        ),
        total_bytes=9000,
        objective=0.5,
        reduction=reduction,
    )
    report = format_grouping_report(plan, _SIZE)
    assert "pitch-zone grouping" in report
    assert "Budget:" in report and "Zones" in report
    assert "60-62" in report and " 67 " in report  # multi-key span and single-key span
    assert report.endswith("\n")


# --- the reduction block --------------------------------------------------------------------------


_COVERED: Final = (KeptRecording(SampleKey(60, 100), duration_s=1.0, required_duration_s=0.8),)
_ONE_GRID: Final = (NarrowedGrid(60, 11_025.0, (EncodingParams(target_rate=11_025, depth_bits=16),)),)


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
        grid_size=12,
        recordings=recordings,
        grids=grids,
    )


def test_the_reduction_block_states_both_sides_of_every_axis() -> None:
    block = format_reduction_block(_summary())
    assert "9 listed" in block and "1 kept" in block
    assert "8 notes" in block and "3 scored classes" in block
    assert "12 encodings" in block and "1.0 shortlisted per key" in block
    assert "11.0-11.0 kHz useful" in block


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


def test_an_instrument_playing_nothing_still_reports_its_grid() -> None:
    block = format_reduction_block(_summary(grids=()))
    assert "12 encodings" in block and "kHz" not in block


def test_both_strategies_report_the_reduction(storage: Storage, reduction: ReductionSummary) -> None:
    """The stage runs once per instrument, so either plan's report states the same search space."""
    point = _point()
    ungrouped = format_report(
        _ungrouped_plan(
            pitches=(PitchPlan(60, 5.0, SampleKey(60, 100), point, (point,)),),
            curve=(RDCurvePoint(lam=1.0, total_bytes=6000, objective=8.0, indices=(0,)),),
            storage=storage,
            reduction=reduction,
        ),
        _SIZE,
    )
    option = ZoneOption(60, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    grouped = format_grouping_report(
        GroupedInstrumentPlan(
            instrument_id="piano",
            budget=split_budget(_MODULE_BYTES / 1024.0, storage),
            velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
            zones=(Zone((60,), SampleKey(60, 100), 1.0, option, (option,)),),
            total_bytes=6000,
            objective=0.2,
            reduction=reduction,
        ),
        _SIZE,
    )
    block = format_reduction_block(reduction)
    assert block in ungrouped and block in grouped
