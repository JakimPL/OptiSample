from __future__ import annotations

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
from optisample.optimize.report import format_grouping_report, format_report
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from trackmod.module.size import SizeReport
from trackmod.module.storage import Storage

_MODULE_BYTES = 64 * 1024
_SIZE = SizeReport(patterns=120, pcm=9000, headers=800, largest_pattern=90)


def _point(rate: int = 22_050, depth: int = 16, size: int = 5000, distortion: float = 0.1) -> OperatingPoint:
    return OperatingPoint(EncodingParams(target_rate=rate, depth_bits=depth), size, distortion, 2400)


def _ungrouped_plan(
    pitches: tuple[PitchPlan, ...], curve: tuple[RDCurvePoint, ...], storage: Storage
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
    )


def test_format_report_has_all_sections(storage: Storage) -> None:
    point = _point()
    plan = _ungrouped_plan(
        pitches=(PitchPlan(60, 5.0, 100, point, (point,)), PitchPlan(67, 4.0, 100, point, (point,))),
        # one curve point that overflows the sample budget and one that fits, to exercise both the
        # "<= budget" marker and its absence.
        curve=(
            RDCurvePoint(lam=2.0, total_bytes=70_000, objective=9.0, indices=(0,)),
            RDCurvePoint(lam=1.0, total_bytes=60_000, objective=8.0, indices=(0,)),
        ),
        storage=storage,
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


def test_report_curve_always_includes_the_final_point(storage: Storage) -> None:
    # A long curve whose display stride (every other vertex) would otherwise skip the last one.
    point = _point()
    curve = tuple(
        RDCurvePoint(lam=float(12 - i), total_bytes=1000 + 400 * i, objective=20.0 - i, indices=(0,)) for i in range(12)
    )
    plan = _ungrouped_plan(pitches=(PitchPlan(60, 5.0, 100, point, (point,)),), curve=curve, storage=storage)
    report = format_report(plan, _SIZE)
    assert f"{bytes_to_kib(curve[-1].total_bytes):7.1f} KiB" in report  # final vertex shown despite the stride


def test_grouping_report_has_the_expected_sections(storage: Storage) -> None:
    multi = ZoneOption(61, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    single = ZoneOption(67, EncodingParams(target_rate=11_025, depth_bits=8), 3000, 0.3, 1200)
    plan = GroupedInstrumentPlan(
        instrument_id="piano",
        budget=split_budget(_MODULE_BYTES / 1024.0, storage),
        velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
        zones=(
            Zone((60, 61, 62), 61, 100, 3.0, multi, (multi,)),  # a merged, multi-key zone
            Zone((67,), 67, 90, 1.0, single, (single,)),  # a lone single-key zone
        ),
        total_bytes=9000,
        objective=0.5,
    )
    report = format_grouping_report(plan, _SIZE)
    assert "pitch-zone grouping" in report
    assert "Budget:" in report and "Zones" in report
    assert "60-62" in report and " 67 " in report  # multi-key span and single-key span
    assert report.endswith("\n")
