from __future__ import annotations

from optisample.dsp.surrogate import EncodingParams
from optisample.metrics.size import bytes_to_kib
from optisample.optimize.knapsack import Allocation, RDCurvePoint, Selection
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans import (
    BudgetBreakdown,
    GroupedInstrumentPlan,
    InstrumentPlan,
    PitchPlan,
    Zone,
    ZoneOption,
)
from optisample.optimize.report import format_grouping_report, format_report
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap

_MODULE_BYTES = 64 * 1024
_SAMPLE_BYTES = 64 * 1024 - 746  # module minus the file + instrument header overhead


def _point(rate: int = 22_050, depth: int = 16, size: int = 5000, distortion: float = 0.1) -> OperatingPoint:
    return OperatingPoint(EncodingParams(target_rate=rate, depth_bits=depth), size, distortion, 2400)


def _ungrouped_plan(pitches: tuple[PitchPlan, ...], curve: tuple[RDCurvePoint, ...]) -> InstrumentPlan:
    return InstrumentPlan(
        instrument_id="piano",
        budget=BudgetBreakdown(module_bytes=_MODULE_BYTES, sample_bytes=_SAMPLE_BYTES),
        velocity_map=VelocityVolumeMap(
            tuple(64 for _ in range(128)),
            (VelocityAnchor(50, -20.0, 40), VelocityAnchor(100, -10.0, 64)),  # only the loudest is the reference
        ),
        pitches=pitches,
        allocation=Allocation(tuple(Selection(str(p.pitch), p.weight, p.chosen) for p in pitches), 5000, 0.5),
        curve=curve,
        method="lagrangian",
    )


def test_format_report_has_all_sections() -> None:
    point = _point()
    plan = _ungrouped_plan(
        pitches=(PitchPlan(60, 5.0, 100, point, (point,)), PitchPlan(67, 4.0, 100, point, (point,))),
        # one curve point that overflows the sample budget and one that fits, to exercise both the
        # "<= budget" marker and its absence.
        curve=(
            RDCurvePoint(lam=2.0, total_bytes=70_000, objective=9.0, indices=(0,)),
            RDCurvePoint(lam=1.0, total_bytes=60_000, objective=8.0, indices=(0,)),
        ),
    )
    report = format_report(plan)
    assert "Instrument 'piano'" in report
    assert "Budget:" in report and "Objective:" in report
    assert "Per-pitch allocation" in report
    assert "C4" in report and "G4" in report  # MIDI 60 / 67 note names
    assert "Velocity->volume map" in report
    assert "[reference]" in report  # the loudest anchor is flagged, the quieter one is not
    assert "Budget->quality curve" in report
    assert report.endswith("\n")


def test_report_curve_always_includes_the_final_point() -> None:
    # A long curve whose display stride (every other vertex) would otherwise skip the last one.
    point = _point()
    curve = tuple(
        RDCurvePoint(lam=float(12 - i), total_bytes=1000 + 400 * i, objective=20.0 - i, indices=(0,)) for i in range(12)
    )
    plan = _ungrouped_plan(pitches=(PitchPlan(60, 5.0, 100, point, (point,)),), curve=curve)
    report = format_report(plan)
    assert f"{bytes_to_kib(curve[-1].total_bytes):7.1f} KiB" in report  # final vertex shown despite the stride


def test_grouping_report_has_the_expected_sections() -> None:
    multi = ZoneOption(61, EncodingParams(target_rate=22_050, depth_bits=16), 6000, 0.2, 2400)
    single = ZoneOption(67, EncodingParams(target_rate=11_025, depth_bits=8), 3000, 0.3, 1200)
    plan = GroupedInstrumentPlan(
        instrument_id="piano",
        budget=BudgetBreakdown(module_bytes=_MODULE_BYTES, sample_bytes=_SAMPLE_BYTES),
        velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
        zones=(
            Zone((60, 61, 62), 61, 100, 3.0, multi, (multi,)),  # a merged, multi-key zone
            Zone((67,), 67, 90, 1.0, single, (single,)),  # a lone single-key zone
        ),
        total_bytes=9000,
        objective=0.5,
    )
    report = format_grouping_report(plan)
    assert "pitch-zone grouping" in report
    assert "Budget:" in report and "Zones" in report
    assert "60-62" in report and " 67 " in report  # multi-key span and single-key span
    assert report.endswith("\n")
