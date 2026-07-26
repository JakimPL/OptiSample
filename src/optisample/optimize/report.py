from collections.abc import Iterable
from typing import Final

from optisample.metrics.size import bytes_to_kib
from optisample.music import note_name
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.plans.budget import (
    BudgetedPlanMixin,
    instrument_overhead,
    populated_instrument_bytes,
)
from trackmod.module.size import SizeReport

RULE_WIDTH: Final = 70
SECTION_RULE: Final = "=" * RULE_WIDTH
SUBSECTION_RULE: Final = "-" * RULE_WIDTH

_CURVE_ROWS: Final = 6


def _format_allocation_table(title: str, header: str, rows: Iterable[str]) -> str:
    """A titled, ruled allocation table: heading, subsection rule, column header, then the data rows."""
    return "\n".join((title, SUBSECTION_RULE, header, *rows))


def format_budget_block(plan: BudgetedPlanMixin, size: SizeReport) -> list[str]:
    """The ``Budget:``/``Used:``/``Written:`` summary shared by the ungrouped and grouped reports.

    The first two lines account for the instrument's own footprint, which is what the solver allocated
    against; the third states what the module file occupies once the audition material is in it.
    """
    storage = plan.budget.storage
    used = plan.used_bytes
    fraction = used / plan.sample_budget_bytes if plan.sample_budget_bytes > 0 else float("nan")
    headroom = plan.sample_budget_bytes - used
    return [
        f"Budget:    {bytes_to_kib(plan.module_budget_bytes):7.1f} KiB module  ->  "
        f"{bytes_to_kib(plan.sample_budget_bytes):7.1f} KiB samples  "
        f"(overhead {instrument_overhead(storage)} B: file {storage.file} + "
        f"instrument {populated_instrument_bytes(storage)})",
        f"Used:      {bytes_to_kib(used):7.1f} KiB samples  ({fraction:6.1%} of budget, "
        f"{bytes_to_kib(headroom):.1f} KiB free)  ->  {bytes_to_kib(plan.module_bytes):.1f} KiB module",
        f"Written:   {bytes_to_kib(size.total):7.1f} KiB module  "
        f"(records {size.headers} B + samples {size.pcm} B + patterns {size.patterns} B)",
    ]


def _format_header(plan: BudgetedPlanMixin, size: SizeReport, title: str, summary: str) -> str:
    """Title, section rule, the shared budget block, then a strategy-specific summary line."""
    return "\n".join((title, SECTION_RULE, *format_budget_block(plan, size), summary))


def _ungrouped_header(plan: InstrumentPlan, size: SizeReport) -> str:
    return _format_header(
        plan,
        size,
        f"Instrument {plan.instrument_id!r} - budget solver (method: {plan.method})",
        f"Objective: {plan.objective:8.4f}  (sum of weight x distortion over "
        f"{len(plan.pitches)} pitches, {plan.total_weight:.1f} s of material)",
    )


def _format_pitches(plan: InstrumentPlan) -> str:
    header = (
        f"{'pitch':>5}  {'note':>4}  {'weight(s)':>9}  {'rep.vel':>7}  "
        f"{'rate(Hz)':>8}  {'depth':>5}  {'size(KiB)':>9}  {'distortion':>10}  {'hull':>4}"
    )
    rows = []
    for pitch in plan.pitches:
        point = pitch.chosen
        rows.append(
            f"{pitch.pitch:>5}  {note_name(pitch.pitch):>4}  {pitch.weight:>9.1f}  "
            f"{pitch.representative_velocity:>7}  {point.params.target_rate:>8}  {point.params.depth_bits:>5}  "
            f"{bytes_to_kib(point.stored_bytes):>9.1f}  {point.distortion:>10.4f}  {len(pitch.hull):>4}"
        )

    return _format_allocation_table("Per-pitch allocation", header, rows)


def _format_velocity_map(plan: InstrumentPlan) -> str:
    anchors = plan.velocity_map.anchors
    reference_volume = max((anchor.volume for anchor in anchors), default=0)
    lines = ["Velocity->volume map (loudness-matched, 0-64; loudest velocity -> 64)", SUBSECTION_RULE]
    for anchor in anchors:
        marker = "  [reference]" if anchor.volume == reference_volume else ""
        lines.append(
            f"  vel {anchor.velocity:>3}  ->  vol {anchor.volume:>2}   ({anchor.loudness_lufs:6.1f} LUFS){marker}"
        )

    lines.append("  (full 0-127 map interpolated between these anchors)")
    return "\n".join(lines)


def _format_curve(plan: InstrumentPlan) -> str:
    curve = plan.curve
    lines = ["Budget->quality curve (Lagrangian sweep over the RD hulls)", SUBSECTION_RULE]
    step = max(1, (len(curve) - 1) // (_CURVE_ROWS - 1)) if len(curve) > 1 else 1
    shown = list(range(0, len(curve), step))
    if shown and shown[-1] != len(curve) - 1:
        shown.append(len(curve) - 1)

    for position in shown:
        point = curve[position]
        fits = "" if point.total_bytes > plan.sample_budget_bytes else "  <= budget"
        lines.append(f"  {bytes_to_kib(point.total_bytes):7.1f} KiB  ->  objective {point.objective:8.4f}{fits}")

    return "\n".join(lines)


def format_report(plan: InstrumentPlan, size: SizeReport) -> str:
    """Render a human-readable summary of an instrument optimization."""
    sections = (_ungrouped_header(plan, size), _format_pitches(plan), _format_velocity_map(plan), _format_curve(plan))
    return "\n\n".join(sections) + "\n"


def _grouped_header(plan: GroupedInstrumentPlan, size: SizeReport) -> str:
    return _format_header(
        plan,
        size,
        f"Instrument {plan.instrument_id!r} - pitch-zone grouping (exact partition + allocation DP)",
        f"Grouping:  {len(plan.zones)} zones cover {len(plan.pitches)} keys  "
        f"(objective {plan.objective:.4f} over {plan.total_weight:.1f} s of material)",
    )


def _format_zones(plan: GroupedInstrumentPlan) -> str:
    header = (
        f"{'keys':>11}  {'rep':>4}  {'rep.vel':>7}  {'rate(Hz)':>8}  "
        f"{'depth':>5}  {'size(KiB)':>9}  {'distortion':>10}  {'options':>7}"
    )
    rows = []
    for zone in plan.zones:
        option = zone.chosen
        keys = f"{zone.pitches[0]}-{zone.pitches[-1]}" if len(zone.pitches) > 1 else str(zone.pitches[0])
        span = f"{keys} ({len(zone.pitches)})"
        rows.append(
            f"{span:>11}  {zone.representative:>4}  {zone.representative_velocity:>7}  "
            f"{option.params.target_rate:>8}  {option.params.depth_bits:>5}  "
            f"{bytes_to_kib(option.stored_bytes):>9.1f}  {option.distortion:>10.4f}  {len(zone.hull):>7}"
        )
    return _format_allocation_table("Zones (one stored sample each, repitched across the zone's keys)", header, rows)


def format_grouping_report(plan: GroupedInstrumentPlan, size: SizeReport) -> str:
    """Render a human-readable summary of a grouped optimization."""
    return "\n\n".join((_grouped_header(plan, size), _format_zones(plan))) + "\n"
