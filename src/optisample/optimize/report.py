"""Human-readable reports for both allocation strategies.

The optimizers in :mod:`optisample.optimize.orchestrate` (one sample per key) and
:mod:`optisample.optimize.grouping` (pitch zones) return structured plans; this module turns either
plan into the fixed-width text summary the CLI and the inspection dump print. Reporting lives apart
from the solvers so the algorithm modules carry no formatting, and so the two reports share one
header, one budget block and one set of rules instead of drifting copies.
"""

from __future__ import annotations

from typing import Protocol

from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, bytes_to_kib
from optisample.music import note_name
from optisample.optimize.grouping import GroupedInstrumentPlan
from optisample.optimize.orchestrate import InstrumentPlan

RULE_WIDTH = 70
SECTION_RULE = "=" * RULE_WIDTH
SUBSECTION_RULE = "-" * RULE_WIDTH

_CURVE_ROWS = 6


class _Budgeted(Protocol):
    """The budget-facing surface every plan exposes -- what :func:`format_budget_block` needs."""

    @property
    def module_budget_bytes(self) -> int: ...

    @property
    def sample_budget_bytes(self) -> int: ...

    @property
    def used_bytes(self) -> int: ...

    @property
    def module_bytes(self) -> int: ...


def format_budget_block(plan: _Budgeted) -> list[str]:
    """The two-line ``Budget:``/``Used:`` summary shared by the ungrouped and grouped reports."""
    overhead = FILE_HEADER_BYTES + INSTRUMENT_HEADER_BYTES
    used = plan.used_bytes
    fraction = used / plan.sample_budget_bytes if plan.sample_budget_bytes > 0 else float("nan")
    headroom = plan.sample_budget_bytes - used
    return [
        f"Budget:    {bytes_to_kib(plan.module_budget_bytes):7.1f} KiB module  ->  "
        f"{bytes_to_kib(plan.sample_budget_bytes):7.1f} KiB samples  "
        f"(overhead {overhead} B: file {FILE_HEADER_BYTES} + instrument {INSTRUMENT_HEADER_BYTES})",
        f"Used:      {bytes_to_kib(used):7.1f} KiB samples  ({fraction:6.1%} of budget, "
        f"{bytes_to_kib(headroom):.1f} KiB free)  ->  {bytes_to_kib(plan.module_bytes):.1f} KiB module",
    ]


def _format_header(plan: _Budgeted, title: str, summary: str) -> str:
    """Title, section rule, the shared budget block, then a strategy-specific summary line."""
    return "\n".join((title, SECTION_RULE, *format_budget_block(plan), summary))


# --- ungrouped report ----------------------------------------------------------------------------


def _ungrouped_header(plan: InstrumentPlan) -> str:
    return _format_header(
        plan,
        f"Instrument {plan.instrument_id!r} - budget solver (method: {plan.method})",
        f"Objective: {plan.objective:8.4f}  (sum of weight x distortion over "
        f"{len(plan.pitches)} pitches, {plan.total_weight:.1f} s of material)",
    )


def _format_pitches(plan: InstrumentPlan) -> str:
    lines = [
        "Per-pitch allocation",
        SUBSECTION_RULE,
        f"{'pitch':>5}  {'note':>4}  {'weight(s)':>9}  {'rep.vel':>7}  "
        f"{'rate(Hz)':>8}  {'depth':>5}  {'size(KiB)':>9}  {'distortion':>10}  {'hull':>4}",
    ]
    for pitch in plan.pitches:
        point = pitch.chosen
        lines.append(
            f"{pitch.pitch:>5}  {note_name(pitch.pitch):>4}  {pitch.weight:>9.1f}  "
            f"{pitch.representative_velocity:>7}  {point.params.target_rate:>8}  {point.params.depth_bits:>5}  "
            f"{bytes_to_kib(point.stored_bytes):>9.1f}  {point.distortion:>10.4f}  {len(pitch.hull):>4}"
        )
    return "\n".join(lines)


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


def format_report(plan: InstrumentPlan) -> str:
    """Render a human-readable summary of an instrument optimization."""
    sections = (_ungrouped_header(plan), _format_pitches(plan), _format_velocity_map(plan), _format_curve(plan))
    return "\n\n".join(sections) + "\n"


# --- grouped report ------------------------------------------------------------------------------


def _grouped_header(plan: GroupedInstrumentPlan) -> str:
    return _format_header(
        plan,
        f"Instrument {plan.instrument_id!r} - pitch-zone grouping (exact partition + allocation DP)",
        f"Grouping:  {len(plan.zones)} zones cover {len(plan.pitches)} keys  "
        f"(objective {plan.objective:.4f} over {plan.total_weight:.1f} s of material)",
    )


def _format_zones(plan: GroupedInstrumentPlan) -> str:
    lines = [
        "Zones (one stored sample each, repitched across the zone's keys)",
        SUBSECTION_RULE,
        f"{'keys':>11}  {'rep':>4}  {'rep.vel':>7}  {'rate(Hz)':>8}  "
        f"{'depth':>5}  {'size(KiB)':>9}  {'distortion':>10}  {'options':>7}",
    ]
    for zone in plan.zones:
        option = zone.chosen
        keys = f"{zone.pitches[0]}-{zone.pitches[-1]}" if len(zone.pitches) > 1 else str(zone.pitches[0])
        span = f"{keys} ({len(zone.pitches)})"
        lines.append(
            f"{span:>11}  {zone.representative:>4}  {zone.representative_velocity:>7}  "
            f"{option.params.target_rate:>8}  {option.params.depth_bits:>5}  "
            f"{bytes_to_kib(option.stored_bytes):>9.1f}  {option.distortion:>10.4f}  {len(zone.hull):>7}"
        )
    return "\n".join(lines)


def format_grouping_report(plan: GroupedInstrumentPlan) -> str:
    """Render a human-readable summary of a grouped optimization."""
    return "\n\n".join((_grouped_header(plan), _format_zones(plan))) + "\n"
