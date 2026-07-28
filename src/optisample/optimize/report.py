from collections.abc import Iterable, Sequence
from typing import Final

from optisample.dsp.surrogate import EncodingParams
from optisample.metrics.size import bytes_to_kib
from optisample.music import note_name
from optisample.optimize.export.coverage import KeyCoverage
from optisample.optimize.layers.slots import SlotLayout
from optisample.optimize.plans import (
    GroupedInstrumentPlan,
    InstrumentPlan,
    StrategyPlan,
)
from optisample.optimize.plans.budget import (
    BudgetedPlanMixin,
    instrument_overhead,
    populated_instrument_bytes,
)
from optisample.optimize.reduce.summary import ReductionSummary
from trackmod.module.size import SizeReport

RULE_WIDTH: Final = 70
SECTION_RULE: Final = "=" * RULE_WIDTH
SUBSECTION_RULE: Final = "-" * RULE_WIDTH

_CURVE_ROWS: Final = 6
_SHORTFALL_ROWS: Final = 3  # shortfalls named in full before the rest are counted, keeping the block short
_HZ_PER_KHZ: Final = 1000.0
_COMPRESSED: Final = "on"
_UNCOMPRESSED: Final = "-"


def _counted(count: int, noun: str) -> str:
    """``count`` beside its noun, taking the plural ``s`` where the count asks for one."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _format_allocation_table(title: str, header: str, rows: Iterable[str]) -> str:
    """A titled, ruled allocation table: heading, subsection rule, column header, then the data rows."""
    return "\n".join((title, SUBSECTION_RULE, header, *rows))


def _compression_mark(params: EncodingParams) -> str:
    """How an allocation row states whether its encoding was compressed on the way to the quantizer."""
    return _COMPRESSED if params.compress else _UNCOMPRESSED


def _format_shortfalls(reduction: ReductionSummary) -> list[str]:
    """Name the kept recordings that run shorter than their pitch's longest note, the rest counted.

    A recording that falls short is scored over as much of the note as was recorded, so the objective
    covers less material than the plan claims; saying which keys they are is what makes that visible.
    """
    shortfalls = reduction.shortfalls
    if not shortfalls:
        return []

    lines = [f"  ! {len(shortfalls)} of {reduction.kept_recordings} kept recordings hold less than asked:"]
    lines += [
        f"      {recording.key.label:<22} {recording.duration_s:5.2f}s of {recording.required_duration_s:5.2f}s"
        for recording in shortfalls[:_SHORTFALL_ROWS]
    ]
    remainder = len(shortfalls) - _SHORTFALL_ROWS
    if remainder > 0:
        lines.append(f"      (and {remainder} more)")

    return lines


def _format_grid(reduction: ReductionSummary) -> str:
    """The full stored grid against the shortlist each played pitch keeps of it, and the band behind it.

    The rate span states what the kept representatives' own content and the playback ceiling justify
    storing, which is the measurement the shortlist is drawn around. An instrument whose material plays
    nothing has no pitch to narrow, so the line reports the grid alone.
    """
    grids = reduction.grids
    lead = f"Stored grid: {reduction.grid_size:>5} encodings  ->  "
    if not grids:
        lead += "swept once the material plays a pitch"
        return lead

    rates = [grid.useful_rate_hz / _HZ_PER_KHZ for grid in grids]
    return (
        f"{lead}{reduction.shortlisted / len(grids):>5.1f} shortlisted per key  "
        f"({min(rates):.1f}-{max(rates):.1f} kHz useful)"
    )


def format_reduction_block(reduction: ReductionSummary) -> str:
    """The pre-optimization stage's own summary: how much smaller each axis of the problem got.

    Three lines, one per axis -- the recorded grid, the material, and the stored encoding grid -- each
    reading ``before -> after``, so the search space the allocation was handed is stated alongside the
    allocation itself. Any recording too short for its material is called out under them.
    """
    lines = [
        "Reduction (pre-optimization)",
        SUBSECTION_RULE,
        f"Recordings:  {reduction.listed_recordings:>5} listed     ->  {reduction.kept_recordings:>5} kept",
        f"Material:    {reduction.played_notes:>5} notes      ->  {reduction.scored_classes:>5} scored classes",
        _format_grid(reduction),
    ]
    return "\n".join(lines + _format_shortfalls(reduction))


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
        f"(overhead {instrument_overhead(storage, plan.budget.instruments)} B: file {storage.file} + "
        f"{plan.budget.instruments} x instrument {populated_instrument_bytes(storage)})",
        f"Used:      {bytes_to_kib(used):7.1f} KiB samples  ({fraction:6.1%} of budget, "
        f"{bytes_to_kib(headroom):.1f} KiB free)  ->  {bytes_to_kib(plan.module_bytes):.1f} KiB module",
        f"Written:   {bytes_to_kib(size.total):7.1f} KiB module  "
        f"(records {size.headers} B + samples {size.pcm} B + patterns {size.patterns} B)",
    ]


def format_keyboard_line(coverage: KeyCoverage) -> str:
    """What of the keyboard the written instruments answer, and how much of it the fill supplied.

    An instrument is recorded over the keys its material plays and answers the rest from the recording
    nearest each one, so the line reads as the stretch played against the stretch playable.
    """
    return (
        f"Keyboard:  {coverage.answered:>7} of {coverage.numbered} keys answered  "
        f"({coverage.played} played, {coverage.filled} filled from the nearest recording)"
    )


def format_instruments_line(plan: BudgetedPlanMixin, layout: SlotLayout) -> str:
    """How many instruments the budget reserved against how many the plan was written as.

    The reserve is taken before the allocation runs, at the most instruments the layers' keys could fill
    (:func:`~optisample.optimize.layers.slots.reserved_slots`), so a plan storing zones rather than single
    keys comes out as fewer and the bytes the difference held back stay free.
    """
    reserved = plan.budget.instruments
    spare = (reserved - layout.count) * populated_instrument_bytes(plan.budget.storage)
    return (
        f"Instruments: {reserved:>5} reserved  ->  {layout.count} written  "
        f"({bytes_to_kib(spare):.1f} KiB of the reserve left free)"
    )


def _format_header(title: str, blocks: Sequence[str], summary: str) -> str:
    """Title, section rule, the blocks shared by both strategies, then a strategy-specific summary."""
    return "\n".join((title, SECTION_RULE, *blocks, summary))


def _shared_blocks(plan: BudgetedPlanMixin, size: SizeReport, coverage: KeyCoverage, layout: SlotLayout) -> list[str]:
    """What every report opens with: the byte budget, the keyboard answered and the instruments written."""
    return [*format_budget_block(plan, size), format_keyboard_line(coverage), format_instruments_line(plan, layout)]


def _weighting_note(plan: StrategyPlan) -> str:
    """How the objective weighed each note, which is what makes two objectives comparable."""
    return f"energy^{plan.energy_exponent:g}-weighted"


def _ungrouped_header(plan: InstrumentPlan, size: SizeReport, coverage: KeyCoverage, layout: SlotLayout) -> str:
    return _format_header(
        f"Instrument {plan.instrument_id!r} - budget solver (method: {plan.method})",
        _shared_blocks(plan, size, coverage, layout),
        f"Objective: {plan.objective:8.4f}  ({_weighting_note(plan)}, over "
        f"{len(plan.pitches)} pitches, {plan.total_weight:.1f} s of material)",
    )


def _format_pitches(plan: InstrumentPlan) -> str:
    header = (
        f"{'pitch':>5}  {'note':>4}  {'weight(s)':>9}  {'rep.vel':>7}  {'rate(Hz)':>8}  "
        f"{'depth':>5}  {'comp':>4}  {'size(KiB)':>9}  {'distortion':>10}  {'hull':>4}"
    )
    rows = []
    for pitch in plan.pitches:
        point = pitch.chosen
        rows.append(
            f"{pitch.pitch:>5}  {note_name(pitch.pitch):>4}  {pitch.weight:>9.1f}  "
            f"{pitch.representative_key.velocity:>7}  {point.params.target_rate:>8}  {point.params.depth_bits:>5}  "
            f"{_compression_mark(point.params):>4}  {bytes_to_kib(point.stored_bytes):>9.1f}  "
            f"{point.distortion:>10.4f}  {len(pitch.hull):>4}"
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


def format_report(plan: InstrumentPlan, size: SizeReport, coverage: KeyCoverage, layout: SlotLayout) -> str:
    """Render a human-readable summary of an instrument optimization."""
    sections = (
        _ungrouped_header(plan, size, coverage, layout),
        format_reduction_block(plan.reduction),
        _format_pitches(plan),
        _format_velocity_map(plan),
        _format_curve(plan),
    )
    return "\n\n".join(sections) + "\n"


def _grouped_header(plan: GroupedInstrumentPlan, size: SizeReport, coverage: KeyCoverage, layout: SlotLayout) -> str:
    return _format_header(
        f"Instrument {plan.instrument_id!r} - pitch-zone grouping (exact partition + allocation DP)",
        _shared_blocks(plan, size, coverage, layout),
        f"Grouping:  {_counted(len(plan.zones), 'zone')} over {_counted(len(plan.pitches), 'key')} and "
        f"{_counted(plan.layers.count, 'velocity layer')}  "
        f"(objective {plan.objective:.4f}, {_weighting_note(plan)}, "
        f"over {plan.total_weight:.1f} s of material)",
    )


def _format_instruments(layout: SlotLayout) -> str:
    """The instruments the plan is written as, priced one row each.

    A row is what a written instrument answers for and what it cost: the dynamics it holds, the keys its
    samples were stored for, the bytes they spend and the share of the objective they carry. The rows sum
    to the plan's own totals, which is how a split reads against the plainer plan it beat, and a band
    written as several instruments states each one's own stretch of keyboard.
    """
    header = (
        f"{'id':>3}  {'band':>9}  {'span':>9}  {'keys':>5}  {'samples':>7}  {'size(KiB)':>9}  "
        f"{'weight(s)':>9}  {'objective':>10}"
    )
    rows = [
        f"{index:>3}  {slot.band.label:>9}  {slot.span:>9}  {slot.keys:>5}  {len(slot.samples):>7}  "
        f"{bytes_to_kib(slot.stored_bytes):>9.1f}  {slot.weight:>9.1f}  {slot.objective_share:>10.4f}"
        for index, slot in enumerate(layout.slots)
    ]
    return _format_allocation_table("Instruments (a note's dynamic and pitch pick the one it plays)", header, rows)


def _format_zones(plan: GroupedInstrumentPlan) -> str:
    header = (
        f"{'layer':>5}  {'keys':>11}  {'rep':>4}  {'rep.vel':>7}  {'rate(Hz)':>8}  {'depth':>5}  "
        f"{'comp':>4}  {'size(KiB)':>9}  {'distortion':>10}  {'options':>7}"
    )
    rows = []
    for zone in plan.zones:
        option = zone.chosen
        keys = f"{zone.pitches[0]}-{zone.pitches[-1]}" if len(zone.pitches) > 1 else str(zone.pitches[0])
        span = f"{keys} ({len(zone.pitches)})"
        rows.append(
            f"{zone.layer:>5}  {span:>11}  {zone.representative:>4}  {zone.representative_key.velocity:>7}  "
            f"{option.params.target_rate:>8}  {option.params.depth_bits:>5}  "
            f"{_compression_mark(option.params):>4}  {bytes_to_kib(option.stored_bytes):>9.1f}  "
            f"{option.distortion:>10.4f}  {len(zone.hull):>7}"
        )
    return _format_allocation_table("Zones (one stored sample each, repitched across the zone's keys)", header, rows)


def format_grouping_report(
    plan: GroupedInstrumentPlan, size: SizeReport, coverage: KeyCoverage, layout: SlotLayout
) -> str:
    """Render a human-readable summary of a grouped optimization."""
    sections = (
        _grouped_header(plan, size, coverage, layout),
        format_reduction_block(plan.reduction),
        _format_instruments(layout),
        _format_zones(plan),
    )
    return "\n\n".join(sections) + "\n"
