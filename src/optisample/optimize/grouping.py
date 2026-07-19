"""Pitch-zone grouping: cover a keyboard with fewer stored samples by repitching representatives.

IT stores one sample per key, but nothing forces every key to own a *distinct* sample -- a tracker
happily repitches one recording across a range of keys (that is what ``C5Speed`` + the note map are
for). So when the byte budget is tight we partition the used pitches into **contiguous zones**, store
one representative recording per zone, and let the player transpose it to the zone's other keys. Each
sample we drop saves its PCM *and* its 80-byte header; what we pay is repitching artifacts plus the
timbre a single recording cannot match across the zone.

The method has three parts, all reusing the rate-distortion machinery of P2/P3:

* **Representative selection folded into the RD options.** For a zone we enumerate every member as a
  candidate representative crossed with every encoding, giving ``(bytes, distortion)`` points. At the
  minimum-byte-pressure end the representative that reconstructs the whole zone best is exactly
  k-medoids/PAM (the medoid is the member minimizing within-zone distortion); a tighter budget may
  prefer a different member/encoding, so the choice is made jointly with allocation, not before it.
* **Partitioning + allocation by one exact DP.** A byte-indexed Bellman DP over the ordered pitches
  chooses the zone boundaries *and* each zone's ``(representative, encoding)`` at once:
  ``dp[j][b]`` = least distortion covering the first ``j`` pitches in exactly ``b`` bytes. This is the
  multiple-choice knapsack of :mod:`optisample.optimize.knapsack` with an extra partition axis, and
  like ``solve_exact`` it is exact -- no Lagrangian duality gap. That matters here: forcing every
  zone to a single pitch recovers the ungrouped P3 allocation exactly, so an *exact* solver can never
  return a grouping worse than ungrouped at the same budget (an approximate one can, and does).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.surrogate import EncodeContext, EncodingParams, default_encode_config, encode, semitone_ratio
from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, bytes_to_kib, kib_to_bytes
from optisample.model import InstrumentSpec
from optisample.optimize.knapsack import BudgetInfeasibleError
from optisample.optimize.operating_points import default_rates, lower_convex_hull
from optisample.optimize.orchestrate import (
    BudgetBreakdown,
    OptimizeSettings,
    format_budget_block,
    load_instrument_audio,
    prepare_run,
)
from optisample.optimize.tasks import AudioMap, EvalContext, PitchTask, score_reconstruction
from optisample.optimize.velocity_map import VelocityVolumeMap

_Range = tuple[int, int]  # half-open [i, j) index range into the ordered pitch tasks
_ZoneOptions = dict[_Range, tuple["ZoneOption", ...]]


@dataclass(frozen=True)
class ZoneOption:
    """One way to realize a zone: which member is the stored representative and how it is encoded.

    ``distortion`` is the *usage-weighted, summed* reconstruction distortion over every pitch the
    zone covers (so it is directly comparable to the ungrouped objective), and ``stored_bytes``
    already includes the one 80-byte sample header the zone costs.
    """

    representative: int
    params: EncodingParams
    stored_bytes: int
    distortion: float
    frames: int


@dataclass(frozen=True)
class Zone:
    """A contiguous run of keys served by one stored sample, with the option the solver chose."""

    pitches: tuple[int, ...]
    representative: int
    representative_velocity: int
    weight: float  # total material usage (seconds) across the zone's pitches
    chosen: ZoneOption
    hull: tuple[ZoneOption, ...]


@dataclass(frozen=True)
class GroupingResult:
    """The solver's output: the chosen zones and the totals they add up to."""

    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float


@dataclass(frozen=True)
class GroupedInstrumentPlan:
    """A full grouped optimization: the velocity map, the zones, and the budget they fit within."""

    instrument_id: str
    budget: BudgetBreakdown
    velocity_map: VelocityVolumeMap
    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float

    @property
    def used_bytes(self) -> int:
        return self.total_bytes

    @property
    def sample_budget_bytes(self) -> int:
        return self.budget.sample_bytes

    @property
    def module_budget_bytes(self) -> int:
        return self.budget.module_bytes

    @property
    def module_bytes(self) -> int:
        return self.used_bytes + FILE_HEADER_BYTES + INSTRUMENT_HEADER_BYTES

    @property
    def pitches(self) -> tuple[int, ...]:
        """Every covered key, ascending (the zones already partition them in order)."""
        return tuple(pitch for zone in self.zones for pitch in zone.pitches)

    @property
    def total_weight(self) -> float:
        return sum(zone.weight for zone in self.zones)


# --- zone cost model -----------------------------------------------------------------------------


def _zone_trim(range_tasks: Sequence[PitchTask], representative: int) -> float:
    """Stored duration the representative needs so every covered key's longest note plays in full.

    Playing key ``p`` from a sample rooted at ``representative`` runs it at ``2**((p - rep)/12)`` times
    speed, so a higher key consumes stored frames faster and needs a proportionally longer sample.
    """
    return max(task.max_duration_s * semitone_ratio(task.pitch - representative) for task in range_tasks)


def _zone_options(range_tasks: Sequence[PitchTask], ctx: EvalContext) -> list[ZoneOption]:
    """Every ``(representative, encoding)`` for one candidate zone, with its cost and total distortion."""
    rates = ctx.grid.rates if ctx.grid.rates is not None else default_rates(ctx.sample_rate)
    options: list[ZoneOption] = []
    for rep_task in range_tasks:  # k-medoids candidates: each covered key's own recording
        representative = rep_task.pitch
        trim_s = _zone_trim(range_tasks, representative)
        for loop in ctx.grid.loops:
            for depth in ctx.grid.depths:
                for rate in rates:
                    params = EncodingParams(rate, depth, trim_s, ctx.grid.dither, ctx.grid.noise_shaping, loop)
                    # transitional: EncodeConfig is threaded through EvalContext in phase 6.
                    encode_ctx = EncodeContext(root_pitch=representative, config=default_encode_config(), rng=ctx.rng)
                    stored = encode(rep_task.representative, ctx.sample_rate, params, encode_ctx)
                    distortion = sum(task.weight * score_reconstruction(stored, task, ctx) for task in range_tasks)
                    options.append(ZoneOption(representative, params, stored.stored_bytes, distortion, stored.frames))
    return options


def build_zone_options(tasks: Sequence[PitchTask], ctx: EvalContext) -> _ZoneOptions:
    """Score every contiguous pitch range ``[i, j)`` -- the menu the partition+allocation DP chooses from.

    This is the expensive step (an encode + reconstruction score per representative, encoding and
    covered pitch); the DP that consumes it is cheap.
    """
    count = len(tasks)
    options: _ZoneOptions = {}
    for i in range(count):
        for j in range(i + 1, count + 1):
            options[(i, j)] = tuple(_zone_options(tasks[i:j], ctx))
    return options


def zone_hull(options: Sequence[ZoneOption]) -> list[ZoneOption]:
    """A zone's byte-vs-distortion frontier over its ``(representative, encoding)`` options.

    Delegates to the shared :func:`optisample.optimize.operating_points.lower_convex_hull`; kept as a
    named entry point because this is where a zone's representative selection becomes visible (each
    hull vertex is the best member-plus-encoding at its byte level). Used for reporting.
    """
    return lower_convex_hull(options)


# --- exact partition + allocation DP -------------------------------------------------------------


def _cheapest_partition_bytes(options: _ZoneOptions, count: int) -> int:
    """Fewest bytes any partition can use (each zone at its smallest option) -- the feasibility floor."""
    dp = [0] + [np.iinfo(np.int64).max] * count
    for j in range(1, count + 1):
        for i in range(j):
            cheapest = min(option.stored_bytes for option in options[(i, j)])
            dp[j] = min(dp[j], dp[i] + cheapest)
    return int(dp[count])


def _forward_dp(
    options: _ZoneOptions, count: int, budget_bytes: int
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.int64]], list[NDArray[np.int64]]]:
    """Fill ``dp[j][b]`` = least distortion covering the first ``j`` pitches in exactly ``b`` bytes."""
    size = budget_bytes + 1
    dp = [np.full(size, np.inf, dtype=np.float64) for _ in range(count + 1)]
    dp[0][0] = 0.0
    from_i = [np.full(size, -1, dtype=np.int64) for _ in range(count + 1)]
    from_opt = [np.full(size, -1, dtype=np.int64) for _ in range(count + 1)]
    for j in range(1, count + 1):
        for i in range(j):
            for index, option in enumerate(options[(i, j)]):
                cost = option.stored_bytes
                if cost > budget_bytes:
                    continue
                candidate = dp[i][: size - cost] + option.distortion
                target = dp[j][cost:]
                improved = candidate < target
                target[improved] = candidate[improved]
                from_i[j][cost:][improved] = i
                from_opt[j][cost:][improved] = index
    return dp, from_i, from_opt


def _reconstruct(
    options: _ZoneOptions, from_i: list[NDArray[np.int64]], from_opt: list[NDArray[np.int64]], count: int, total: int
) -> list[tuple[int, int, ZoneOption]]:
    """Walk the DP backpointers from ``(count, total)`` back to ``(0, 0)`` to recover the zones."""
    segments: list[tuple[int, int, ZoneOption]] = []
    j, budget = count, total
    while j > 0:
        i = int(from_i[j][budget])
        option = options[(i, j)][int(from_opt[j][budget])]
        segments.append((i, j, option))
        budget -= option.stored_bytes
        j = i
    segments.reverse()
    return segments


def _build_zone(
    tasks: Sequence[PitchTask], span: _Range, chosen: ZoneOption, zone_options: Sequence[ZoneOption]
) -> Zone:
    """Attach the covered keys, the representative's stored velocity and the RD hull to a chosen zone."""
    range_tasks = tasks[span[0] : span[1]]
    rep_task = next(task for task in range_tasks if task.pitch == chosen.representative)
    return Zone(
        pitches=tuple(task.pitch for task in range_tasks),
        representative=chosen.representative,
        representative_velocity=rep_task.representative_velocity,
        weight=sum(task.weight for task in range_tasks),
        chosen=chosen,
        hull=tuple(zone_hull(zone_options)),
    )


def solve_grouping(tasks: Sequence[PitchTask], options: _ZoneOptions, budget_bytes: int) -> GroupingResult:
    """Exact partition + allocation: least-distortion set of zones whose bytes fit ``budget_bytes``."""
    count = len(tasks)
    if count == 0:
        return GroupingResult(zones=(), total_bytes=0, objective=0.0)
    cheapest = _cheapest_partition_bytes(options, count)
    if cheapest > budget_bytes:
        raise BudgetInfeasibleError(cheapest, budget_bytes)
    dp, from_i, from_opt = _forward_dp(options, count, budget_bytes)
    reachable = np.flatnonzero(np.isfinite(dp[count]))
    best_bytes = int(reachable[int(np.argmin(dp[count][reachable]))])
    segments = _reconstruct(options, from_i, from_opt, count, best_bytes)
    zones = tuple(_build_zone(tasks, (i, j), option, options[(i, j)]) for i, j, option in segments)
    return GroupingResult(zones=zones, total_bytes=best_bytes, objective=float(dp[count][best_bytes]))


# --- orchestration -------------------------------------------------------------------------------


def optimize_instrument_grouped(
    instrument: InstrumentSpec, audio: AudioMap, sample_rate: int, settings: OptimizeSettings = OptimizeSettings()
) -> GroupedInstrumentPlan:
    """Optimize one instrument with pitch-zone grouping and return a structured plan.

    Reuses the ungrouped pipeline's velocity map and per-pitch tasks; only the allocation changes
    (zones of repitched representatives instead of one sample per key). ``settings.method`` is not
    used -- grouping always solves partition and allocation together with the exact DP.
    """
    velocity_map, ctx, tasks = prepare_run(instrument, audio, sample_rate, settings)
    options = build_zone_options(tasks, ctx)

    module_budget = kib_to_bytes(instrument.budget_kb)
    sample_budget = module_budget - FILE_HEADER_BYTES - INSTRUMENT_HEADER_BYTES
    result = solve_grouping(tasks, options, sample_budget)

    return GroupedInstrumentPlan(
        instrument_id=instrument.id,
        budget=BudgetBreakdown(module_bytes=module_budget, sample_bytes=sample_budget),
        velocity_map=velocity_map,
        zones=result.zones,
        total_bytes=result.total_bytes,
        objective=result.objective,
    )


def run_instrument_grouped(
    instrument: InstrumentSpec, settings: OptimizeSettings = OptimizeSettings()
) -> GroupedInstrumentPlan:
    """Load an instrument's recordings from disk and optimize it with grouping."""
    audio, sample_rate = load_instrument_audio(instrument)
    return optimize_instrument_grouped(instrument, audio, sample_rate, settings)


# --- reporting -----------------------------------------------------------------------------------


def _format_header(plan: GroupedInstrumentPlan) -> str:
    return "\n".join(
        (
            f"Instrument {plan.instrument_id!r} - pitch-zone grouping (exact partition + allocation DP)",
            "=" * 70,
            *format_budget_block(plan),
            f"Grouping:  {len(plan.zones)} zones cover {len(plan.pitches)} keys  "
            f"(objective {plan.objective:.4f} over {plan.total_weight:.1f} s of material)",
        )
    )


def _format_zones(plan: GroupedInstrumentPlan) -> str:
    lines = [
        "Zones (one stored sample each, repitched across the zone's keys)",
        "-" * 70,
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
    return "\n\n".join((_format_header(plan), _format_zones(plan))) + "\n"
