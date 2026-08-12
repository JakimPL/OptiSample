from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optisample.optimize.dp import ByteGrid, require_feasible
from optisample.optimize.grouping.cost_model import (
    ZoneSegment,
    _Range,
    _ZoneOptions,
    axis_tasks,
    combined_options,
    segment_offsets,
    zone_hull,
    zone_starts,
)
from optisample.optimize.plans.grouped import GroupingResult, Zone, ZoneOption
from optisample.optimize.tasks import PitchTask


@dataclass(frozen=True)
class _Recovered:
    """One recovered zone: the half-open axis-index range ``[start, stop)`` and its chosen option."""

    start: int
    stop: int
    option: ZoneOption


@dataclass(frozen=True)
class _Walk:
    """What the forward pass reached, and how it got there.

    ``distortion[stop][total]`` is the least distortion covering the first ``stop`` keys of the axis for
    exactly ``total`` charged steps; the two backpointer tables name the zone the walk arrived through --
    where it started, and which of that span's options it took.
    """

    distortion: list[NDArray[np.float64]]
    from_start: list[NDArray[np.int64]]
    from_option: list[NDArray[np.int64]]

    @property
    def covered(self) -> NDArray[np.float64]:
        """The distortion each charged step total reaches once every key on the axis is covered."""
        return self.distortion[-1]


def _charged(option: ZoneOption, grid: ByteGrid, reserve: int) -> int:
    """What one zone costs the walk: the bytes it stores plus the reserve each stored sample is charged.

    Charging every stored sample above its own bytes is how a cap on the sample count is met: the walk
    then prefers fewer and wider zones, and the bytes the plan truly spends are read back off the options
    it chose (:func:`solve_grouping`).

    The answer is in the steps ``grid`` walks the budget in, taken up to the step that holds the bytes
    charged, so a partition the walk carries is one the budget carries.
    """
    return grid.cost(option.stored_bytes + reserve)


def _cheapest_partition_steps(
    options: _ZoneOptions,
    starts: Sequence[Sequence[int]],
    grid: ByteGrid,
    reserve: int,
) -> int:
    """Fewest charged steps any partition can use (each zone at its smallest option) -- the feasibility floor."""
    count = len(starts) - 1
    dp = [0] + [np.iinfo(np.int64).max] * count
    for stop in range(1, count + 1):
        for start in starts[stop]:
            cheapest = min(_charged(option, grid, reserve) for option in options[(start, stop)])
            dp[stop] = min(dp[stop], dp[start] + cheapest)

    return int(dp[count])


def _relax(
    dp_from: NDArray[np.float64],
    dp_to: NDArray[np.float64],
    cost: int,
    distortion: float,
) -> NDArray[np.bool_]:
    """Extend every byte total reached in ``dp_from`` by one more zone costing ``cost`` at ``distortion``.

    Writes the improved distortions into ``dp_to`` in place and reports where they landed, so the caller
    can point those same byte totals back at the zone that reached them.
    """
    candidate = dp_from[: dp_to.size - cost] + distortion
    target = dp_to[cost:]
    improved = candidate < target
    target[improved] = candidate[improved]
    return np.asarray(improved, dtype=np.bool_)


def _forward_dp(
    options: _ZoneOptions,
    starts: Sequence[Sequence[int]],
    grid: ByteGrid,
    reserve: int,
) -> _Walk:
    """Fill the walk: least distortion covering the first ``j`` pitches in exactly ``b`` charged steps."""
    count = len(starts) - 1
    size = grid.steps + 1
    distortion = [np.full(size, np.inf, dtype=np.float64) for _ in range(count + 1)]
    distortion[0][0] = 0.0
    walk = _Walk(
        distortion=distortion,
        from_start=[np.full(size, -1, dtype=np.int64) for _ in range(count + 1)],
        from_option=[np.full(size, -1, dtype=np.int64) for _ in range(count + 1)],
    )
    for stop in range(1, count + 1):
        for start in starts[stop]:
            for index, option in enumerate(options[(start, stop)]):
                cost = _charged(option, grid, reserve)
                if cost > grid.steps:
                    continue

                improved = _relax(walk.distortion[start], walk.distortion[stop], cost, option.distortion)
                walk.from_start[stop][cost:][improved] = start
                walk.from_option[stop][cost:][improved] = index

    return walk


def _reconstruct(options: _ZoneOptions, walk: _Walk, total: int, grid: ByteGrid, reserve: int) -> list[_Recovered]:
    """Walk the backpointers from the fully covered axis at ``total`` charged steps back to the start."""
    recovered: list[_Recovered] = []
    stop, budget = len(walk.distortion) - 1, total
    while stop > 0:
        start = int(walk.from_start[stop][budget])
        option = options[(start, stop)][int(walk.from_option[stop][budget])]
        recovered.append(_Recovered(start=start, stop=stop, option=option))
        budget -= _charged(option, grid, reserve)
        stop = start

    recovered.reverse()
    return recovered


def _build_zone(
    tasks: Sequence[PitchTask],
    span: _Range,
    layer: int,
    chosen: ZoneOption,
    zone_options: Sequence[ZoneOption],
) -> Zone:
    """Attach the covered keys, the layer, the representative's velocity and the RD hull to a zone."""
    range_tasks = tasks[span[0] : span[1]]
    rep_task = next(task for task in range_tasks if task.pitch == chosen.representative)
    return Zone(
        pitches=tuple(task.pitch for task in range_tasks),
        layer=layer,
        representative_key=rep_task.representative_key,
        weight=sum(task.weight for task in range_tasks),
        chosen=chosen,
        hull=tuple(zone_hull(zone_options)),
    )


def _layer_at(offsets: Sequence[int], position: int) -> int:
    """Which segment -- and so which layer -- the key at ``position`` on the axis came from."""
    return int(np.searchsorted(offsets, position, side="right")) - 1


def solve_grouping(
    segments: Sequence[ZoneSegment],
    options: Sequence[_ZoneOptions],
    grid: ByteGrid,
    *,
    reserve: int,
) -> GroupingResult:
    """Partition + allocation: least-distortion set of zones whose charged bytes fit the budget ``grid`` holds.

    The segments' keys are laid end to end into one axis and their option tables offset onto it, so the
    DP walks every layer in a single pass and the budget is shared across all of them. ``options`` holds
    the candidate zones the cost model scored, which the span cap and the segment boundaries leave
    sparse, so the passes below step between the range boundaries :func:`zone_starts` reports as
    connected.

    ``grid`` states the budget and the steps it is walked in, and every cost the walk prices is taken up
    to the step above (:func:`_charged`), so the answer is exact over the byte totals the grid states and
    fits the budget over every total there is. A grid of whole bytes states them all, which is the exact
    allocation.

    ``reserve`` is what each stored sample is charged on top of the bytes it occupies (:func:`_charged`),
    which is the price a caller raises to draw the walk toward fewer zones; :data:`NO_RESERVE` leaves
    every option at its own size. The zones answered are the ones the walk chose and ``total_bytes`` the
    bytes they truly store, so the reserve shapes the partition and the plan reports what it holds.

    Raises:
        BudgetInfeasibleError: when the cheapest partition's charged bytes overrun what the grid holds.
    """
    tasks = axis_tasks(segments)
    count = len(tasks)
    if count == 0:
        return GroupingResult(zones=(), total_bytes=0, objective=0.0)
    combined = combined_options(segments, options)
    offsets = segment_offsets(segments)
    starts = zone_starts(combined, count)
    cheapest = _cheapest_partition_steps(combined, starts, grid, reserve)
    require_feasible(grid.spent(cheapest), grid.usable_bytes)
    walk = _forward_dp(combined, starts, grid, reserve)
    reachable = np.flatnonzero(np.isfinite(walk.covered))
    best_steps = int(reachable[int(np.argmin(walk.covered[reachable]))])
    chosen = _reconstruct(combined, walk, best_steps, grid, reserve)
    zones = tuple(
        _build_zone(
            tasks,
            (zone.start, zone.stop),
            _layer_at(offsets, zone.start),
            zone.option,
            combined[(zone.start, zone.stop)],
        )
        for zone in chosen
    )
    return GroupingResult(
        zones=zones,
        total_bytes=sum(zone.option.stored_bytes for zone in chosen),
        objective=float(walk.covered[best_steps]),
    )
