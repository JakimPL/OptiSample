from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optisample.optimize.dp import require_feasible
from optisample.optimize.grouping.cost_model import (
    _Range,
    _ZoneOptions,
    zone_hull,
    zone_starts,
)
from optisample.optimize.plans.grouped import GroupingResult, Zone, ZoneOption
from optisample.optimize.tasks import PitchTask


@dataclass(frozen=True)
class _Segment:
    """One recovered zone: the half-open pitch-index range ``[start, stop)`` and its chosen option."""

    start: int
    stop: int
    option: ZoneOption


def _cheapest_partition_bytes(options: _ZoneOptions, starts: Sequence[Sequence[int]]) -> int:
    """Fewest bytes any partition can use (each zone at its smallest option) -- the feasibility floor."""
    count = len(starts) - 1
    dp = [0] + [np.iinfo(np.int64).max] * count
    for stop in range(1, count + 1):
        for start in starts[stop]:
            cheapest = min(option.stored_bytes for option in options[(start, stop)])
            dp[stop] = min(dp[stop], dp[start] + cheapest)

    return int(dp[count])


def _relax(dp_from: NDArray[np.float64], dp_to: NDArray[np.float64], option: ZoneOption) -> NDArray[np.bool_]:
    """Extend every byte total reached in ``dp_from`` by one more zone stored as ``option``.

    Writes the improved distortions into ``dp_to`` in place and reports where they landed, so the caller
    can point those same byte totals back at the zone that reached them.
    """
    cost = option.stored_bytes
    candidate = dp_from[: dp_to.size - cost] + option.distortion
    target = dp_to[cost:]
    improved = candidate < target
    target[improved] = candidate[improved]
    return np.asarray(improved, dtype=np.bool_)


def _forward_dp(
    options: _ZoneOptions,
    starts: Sequence[Sequence[int]],
    budget_bytes: int,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.int64]], list[NDArray[np.int64]]]:
    """Fill ``dp[j][b]`` = least distortion covering the first ``j`` pitches in exactly ``b`` bytes."""
    count = len(starts) - 1
    size = budget_bytes + 1
    dp = [np.full(size, np.inf, dtype=np.float64) for _ in range(count + 1)]
    dp[0][0] = 0.0
    from_start = [np.full(size, -1, dtype=np.int64) for _ in range(count + 1)]
    from_option = [np.full(size, -1, dtype=np.int64) for _ in range(count + 1)]
    for stop in range(1, count + 1):
        for start in starts[stop]:
            for index, option in enumerate(options[(start, stop)]):
                if option.stored_bytes > budget_bytes:
                    continue

                improved = _relax(dp[start], dp[stop], option)
                from_start[stop][option.stored_bytes :][improved] = start
                from_option[stop][option.stored_bytes :][improved] = index

    return dp, from_start, from_option


def _reconstruct(
    options: _ZoneOptions,
    from_start: list[NDArray[np.int64]],
    from_option: list[NDArray[np.int64]],
    count: int,
    total: int,
) -> list[_Segment]:
    """Walk the DP backpointers from ``(count, total)`` back to ``(0, 0)`` to recover the zones."""
    segments: list[_Segment] = []
    stop, budget = count, total
    while stop > 0:
        start = int(from_start[stop][budget])
        option = options[(start, stop)][int(from_option[stop][budget])]
        segments.append(_Segment(start=start, stop=stop, option=option))
        budget -= option.stored_bytes
        stop = start

    segments.reverse()
    return segments


def _build_zone(
    tasks: Sequence[PitchTask],
    span: _Range,
    chosen: ZoneOption,
    zone_options: Sequence[ZoneOption],
) -> Zone:
    """Attach the covered keys, the representative's stored velocity and the RD hull to a chosen zone."""
    range_tasks = tasks[span[0] : span[1]]
    rep_task = next(task for task in range_tasks if task.pitch == chosen.representative)
    return Zone(
        pitches=tuple(task.pitch for task in range_tasks),
        representative_key=rep_task.representative_key,
        weight=sum(task.weight for task in range_tasks),
        chosen=chosen,
        hull=tuple(zone_hull(zone_options)),
    )


def solve_grouping(
    tasks: Sequence[PitchTask],
    options: _ZoneOptions,
    budget_bytes: int,
) -> GroupingResult:
    """Exact partition + allocation: least-distortion set of zones whose bytes fit ``budget_bytes``.

    ``options`` holds the candidate zones the cost model scored, which the span cap leaves sparse, so
    the passes below step between the range boundaries :func:`zone_starts` reports as connected.
    """
    count = len(tasks)
    if count == 0:
        return GroupingResult(zones=(), total_bytes=0, objective=0.0)
    starts = zone_starts(options, count)
    require_feasible(_cheapest_partition_bytes(options, starts), budget_bytes)
    dp, from_start, from_option = _forward_dp(options, starts, budget_bytes)
    reachable = np.flatnonzero(np.isfinite(dp[count]))
    best_bytes = int(reachable[int(np.argmin(dp[count][reachable]))])
    segments = _reconstruct(options, from_start, from_option, count, best_bytes)
    zones = tuple(
        _build_zone(
            tasks,
            (segment.start, segment.stop),
            segment.option,
            options[(segment.start, segment.stop)],
        )
        for segment in segments
    )
    return GroupingResult(
        zones=zones,
        total_bytes=best_bytes,
        objective=float(dp[count][best_bytes]),
    )
