from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optisample.optimize.dp import require_feasible
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
) -> list[_Recovered]:
    """Walk the DP backpointers from ``(count, total)`` back to ``(0, 0)`` to recover the zones."""
    recovered: list[_Recovered] = []
    stop, budget = count, total
    while stop > 0:
        start = int(from_start[stop][budget])
        option = options[(start, stop)][int(from_option[stop][budget])]
        recovered.append(_Recovered(start=start, stop=stop, option=option))
        budget -= option.stored_bytes
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
    budget_bytes: int,
) -> GroupingResult:
    """Exact partition + allocation: least-distortion set of zones whose bytes fit ``budget_bytes``.

    The segments' keys are laid end to end into one axis and their option tables offset onto it, so the
    DP walks every layer in a single pass and the budget is shared across all of them. ``options`` holds
    the candidate zones the cost model scored, which the span cap and the segment boundaries leave
    sparse, so the passes below step between the range boundaries :func:`zone_starts` reports as
    connected.
    """
    tasks = axis_tasks(segments)
    count = len(tasks)
    if count == 0:
        return GroupingResult(zones=(), total_bytes=0, objective=0.0)
    combined = combined_options(segments, options)
    offsets = segment_offsets(segments)
    starts = zone_starts(combined, count)
    require_feasible(_cheapest_partition_bytes(combined, starts), budget_bytes)
    dp, from_start, from_option = _forward_dp(combined, starts, budget_bytes)
    reachable = np.flatnonzero(np.isfinite(dp[count]))
    best_bytes = int(reachable[int(np.argmin(dp[count][reachable]))])
    chosen = _reconstruct(combined, from_start, from_option, count, best_bytes)
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
        total_bytes=best_bytes,
        objective=float(dp[count][best_bytes]),
    )
