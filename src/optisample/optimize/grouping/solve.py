"""The exact partition + allocation DP over the zone-cost menu.

A byte-indexed Bellman DP over the ordered pitches chooses the zone boundaries *and* each zone's
``(representative, encoding)`` at once: ``dp[j][b]`` = least distortion covering the first ``j``
pitches in exactly ``b`` bytes. This is the multiple-choice knapsack of
:mod:`optisample.optimize.knapsack` with an extra partition axis, and like ``solve_exact`` it is exact
-- no Lagrangian duality gap. That matters: forcing every zone to a single pitch recovers the
ungrouped allocation exactly, so an *exact* solver can never return a grouping worse than ungrouped at
the same budget (an approximate one can, and does).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from optisample.optimize.dp import require_feasible
from optisample.optimize.grouping.cost_model import _Range, _ZoneOptions, zone_hull
from optisample.optimize.plans.grouped import GroupingResult, Zone, ZoneOption
from optisample.optimize.tasks import PitchTask


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
    require_feasible(_cheapest_partition_bytes(options, count), budget_bytes)
    dp, from_i, from_opt = _forward_dp(options, count, budget_bytes)
    reachable = np.flatnonzero(np.isfinite(dp[count]))
    best_bytes = int(reachable[int(np.argmin(dp[count][reachable]))])
    segments = _reconstruct(options, from_i, from_opt, count, best_bytes)
    zones = tuple(_build_zone(tasks, (i, j), option, options[(i, j)]) for i, j, option in segments)
    return GroupingResult(zones=zones, total_bytes=best_bytes, objective=float(dp[count][best_bytes]))
