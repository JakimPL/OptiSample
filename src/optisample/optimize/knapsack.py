"""Budget allocation as a Multiple-Choice Knapsack Problem (MCKP).

With grouping fixed, each stored sample must take exactly one encoding configuration; we choose one
per sample to minimize the material-weighted distortion under a hard byte budget::

    minimize   sum_j  weight_j * distortion_j(config)
    subject to sum_j  bytes_j(config)  <=  budget

Two solvers, both consuming the P2 operating points:

* :func:`solve_exact` -- a pseudo-polynomial dynamic program over bytes. Optimal over *all*
  candidate configs (not just the hull), at the cost of an ``O(items * configs * budget)`` table.
* :func:`solve_lagrangian` / :func:`rd_curve` -- the rate-distortion Lagrangian sweep (Shoham &
  Gersho 1988; Ortega-Ramchandran 1998). Each sample keeps only its lower-convex-hull configs;
  sweeping the slope ``lambda`` from steep to shallow greedily upgrades whichever sample buys the
  most distortion-per-byte next, tracing the *whole* budget->quality curve in one pass. It is exact
  at the curve's breakpoints and near-optimal between them (the classic Lagrangian duality gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisample.optimize.dp import require_feasible
from optisample.optimize.operating_points import OperatingPoint, lower_convex_hull


@dataclass(frozen=True)
class KnapsackItem:
    """One sample's decision: pick one of ``points`` (its configs), weighted by material usage."""

    key: str
    weight: float
    points: tuple[OperatingPoint, ...]


@dataclass(frozen=True)
class Selection:
    """The config chosen for one item, with its usage weight (for the report)."""

    key: str
    weight: float
    point: OperatingPoint


@dataclass(frozen=True)
class Allocation:
    """A full assignment of one config per item, with its total cost and weighted distortion."""

    selections: tuple[Selection, ...]
    total_bytes: int
    objective: float  # sum_j weight_j * distortion_j


@dataclass(frozen=True)
class RDCurvePoint:
    """One vertex of the achievable budget->quality curve from the Lagrangian sweep."""

    lam: float  # slope threshold; this point is optimal for budgets it just fits under
    total_bytes: int
    objective: float
    indices: tuple[int, ...]  # chosen hull-vertex index per item


def _cheapest_total(items: tuple[KnapsackItem, ...]) -> int:
    """Bytes of the minimum-cost allocation (each item at its smallest config)."""
    return sum(min(point.stored_bytes for point in item.points) for item in items)


def _forward_dp(items: tuple[KnapsackItem, ...], budget_bytes: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Fill ``dp[b]`` = min objective at total cost exactly ``b``; return it and per-item choices."""
    size = budget_bytes + 1
    dp = np.full(size, np.inf, dtype=np.float64)
    dp[0] = 0.0
    choices: list[np.ndarray] = []
    for item in items:
        new_dp = np.full(size, np.inf, dtype=np.float64)
        choice = np.full(size, -1, dtype=np.int64)
        for index, point in enumerate(item.points):
            cost = point.stored_bytes
            if cost > budget_bytes:
                continue
            candidate = dp[: size - cost] + item.weight * point.distortion
            target = new_dp[cost:]
            improved = candidate < target
            target[improved] = candidate[improved]
            choice[cost:][improved] = index
        dp = new_dp
        choices.append(choice)
    return dp, choices


def _reconstruct(items: tuple[KnapsackItem, ...], choices: list[np.ndarray], total_bytes: int) -> tuple[Selection, ...]:
    """Walk the DP backpointers from ``total_bytes`` to recover one selection per item."""
    selections: list[Selection] = []
    budget = total_bytes
    for item, choice in zip(reversed(items), reversed(choices)):
        index = int(choice[budget])
        point = item.points[index]
        selections.append(Selection(key=item.key, weight=item.weight, point=point))
        budget -= point.stored_bytes
    selections.reverse()
    return tuple(selections)


def solve_exact(items: tuple[KnapsackItem, ...], budget_bytes: int) -> Allocation:
    """Exact MCKP: the min-distortion assignment of one config per item within ``budget_bytes``."""
    if not items:
        return Allocation(selections=(), total_bytes=0, objective=0.0)
    require_feasible(_cheapest_total(items), budget_bytes)
    dp, choices = _forward_dp(items, budget_bytes)
    reachable = np.flatnonzero(np.isfinite(dp))
    best_bytes = int(reachable[int(np.argmin(dp[reachable]))])
    selections = _reconstruct(items, choices, best_bytes)
    return Allocation(selections=selections, total_bytes=best_bytes, objective=float(dp[best_bytes]))


def _hulls(items: tuple[KnapsackItem, ...]) -> list[tuple[OperatingPoint, ...]]:
    """Each item reduced to its lower-convex-hull configs (ascending bytes, descending distortion)."""
    return [tuple(lower_convex_hull(item.points)) for item in items]


def _hull_upgrades(
    items: tuple[KnapsackItem, ...], hulls: list[tuple[OperatingPoint, ...]]
) -> list[tuple[float, int, int]]:
    """Every single-step hull upgrade as ``(slope, item, step)``, steepest distortion-per-byte first.

    Each hull is ordered cheapest-first, so stepping vertex ``k`` -> ``k+1`` spends ``delta_bytes`` more
    to save ``weight * delta_distortion`` distortion. Sorting by that slope descending is exactly the
    order the Lagrangian sweep applies upgrades: the best saving per byte is always bought next.
    """
    upgrades: list[tuple[float, int, int]] = []
    for item_pos, (item, hull) in enumerate(zip(items, hulls)):
        for step in range(len(hull) - 1):
            delta_bytes = hull[step + 1].stored_bytes - hull[step].stored_bytes
            delta_distortion = hull[step].distortion - hull[step + 1].distortion
            upgrades.append((item.weight * delta_distortion / delta_bytes, item_pos, step))
    upgrades.sort(key=lambda upgrade: upgrade[0], reverse=True)
    return upgrades


def _lagrangian_curve(
    items: tuple[KnapsackItem, ...],
) -> tuple[list[RDCurvePoint], list[tuple[OperatingPoint, ...]]]:
    """Trace the budget->quality curve: start all-cheapest, then apply hull upgrades steepest-first."""
    hulls = _hulls(items)
    index = [0] * len(items)
    total_bytes = sum(hull[0].stored_bytes for hull in hulls)
    objective = sum(item.weight * hull[0].distortion for item, hull in zip(items, hulls))

    curve = [RDCurvePoint(lam=np.inf, total_bytes=total_bytes, objective=objective, indices=tuple(index))]
    for lam, item_pos, step in _hull_upgrades(items, hulls):
        hull = hulls[item_pos]
        total_bytes += hull[step + 1].stored_bytes - hull[step].stored_bytes
        objective -= items[item_pos].weight * (hull[step].distortion - hull[step + 1].distortion)
        index[item_pos] += 1
        curve.append(RDCurvePoint(lam=lam, total_bytes=total_bytes, objective=objective, indices=tuple(index)))
    return curve, hulls


def rd_curve(items: tuple[KnapsackItem, ...]) -> list[RDCurvePoint]:
    """The full achievable budget->quality curve (ascending bytes, descending objective)."""
    if not items:
        return [RDCurvePoint(lam=np.inf, total_bytes=0, objective=0.0, indices=())]
    return _lagrangian_curve(items)[0]


def solve_lagrangian(items: tuple[KnapsackItem, ...], budget_bytes: int) -> Allocation:
    """Near-optimal MCKP via the Lagrangian sweep: the richest hull point that fits ``budget_bytes``.

    The curve's first point is the all-cheapest allocation (checked for feasibility), and byte cost
    rises monotonically along it, so the last point within budget is the highest-quality feasible one.
    """
    if not items:
        return Allocation(selections=(), total_bytes=0, objective=0.0)
    curve, hulls = _lagrangian_curve(items)
    require_feasible(curve[0].total_bytes, budget_bytes)
    feasible = [point for point in curve if point.total_bytes <= budget_bytes]
    best = feasible[-1]
    selections = tuple(
        Selection(key=item.key, weight=item.weight, point=hull[vertex])
        for item, hull, vertex in zip(items, hulls, best.indices)
    )
    return Allocation(selections=selections, total_bytes=best.total_bytes, objective=best.objective)
