from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


def _forward_dp(
    items: tuple[KnapsackItem, ...], budget_bytes: int
) -> tuple[NDArray[np.float64], list[NDArray[np.int64]]]:
    """Fill ``dp[b]`` = min objective at total cost exactly ``b``; return it and per-item choices."""
    size = budget_bytes + 1
    dp = np.full(size, np.inf, dtype=np.float64)
    dp[0] = 0.0
    choices: list[NDArray[np.int64]] = []
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


def _reconstruct(
    items: tuple[KnapsackItem, ...],
    choices: list[NDArray[np.int64]],
    total_bytes: int,
) -> tuple[Selection, ...]:
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


@dataclass(frozen=True)
class _HullUpgrade:
    """One cheapest-first hull step: item ``item_index`` moves to its next vertex for ``slope`` savings."""

    slope: float  # material-weighted distortion saved per extra byte spent on this step
    item_index: int
    step: int


def _hull_upgrades(
    items: tuple[KnapsackItem, ...],
    hulls: list[tuple[OperatingPoint, ...]],
) -> list[_HullUpgrade]:
    """Every single-step hull upgrade, steepest distortion-per-byte first.

    Each hull is ordered cheapest-first, so stepping vertex ``k`` -> ``k+1`` spends ``delta_bytes`` more
    to save ``weight * delta_distortion`` distortion. Sorting by that slope descending is exactly the
    order the Lagrangian sweep applies upgrades: the best saving per byte is always bought next.
    """
    upgrades: list[_HullUpgrade] = []
    for item_pos, (item, hull) in enumerate(zip(items, hulls)):
        for step in range(len(hull) - 1):
            delta_bytes = hull[step + 1].stored_bytes - hull[step].stored_bytes
            delta_distortion = hull[step].distortion - hull[step + 1].distortion
            upgrades.append(_HullUpgrade(item.weight * delta_distortion / delta_bytes, item_pos, step))
    upgrades.sort(key=lambda upgrade: upgrade.slope, reverse=True)
    return upgrades


def _lagrangian_curve(
    items: tuple[KnapsackItem, ...],
) -> tuple[list[RDCurvePoint], list[tuple[OperatingPoint, ...]]]:
    """Trace the budget->quality curve: start all-cheapest, then apply hull upgrades steepest-first."""
    hulls = _hulls(items)
    index = [0] * len(items)
    total_bytes = sum(hull[0].stored_bytes for hull in hulls)
    objective = sum(item.weight * hull[0].distortion for item, hull in zip(items, hulls))

    curve = [
        RDCurvePoint(
            lam=np.inf,
            total_bytes=total_bytes,
            objective=objective,
            indices=tuple(index),
        )
    ]
    for upgrade in _hull_upgrades(items, hulls):
        hull = hulls[upgrade.item_index]
        total_bytes += hull[upgrade.step + 1].stored_bytes - hull[upgrade.step].stored_bytes
        objective -= items[upgrade.item_index].weight * (
            hull[upgrade.step].distortion - hull[upgrade.step + 1].distortion
        )
        index[upgrade.item_index] += 1
        curve.append(
            RDCurvePoint(
                lam=upgrade.slope,
                total_bytes=total_bytes,
                objective=objective,
                indices=tuple(index),
            )
        )
    return curve, hulls


def rd_curve(items: tuple[KnapsackItem, ...]) -> list[RDCurvePoint]:
    """The full achievable budget->quality curve (ascending bytes, descending objective)."""
    if not items:
        return [
            RDCurvePoint(
                lam=np.inf,
                total_bytes=0,
                objective=0.0,
                indices=(),
            )
        ]
    return _lagrangian_curve(items)[0]


def solve_lagrangian(
    items: tuple[KnapsackItem, ...],
    budget_bytes: int,
) -> Allocation:
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
    return Allocation(
        selections=selections,
        total_bytes=best.total_bytes,
        objective=best.objective,
    )
