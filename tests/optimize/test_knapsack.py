from __future__ import annotations

import itertools

import numpy as np
import pytest

from optisample.dsp.surrogate import EncodingParams
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.knapsack import (
    Allocation,
    KnapsackItem,
    rd_curve,
    solve_exact,
    solve_lagrangian,
)
from optisample.optimize.operating_points import OperatingPoint


def op(stored_bytes: int, distortion: float) -> OperatingPoint:
    return OperatingPoint(EncodingParams(target_rate=1, depth_bits=16), stored_bytes, distortion, stored_bytes)


def item(key: str, weight: float, points: list[tuple[int, float]]) -> KnapsackItem:
    return KnapsackItem(key=key, weight=weight, points=tuple(op(b, d) for b, d in points))


def brute_force(items: tuple[KnapsackItem, ...], budget: int) -> tuple[float, int]:
    """Reference optimum by exhaustive search: (min objective, its bytes)."""
    best = (np.inf, 0)
    for combo in itertools.product(*[range(len(it.points)) for it in items]):
        total = sum(items[i].points[c].stored_bytes for i, c in enumerate(combo))
        if total > budget:
            continue
        objective = sum(items[i].weight * items[i].points[c].distortion for i, c in enumerate(combo))
        if objective < best[0]:
            best = (objective, total)
    return best


def random_items(seed: int, n: int = 5, configs: int = 4) -> tuple[KnapsackItem, ...]:
    rng = np.random.default_rng(seed)
    items = []
    for j in range(n):
        points = list(zip(rng.integers(80, 500, configs).tolist(), (rng.random(configs) + 0.05).tolist()))
        items.append(item(str(j), float(rng.random() * 5 + 0.5), points))
    return tuple(items)


def cheapest_bytes(items: tuple[KnapsackItem, ...]) -> int:
    return sum(min(p.stored_bytes for p in it.points) for it in items)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_solve_exact_matches_brute_force(seed: int) -> None:
    items = random_items(seed)
    low = cheapest_bytes(items)
    high = sum(max(p.stored_bytes for p in it.points) for it in items)
    for budget in (low, (low + high) // 2, high):
        allocation = solve_exact(items, budget)
        expected_objective, _ = brute_force(items, budget)
        assert allocation.objective == pytest.approx(expected_objective, abs=1e-9)
        assert allocation.total_bytes <= budget
        # the reported objective is exactly the sum over the reconstructed selections
        recomputed = sum(sel.weight * sel.point.distortion for sel in allocation.selections)
        assert recomputed == pytest.approx(allocation.objective, abs=1e-9)


def test_solve_exact_reconstructs_one_selection_per_item_in_order() -> None:
    items = random_items(0)
    allocation = solve_exact(items, cheapest_bytes(items) + 300)
    assert tuple(sel.key for sel in allocation.selections) == tuple(it.key for it in items)
    assert allocation.total_bytes == sum(sel.point.stored_bytes for sel in allocation.selections)


def test_tighter_budget_never_lowers_objective() -> None:
    items = random_items(1)
    low = cheapest_bytes(items)
    tight = solve_exact(items, low + 100)
    loose = solve_exact(items, low + 600)
    assert tight.objective >= loose.objective - 1e-12
    assert tight.total_bytes <= loose.total_bytes + 1e-9


def test_solve_exact_infeasible_raises_with_minimum() -> None:
    items = random_items(2)
    low = cheapest_bytes(items)
    with pytest.raises(BudgetInfeasibleError) as excinfo:
        solve_exact(items, low - 1)
    assert excinfo.value.min_bytes == low
    assert excinfo.value.budget_bytes == low - 1


def test_empty_items_is_trivial() -> None:
    assert solve_exact((), 1000) == Allocation(selections=(), total_bytes=0, objective=0.0)
    assert solve_lagrangian((), 1000) == Allocation(selections=(), total_bytes=0, objective=0.0)
    assert [p.total_bytes for p in rd_curve(())] == [0]


def test_weight_steers_the_single_affordable_upgrade() -> None:
    # Two identical hulls; the budget affords exactly one upgrade → the heavier item must win it.
    configs = [(100, 1.0), (200, 0.0)]
    heavy = item("heavy", 3.0, configs)
    light = item("light", 1.0, configs)
    allocation = solve_exact((heavy, light), 300)  # 200 + 100, cannot afford 200 + 200
    chosen = {sel.key: sel.point.stored_bytes for sel in allocation.selections}
    assert chosen == {"heavy": 200, "light": 100}
    assert allocation.objective == pytest.approx(1.0)  # 3*0 + 1*1


def test_rd_curve_is_monotone_and_convex() -> None:
    items = random_items(3)
    curve = rd_curve(items)
    assert len(curve) >= 2
    assert all(a.total_bytes < b.total_bytes for a, b in zip(curve, curve[1:]))
    assert all(a.objective > b.objective - 1e-12 for a, b in zip(curve, curve[1:]))
    assert curve[0].lam == np.inf
    assert all(a.lam >= b.lam for a, b in zip(curve[1:], curve[2:]))  # slopes shrink as bytes grow


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lagrangian_is_feasible_and_no_better_than_exact(seed: int) -> None:
    items = random_items(seed)
    low = cheapest_bytes(items)
    for budget in (low + 50, low + 250, low + 700):
        lagrangian = solve_lagrangian(items, budget)
        exact = solve_exact(items, budget)
        assert lagrangian.total_bytes <= budget
        assert lagrangian.objective >= exact.objective - 1e-9  # exact is optimal; Lagrangian can't beat it


def test_lagrangian_is_exact_at_hull_breakpoints() -> None:
    items = random_items(1)
    for point in rd_curve(items):
        allocation = solve_lagrangian(items, point.total_bytes)
        assert allocation.total_bytes == point.total_bytes
        assert allocation.objective == pytest.approx(point.objective, abs=1e-9)
        # at a breakpoint the exact DP agrees too
        assert solve_exact(items, point.total_bytes).objective == pytest.approx(point.objective, abs=1e-9)


def test_lagrangian_infeasible_raises() -> None:
    items = random_items(0)
    low = cheapest_bytes(items)
    with pytest.raises(BudgetInfeasibleError) as excinfo:
        solve_lagrangian(items, low - 1)
    assert excinfo.value.min_bytes == low


def test_single_config_items_are_fixed() -> None:
    items = (item("a", 2.0, [(150, 0.4)]), item("b", 1.0, [(90, 0.9)]))
    allocation = solve_exact(items, 1000)
    assert allocation.total_bytes == 240
    assert allocation.objective == pytest.approx(2.0 * 0.4 + 1.0 * 0.9)
    assert len(rd_curve(items)) == 1  # nothing to upgrade
