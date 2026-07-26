from collections.abc import Mapping, Sequence

from optisample.optimize.knapsack import (
    Allocation,
    KnapsackItem,
    solve_exact,
    solve_lagrangian,
)
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans import Method, PitchPlan
from optisample.optimize.tasks import PitchTask


def _pitch_plans(
    tasks: Sequence[PitchTask],
    allocation: Allocation,
    hulls: Mapping[int, tuple[OperatingPoint, ...]],
) -> tuple[PitchPlan, ...]:
    """Attach the solver's chosen config to each pitch task."""
    chosen = {selection.key: selection.point for selection in allocation.selections}
    return tuple(
        PitchPlan(
            pitch=task.pitch,
            weight=task.weight,
            representative_key=task.representative_key,
            chosen=chosen[str(task.pitch)],
            hull=hulls[task.pitch],
        )
        for task in tasks
    )


def solve_allocation(
    tasks: Sequence[PitchTask],
    items: tuple[KnapsackItem, ...],
    hulls: Mapping[int, tuple[OperatingPoint, ...]],
    budget_bytes: int,
    method: Method,
) -> tuple[Allocation, tuple[PitchPlan, ...]]:
    """Run the MCKP solver under the byte budget and attach each pitch's chosen config.

    ``method`` selects the exact DP or the faster Lagrangian sweep; both minimize weighted distortion
    subject to the sample-byte budget.
    """
    solver = solve_exact if method == "exact" else solve_lagrangian
    allocation = solver(items, budget_bytes)
    return allocation, _pitch_plans(tasks, allocation, hulls)
