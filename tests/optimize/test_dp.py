from __future__ import annotations

from typing import Final

import pytest

from optisample.optimize.dp import WHOLE_BYTES, BudgetInfeasibleError, byte_grid, require_feasible

_BUDGET: Final = 1_000
_EXACT: Final = None  # the resolution that walks every byte total a budget holds
_ZONES: Final = 24  # stored samples a plan of the shipped cap holds, each giving up under one step


def test_require_feasible_allows_a_budget_that_fits() -> None:
    require_feasible(100, 100)  # exactly fits -- no raise
    require_feasible(80, 100)  # room to spare -- no raise


def test_require_feasible_raises_when_the_cheapest_allocation_overflows() -> None:
    with pytest.raises(BudgetInfeasibleError) as excinfo:
        require_feasible(101, 100)
    assert excinfo.value.min_bytes == 101
    assert excinfo.value.budget_bytes == 100


# --- the steps a budget is walked in ---


def test_a_budget_asked_for_to_the_byte_states_every_total_it_holds() -> None:
    grid = byte_grid(_BUDGET, _EXACT)
    assert grid.granularity == WHOLE_BYTES
    assert grid.steps == _BUDGET
    assert grid.usable_bytes == _BUDGET


@pytest.mark.parametrize("resolution", [1, 7, 64, 250, 999, _BUDGET, 4 * _BUDGET])
def test_a_grid_resolves_its_budget_into_at_most_the_steps_asked_for(resolution: int) -> None:
    """The granularity follows the budget, so one walk fills the table the resolution names."""
    grid = byte_grid(_BUDGET, resolution)
    assert grid.steps <= resolution
    assert grid.granularity >= WHOLE_BYTES


def test_a_resolution_the_budget_already_sits_inside_states_every_total() -> None:
    """A budget with fewer bytes than the steps asked for is walked to the byte, which is exact."""
    assert byte_grid(_BUDGET, 4 * _BUDGET) == byte_grid(_BUDGET, _EXACT)


@pytest.mark.parametrize("resolution", [7, 64, 250])
def test_what_a_walk_may_spend_stays_inside_the_budget_it_was_handed(resolution: int) -> None:
    """The budget is kept to the whole steps it holds, so a plan filling the grid fits the pack."""
    grid = byte_grid(_BUDGET, resolution)
    assert grid.usable_bytes <= _BUDGET
    assert _BUDGET - grid.usable_bytes < grid.granularity


@pytest.mark.parametrize("stored_bytes", [0, 1, 63, 64, 65, 999])
def test_a_cost_is_charged_at_the_step_that_holds_it(stored_bytes: int) -> None:
    """Taking each cost up to the step above is what keeps a plan's true bytes inside what it is charged."""
    grid = byte_grid(_BUDGET, 64)
    charged = grid.cost(stored_bytes)
    assert grid.spent(charged) >= stored_bytes
    assert grid.spent(charged) - stored_bytes < grid.granularity


def test_what_a_plan_gives_up_is_the_zones_it_stores_times_one_step() -> None:
    """Every zone rounds up by under one step, so the bytes a grid leaves unspendable are bounded and small."""
    grid = byte_grid(512 * 1024, 16_384)
    assert _ZONES * grid.granularity < 512 * 1024 // 100  # under a hundredth of the pack
