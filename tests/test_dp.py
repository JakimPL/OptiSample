from __future__ import annotations

import pytest

from optisample.optimize.dp import BudgetInfeasibleError, require_feasible


def test_require_feasible_allows_a_budget_that_fits() -> None:
    require_feasible(100, 100)  # exactly fits -- no raise
    require_feasible(80, 100)  # room to spare -- no raise


def test_require_feasible_raises_when_the_cheapest_allocation_overflows() -> None:
    with pytest.raises(BudgetInfeasibleError) as excinfo:
        require_feasible(101, 100)
    assert excinfo.value.min_bytes == 101
    assert excinfo.value.budget_bytes == 100
