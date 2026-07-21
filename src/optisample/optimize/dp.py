"""Budget-feasibility primitives shared by the byte-indexed allocation DPs.

Both allocation solvers -- the multiple-choice knapsack (:mod:`optisample.optimize.knapsack`) and the
partition + allocation DP (:mod:`optisample.optimize.grouping.solve`) -- minimize distortion under a hard
byte budget, and both must reject a budget too small to hold even the cheapest choice per item. That one
feasibility rule lives here so the two solvers raise the *same* error from the *same* guard.

The forward DP kernels themselves stay separate: knapsack fills a 1-D table with one decision per item,
while grouping fills a 2-D table with an extra partition axis (variable-length zones). Their backpointer
walks follow those shapes -- fixed per-item hops for knapsack, jumps to the previous zone boundary for
grouping -- so each owns its own reconstruction, and the two are documented as siblings that share only
this feasibility guard.
"""

from __future__ import annotations


class BudgetInfeasibleError(Exception):
    """Raised when even the cheapest config per item overflows the budget."""

    def __init__(self, min_bytes: int, budget_bytes: int) -> None:
        self.min_bytes = min_bytes
        self.budget_bytes = budget_bytes
        super().__init__(f"budget {budget_bytes} B too small; cheapest allocation needs {min_bytes} B")


def require_feasible(cheapest_bytes: int, budget_bytes: int) -> None:
    """Raise :class:`BudgetInfeasibleError` when the cheapest allocation cannot fit ``budget_bytes``."""
    if cheapest_bytes > budget_bytes:
        raise BudgetInfeasibleError(cheapest_bytes, budget_bytes)
