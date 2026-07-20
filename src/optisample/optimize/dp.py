"""Budget-feasibility primitives shared by the byte-indexed allocation DPs.

Both allocation solvers -- the multiple-choice knapsack (:mod:`optisample.optimize.knapsack`) and the
partition + allocation DP (:mod:`optisample.optimize.grouping.solve`) -- minimize distortion under a hard
byte budget, and both must reject a budget too small to hold even the cheapest choice per item. That one
feasibility rule lives here so the two solvers raise the *same* error from the *same* guard.

The forward DP kernels themselves stay separate: knapsack fills a 1-D table with one decision per item,
while grouping fills a 2-D table with an extra partition axis (variable-length zones). Their backpointer
walks differ for the same reason -- fixed per-item hops versus jumps to the previous zone boundary -- so a
shared "reconstruct" abstraction would need callback plumbing that obscures more than it removes. They are
documented as siblings, not merged.
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
