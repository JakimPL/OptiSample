class AllocationInfeasibleError(Exception):
    """Raised when the allocation a run asks for is out of reach, so no plan can be written for it.

    The allocation stages answer with a plan or with the reason there is none, and a caller writing
    artifacts records that reason beside the strategy it belongs to.
    """


class BudgetInfeasibleError(AllocationInfeasibleError):
    """Raised when even the cheapest config per item overflows the budget."""

    def __init__(self, min_bytes: int, budget_bytes: int) -> None:
        self.min_bytes = min_bytes
        self.budget_bytes = budget_bytes
        super().__init__(f"budget {budget_bytes} B too small; cheapest allocation needs {min_bytes} B")


def require_feasible(cheapest_bytes: int, budget_bytes: int) -> None:
    """Raise :class:`BudgetInfeasibleError` when the cheapest allocation cannot fit ``budget_bytes``."""
    if cheapest_bytes > budget_bytes:
        raise BudgetInfeasibleError(cheapest_bytes, budget_bytes)
