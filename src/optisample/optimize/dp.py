from __future__ import annotations

from dataclasses import dataclass

WHOLE_BYTES = 1  # the granularity that states every byte total a budget holds


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


@dataclass(frozen=True)
class ByteGrid:
    """The steps an allocation walks a byte budget in.

    A budget DP holds a row per total it can reach, so what one walk costs in time and in memory follows
    the budget it is handed: the same keyboard allocated against a pack four times the size takes four
    times as long and four times the table. Walking in steps of ``granularity`` bytes holds that row to
    ``steps`` however large the budget grows, which is what makes the cost of an allocation a matter of
    the resolution asked for rather than of the pack being built.

    What the resolution buys is paid for in byte totals between two steps, which a plan on this grid
    states as the step above: a cost is charged at the step it reaches and the budget kept to the steps it
    fills, so every total the walk reads is one the true bytes fit inside and the plan answered spends at
    most the budget it was given. Each zone gives up under ``granularity`` bytes that way, so what a walk
    leaves unspendable is the resolution times the zones it stores.
    """

    granularity: int
    steps: int

    @property
    def usable_bytes(self) -> int:
        """The bytes an allocation on this grid may spend: the whole steps the budget holds."""
        return self.spent(self.steps)

    def cost(self, stored_bytes: int) -> int:
        """The steps an allocation of ``stored_bytes`` is charged: its bytes taken up to the step above."""
        return (stored_bytes + self.granularity - 1) // self.granularity

    def spent(self, steps: int) -> int:
        """The bytes ``steps`` of the walk stand for."""
        return steps * self.granularity


def byte_grid(budget_bytes: int, resolution: int | None) -> ByteGrid:
    """The grid ``budget_bytes`` is walked on, resolved into at most ``resolution`` steps.

    The granularity follows the budget, so one walk fills the table the resolution names whatever the
    pack costs, and a budget already inside the resolution is walked to the byte. ``None`` asks for that
    directly, which is the exact allocation and what every budget got before a resolution was stated.
    """
    if resolution is None:
        return ByteGrid(granularity=WHOLE_BYTES, steps=budget_bytes)

    granularity = max(WHOLE_BYTES, (budget_bytes + resolution - 1) // resolution)
    return ByteGrid(granularity=granularity, steps=budget_bytes // granularity)
