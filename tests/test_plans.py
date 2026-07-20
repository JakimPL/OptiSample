from __future__ import annotations

from dataclasses import dataclass

from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, kib_to_bytes
from optisample.optimize.plans import BudgetBreakdown, BudgetedPlanMixin, split_budget


def test_split_budget_reserves_the_header_overhead() -> None:
    budget = split_budget(64.0)
    assert budget.module_bytes == kib_to_bytes(64.0)
    assert budget.sample_bytes == budget.module_bytes - FILE_HEADER_BYTES - INSTRUMENT_HEADER_BYTES


@dataclass(frozen=True)
class _Plan(BudgetedPlanMixin):
    """Minimal concrete plan: supply ``budget`` + ``used_bytes`` and let the mixin derive the rest."""

    budget: BudgetBreakdown
    stored_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.stored_bytes


def test_mixin_derives_budget_ceilings_and_module_size() -> None:
    budget = split_budget(32.0)
    plan = _Plan(budget=budget, stored_bytes=1000)
    assert plan.module_budget_bytes == budget.module_bytes  # from the budget breakdown
    assert plan.sample_budget_bytes == budget.sample_bytes
    assert plan.used_bytes == 1000  # supplied by the concrete plan
    assert plan.module_bytes == 1000 + FILE_HEADER_BYTES + INSTRUMENT_HEADER_BYTES  # used + header overhead
