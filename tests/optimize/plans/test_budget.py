from __future__ import annotations

from dataclasses import dataclass

from optisample.metrics.size import kib_to_bytes
from optisample.optimize.plans import (
    BudgetBreakdown,
    BudgetedPlanMixin,
    instrument_overhead,
    per_key_bytes,
    populated_instrument_bytes,
    split_budget,
)
from trackmod.module.storage import Storage


def test_the_overhead_is_the_file_record_plus_one_populated_instrument(storage: Storage) -> None:
    assert populated_instrument_bytes(storage) == storage.instrument_bytes(samples=1)
    assert instrument_overhead(storage) == storage.file + populated_instrument_bytes(storage)


def test_split_budget_reserves_the_record_overhead(storage: Storage) -> None:
    budget = split_budget(64.0, storage)
    assert budget.module_bytes == kib_to_bytes(64.0)
    assert budget.sample_bytes == budget.module_bytes - instrument_overhead(storage)
    assert budget.storage is storage  # a plan prices what it stored against the table it budgeted from


def test_an_even_split_gives_each_key_its_share_of_the_sample_budget(storage: Storage) -> None:
    budget = split_budget(64.0, storage)
    assert per_key_bytes(budget, 8) == budget.sample_bytes // 8


def test_an_instrument_storing_nothing_reports_the_whole_sample_budget(storage: Storage) -> None:
    budget = split_budget(64.0, storage)
    assert per_key_bytes(budget, 0) == budget.sample_bytes


@dataclass(frozen=True)
class _Plan(BudgetedPlanMixin):
    """Minimal concrete plan: supply ``budget`` + ``used_bytes`` and let the mixin derive the rest."""

    budget: BudgetBreakdown
    stored_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.stored_bytes


def test_mixin_derives_budget_ceilings_and_module_size(storage: Storage) -> None:
    budget = split_budget(32.0, storage)
    plan = _Plan(budget=budget, stored_bytes=1000)
    assert plan.module_budget_bytes == budget.module_bytes  # from the budget breakdown
    assert plan.sample_budget_bytes == budget.sample_bytes
    assert plan.used_bytes == 1000  # supplied by the concrete plan
    assert plan.module_bytes == 1000 + instrument_overhead(storage)  # used + record overhead
