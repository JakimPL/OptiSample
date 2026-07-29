from __future__ import annotations

from dataclasses import dataclass

import pytest

from optisample.metrics.size import kib_to_bytes
from optisample.optimize.plans import (
    SINGLE_LAYER,
    BudgetBreakdown,
    BudgetedPlanMixin,
    instrument_overhead,
    populated_instrument_bytes,
    split_budget,
)
from trackmod.module.storage import Storage

_LAYERS = 3  # a piano-like split into soft, medium and loud, which writes three instruments


def test_the_overhead_is_the_file_record_plus_one_populated_instrument(storage: Storage) -> None:
    assert populated_instrument_bytes(storage) == storage.instrument_bytes(samples=1)
    assert instrument_overhead(storage, SINGLE_LAYER) == storage.file + populated_instrument_bytes(storage)


def test_every_velocity_layer_pays_for_an_instrument_record(storage: Storage) -> None:
    """The honest price of vocabulary: a layer is an instrument, and an instrument costs a record."""
    overhead = instrument_overhead(storage, _LAYERS)
    assert overhead == storage.file + _LAYERS * populated_instrument_bytes(storage)
    assert overhead - instrument_overhead(storage, SINGLE_LAYER) == 2 * populated_instrument_bytes(storage)


def test_split_budget_reserves_the_record_overhead(storage: Storage) -> None:
    budget = split_budget(64.0, storage, SINGLE_LAYER)
    assert budget.module_bytes == kib_to_bytes(64.0)
    assert budget.sample_bytes == budget.module_bytes - instrument_overhead(storage, SINGLE_LAYER)
    assert budget.storage is storage  # a plan prices what it stored against the table it budgeted from


def test_more_layers_leave_fewer_bytes_for_samples(storage: Storage) -> None:
    layered = split_budget(64.0, storage, _LAYERS)
    assert layered.instruments == _LAYERS
    assert layered.module_bytes == split_budget(64.0, storage, SINGLE_LAYER).module_bytes
    assert layered.sample_bytes < split_budget(64.0, storage, SINGLE_LAYER).sample_bytes


@dataclass(frozen=True)
class _Plan(BudgetedPlanMixin):
    """Minimal concrete plan: supply ``budget`` + ``used_bytes`` and let the mixin derive the rest."""

    budget: BudgetBreakdown
    stored_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.stored_bytes


@pytest.mark.parametrize("layers", [SINGLE_LAYER, _LAYERS])
def test_mixin_derives_budget_ceilings_and_module_size(storage: Storage, layers: int) -> None:
    budget = split_budget(32.0, storage, layers)
    plan = _Plan(budget=budget, stored_bytes=1000)
    assert plan.module_budget_bytes == budget.module_bytes  # from the budget breakdown
    assert plan.sample_budget_bytes == budget.sample_bytes
    assert plan.used_bytes == 1000  # supplied by the concrete plan
    assert plan.module_bytes == 1000 + instrument_overhead(storage, layers)  # used + record overhead
