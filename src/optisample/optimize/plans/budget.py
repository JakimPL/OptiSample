"""The byte-budget value object and the budget arithmetic every instrument plan shares.

An instrument's KiB budget covers the *whole* IT module; the sample PCM only gets what is left after
the file and instrument headers. That split, and the "how much of it did we spend" math, are identical
for the ungrouped and grouped plans -- so they live here once, as :func:`split_budget` and
:class:`BudgetedPlanMixin`, rather than being copied into each plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, kib_to_bytes


@dataclass(frozen=True)
class BudgetBreakdown:
    """The byte budget split into the whole module and the part left for sample PCM + headers."""

    module_bytes: int
    sample_bytes: int  # module_bytes minus the file + instrument header overhead


def split_budget(budget_kb: float) -> BudgetBreakdown:
    """Split an instrument's KiB budget into the whole module and the samples part left after headers."""
    module_bytes = kib_to_bytes(budget_kb)
    sample_bytes = module_bytes - FILE_HEADER_BYTES - INSTRUMENT_HEADER_BYTES
    return BudgetBreakdown(module_bytes=module_bytes, sample_bytes=sample_bytes)


class BudgetedPlanMixin:
    """Budget-facing properties common to every instrument plan.

    A concrete plan supplies the ``budget`` field and its own ``used_bytes`` (the ungrouped plan reads
    it from the allocation, the grouped plan from its zone totals); this mixin derives the rest -- the
    module/sample budget ceilings and the on-disk module size -- so that math has one home. Being the
    single budget surface, it is also the type the report's budget block accepts.
    """

    budget: BudgetBreakdown

    @property
    def used_bytes(self) -> int:
        """Bytes the stored samples occupy; each concrete plan derives this differently."""
        raise NotImplementedError  # pragma: no cover

    @property
    def module_budget_bytes(self) -> int:
        return self.budget.module_bytes

    @property
    def sample_budget_bytes(self) -> int:
        return self.budget.sample_bytes

    @property
    def module_bytes(self) -> int:
        return self.used_bytes + FILE_HEADER_BYTES + INSTRUMENT_HEADER_BYTES
