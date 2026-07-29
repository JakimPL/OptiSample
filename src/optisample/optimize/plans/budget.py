from dataclasses import dataclass
from typing import Final

from optisample.metrics.size import kib_to_bytes
from trackmod.module.storage import Storage

_POPULATED_INSTRUMENT: Final = 1  # slot count that puts an instrument in its format's full header form


def populated_instrument_bytes(storage: Storage) -> int:
    """What the record of one instrument owning at least one stored sample costs."""
    return storage.instrument_bytes(samples=_POPULATED_INSTRUMENT)


def instrument_overhead(storage: Storage, instruments: int) -> int:
    """Every byte a module spends before its first stored sample: the file record plus its instruments.

    Each velocity layer is written as at least one instrument, and a format numbering few samples inside
    one writes a wide layer as several (:func:`~optisample.optimize.layers.slots.pack_slots`), so what the
    plan pays before a single frame of audio is kept follows from how many instruments it comes out as --
    the honest price the layer decision is taken against. The audition patterns and the order list are
    left out: they are material a module happens to play, while the budget measures what carrying the
    instrument itself costs. What the written file occupies exactly is reported beside the budget, read
    from the module's own size.
    """
    return storage.file + instruments * populated_instrument_bytes(storage)


@dataclass(frozen=True)
class BudgetBreakdown:
    """The byte budget split into the whole module and the part left for stored samples.

    ``storage`` is the format cost table the split was taken against, kept so a plan prices what it
    stored against the same format its budget was drawn from, and ``instruments`` how many instrument
    records the split reserved room for, so a plan storing several velocity layers -- or a layer wide
    enough to be written as several instruments -- starts from the smaller sample budget its own
    vocabulary costs.
    """

    storage: Storage
    instruments: int
    module_bytes: int
    sample_bytes: int  # module_bytes minus the file and instrument records


def split_budget(budget_kb: float, storage: Storage, instruments: int) -> BudgetBreakdown:
    """Split an instrument's KiB budget into the whole module and the samples part left after records."""
    module_bytes = kib_to_bytes(budget_kb)
    return BudgetBreakdown(
        storage=storage,
        instruments=instruments,
        module_bytes=module_bytes,
        sample_bytes=module_bytes - instrument_overhead(storage, instruments),
    )


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
        return self.used_bytes + instrument_overhead(self.budget.storage, self.budget.instruments)
