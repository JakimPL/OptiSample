from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.optimize.dp import AllocationInfeasibleError, BudgetInfeasibleError
from optisample.optimize.grouping.cost_model import ZoneSegment, _ZoneOptions
from optisample.optimize.grouping.solve import solve_grouping
from optisample.optimize.plans.grouped import NO_RESERVE, GroupingResult, SampleReserve
from optisample.progress import ProgressStep

_REFINEMENTS: Final = 8  # halvings spent narrowing the charge once one that meets the cap is known
_ONE_ZONE: Final = 1  # zones a partition holds beyond the cap for the charge to price it out
_FREE_AND_CEILING: Final = 2  # walks a capped solve spends before it starts narrowing

PROBES_PER_SOLVE: Final = _REFINEMENTS + _FREE_AND_CEILING  # most walks one capped solve spends


class SampleCapInfeasibleError(AllocationInfeasibleError):
    """Raised when the charge that prices every partition past the cap out leaves the budget carrying none."""

    def __init__(self, cap: int, reserve: int, budget_bytes: int) -> None:
        self.cap = cap
        self.reserve = reserve
        self.budget_bytes = budget_bytes
        super().__init__(
            f"a plan of at most {cap} stored samples is out of reach: charging {reserve} B per sample, "
            f"which is what prices a wider partition out, leaves the {budget_bytes} B sample budget "
            f"carrying no partition at all"
        )


@dataclass(frozen=True)
class CappedGrouping:
    """A solve that holds at most the samples the cap allows, and what holding it there took."""

    result: GroupingResult
    reserve: SampleReserve


@dataclass(frozen=True)
class _Search:
    """One candidate split's scored zones, walked again at whichever charge per stored sample is asked.

    Each probe is a full partition-and-allocation pass over these same option tables, which is what the
    search spends: the charge is looked for by walking, since what a charge yields is known only once the
    walk has run. ``probe`` counts each of them off, so a caller holding several searches states one bar
    across the walks they spend between them (:data:`PROBES_PER_SOLVE`) -- the stage runs long enough on a
    real keyboard that a run says where in it it stands.
    """

    segments: Sequence[ZoneSegment]
    options: Sequence[_ZoneOptions]
    budget_bytes: int
    cap: int
    probe: ProgressStep

    @property
    def ceiling(self) -> int:
        """The charge at which every partition the budget still carries holds at most ``cap`` zones.

        A partition of more than ``cap`` zones pays the charge more than ``cap`` times, so a charge above
        the budget split ``cap + 1`` ways prices all of them past the budget and leaves the walk choosing
        among the partitions the cap allows.
        """
        return self.budget_bytes // (self.cap + _ONE_ZONE) + 1

    def within(self, result: GroupingResult) -> bool:
        """Whether ``result`` keeps at most the stored samples the cap allows."""
        return len(result.zones) <= self.cap

    def walk(self, reserve: int) -> GroupingResult:
        """The best partition the budget carries when each stored sample is charged ``reserve``.

        Raises:
            BudgetInfeasibleError: when the cheapest partition's charged bytes overrun the budget.
        """
        self.probe()
        return solve_grouping(self.segments, self.options, self.budget_bytes, reserve=reserve)

    def narrow(self, kept: GroupingResult) -> tuple[int, GroupingResult]:
        """The smallest charge the search reaches that still holds the walk within the cap, and its plan.

        Bisects between the charge that left the walk over the cap and :attr:`ceiling`, which ``kept`` is
        the plan of and which is known to meet it, keeping the cheapest charge that met it. A smaller
        charge leaves more of the budget for the samples themselves, so the narrowing buys objective back.
        Every probe walks the whole axis, so the search spends a fixed number of them and answers with the
        best charge it reached, which is the approximation this relaxation of a count constraint makes.
        """
        low, high, best = NO_RESERVE, self.ceiling, kept
        for _ in range(_REFINEMENTS):
            if low + 1 >= high:
                break

            middle = (low + high) // 2
            attempt = self.walk(middle)
            if self.within(attempt):
                high, best = middle, attempt
            else:
                low = middle

        return high, best


def solve_within_cap(
    segments: Sequence[ZoneSegment],
    options: Sequence[_ZoneOptions],
    budget_bytes: int,
    cap: int,
    *,
    probe: ProgressStep,
) -> CappedGrouping:
    """Solve the partition and allocation, held to at most ``cap`` stored samples across every layer.

    The budget alone decides the plan whenever it already stores few enough samples, which is the case
    every format-sized cap meets and the one that costs a single walk. A cap the free solve overruns is
    met by charging each stored sample beyond its own bytes until the walk keeps at most ``cap`` of them:
    :attr:`_Search.ceiling` is the charge that prices every wider partition out of the budget, and
    :meth:`_Search.narrow` then works back toward the smallest charge that still holds.

    Charging per stored sample is a Lagrangian relaxation of the count constraint, so it answers with the
    best plan a charge induces, which the options' byte-vs-distortion frontier may leave a hair off the
    best plan of exactly ``cap`` samples. The exact alternative -- a zone-count axis in the walk -- costs
    a table ``cap`` times the size, which is what makes the relaxation the one solved here.

    Raises:
        BudgetInfeasibleError: when the budget carries no partition even before a charge is added.
        SampleCapInfeasibleError: when the charge meeting the cap leaves the budget carrying no partition.
    """
    search = _Search(segments=segments, options=options, budget_bytes=budget_bytes, cap=cap, probe=probe)
    free = search.walk(NO_RESERVE)
    if search.within(free):
        return CappedGrouping(
            free,
            SampleReserve(cap=cap, bytes_per_sample=NO_RESERVE, objective_uncapped=free.objective),
        )

    try:
        charged = search.walk(search.ceiling)
    except BudgetInfeasibleError as error:
        raise SampleCapInfeasibleError(cap, search.ceiling, budget_bytes) from error

    reserve, result = search.narrow(charged)
    return CappedGrouping(
        result,
        SampleReserve(cap=cap, bytes_per_sample=reserve, objective_uncapped=free.objective),
    )
