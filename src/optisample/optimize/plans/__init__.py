"""Plan value objects for both allocation strategies, kept apart from the solvers that build them.

The plans are the data shapes the optimizers emit and the exporter/report consume: the shared budget
math (:mod:`.budget`), the ungrouped one-sample-per-key plan (:mod:`.ungrouped`), and the pitch-zone
grouping plan (:mod:`.grouped`). Grouped as a subpackage rather than one bag-of-classes module.
"""

from optisample.optimize.plans.budget import BudgetBreakdown, BudgetedPlanMixin, split_budget
from optisample.optimize.plans.grouped import GroupedInstrumentPlan, GroupingResult, Zone, ZoneOption
from optisample.optimize.plans.ungrouped import InstrumentPlan, PitchPlan

__all__ = [
    "BudgetBreakdown",
    "BudgetedPlanMixin",
    "split_budget",
    "InstrumentPlan",
    "PitchPlan",
    "GroupedInstrumentPlan",
    "GroupingResult",
    "Zone",
    "ZoneOption",
]
