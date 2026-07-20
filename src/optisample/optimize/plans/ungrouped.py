"""Plan value objects for the ungrouped strategy: one stored sample per used key."""

from __future__ import annotations

from dataclasses import dataclass

from optisample.optimize.knapsack import Allocation, RDCurvePoint
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans.budget import BudgetBreakdown, BudgetedPlanMixin
from optisample.optimize.velocity_map import VelocityVolumeMap


@dataclass(frozen=True)
class PitchPlan:
    """The outcome for one pitch: its representative recording and the config the solver chose."""

    pitch: int
    weight: float
    representative_velocity: int
    chosen: OperatingPoint
    hull: tuple[OperatingPoint, ...]


@dataclass(frozen=True)
class InstrumentPlan(BudgetedPlanMixin):
    """The full result for one instrument: the map, per-pitch choices, and the allocation."""

    instrument_id: str
    budget: BudgetBreakdown
    velocity_map: VelocityVolumeMap
    pitches: tuple[PitchPlan, ...]
    allocation: Allocation
    curve: tuple[RDCurvePoint, ...]
    method: str

    @property
    def used_bytes(self) -> int:
        return self.allocation.total_bytes

    @property
    def objective(self) -> float:
        return self.allocation.objective

    @property
    def total_weight(self) -> float:
        return sum(plan.weight for plan in self.pitches)
