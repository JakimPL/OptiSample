"""Plan value objects for the ungrouped strategy: one stored sample per used key."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from optisample.music import note_name
from optisample.optimize.knapsack import Allocation, RDCurvePoint
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans.budget import BudgetBreakdown, BudgetedPlanMixin
from optisample.optimize.plans.strategy import Method, SampleUnit
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.velocity_map import VelocityVolumeMap


@dataclass(frozen=True)
class PitchPlan:
    """The outcome for one pitch: its representative recording and the config the solver chose."""

    pitch: int
    weight: float
    representative_key: SampleKey
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
    method: Method
    strategy: Literal["ungrouped"] = "ungrouped"

    @property
    def used_bytes(self) -> int:
        return self.allocation.total_bytes

    @property
    def objective(self) -> float:
        return self.allocation.objective

    @property
    def total_weight(self) -> float:
        return sum(plan.weight for plan in self.pitches)

    def sample_units(self) -> tuple[SampleUnit, ...]:
        """One stored sample per kept pitch, each serving only its own key (an identity note map)."""
        return tuple(
            SampleUnit(
                label=f"p{pitch.pitch:03d}_{note_name(pitch.pitch)}",
                representative_key=pitch.representative_key,
                keys=(pitch.pitch,),
                params=pitch.chosen.params,
                frames=pitch.chosen.frames,
                stored_bytes=pitch.chosen.stored_bytes,
                distortion=pitch.chosen.distortion,
                hull_size=len(pitch.hull),
                weight=pitch.weight,
            )
            for pitch in self.pitches
        )
