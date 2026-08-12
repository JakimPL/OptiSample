from dataclasses import dataclass
from typing import Literal

from optisample.keys import SampleKey
from optisample.music import pitch_label
from optisample.optimize.knapsack import Allocation, RDCurvePoint
from optisample.optimize.layers.bands import UNSPLIT, VelocityLayers
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans.budget import BudgetBreakdown, BudgetedPlanMixin
from optisample.optimize.plans.strategy import FIRST_LAYER, Method, SampleUnit
from optisample.optimize.reduce.summary import ReductionSummary
from optisample.optimize.velocity_map import VelocityVolumeMap


@dataclass(frozen=True)
class PitchPlan:
    """The outcome for one pitch: its representative recording and the config the solver chose.

    ``weight`` is the playing time the material spends here, which is what the reports state, and
    ``objective_weight`` what that time is worth to the objective once each note's own level has scaled
    it -- the multiplier the solver ranked this pitch's configs under.
    """

    pitch: int
    weight: float
    objective_weight: float
    representative_key: SampleKey
    chosen: OperatingPoint
    hull: tuple[OperatingPoint, ...]


@dataclass(frozen=True)
class InstrumentPlan(BudgetedPlanMixin):
    """The full result for one instrument: the map, per-pitch choices, and the allocation.

    ``energy_exponent`` is how steeply each note's own energy scaled its distortion, which states what
    the ``objective`` means and so which other plans it may be compared with.
    """

    instrument_id: str
    budget: BudgetBreakdown
    velocity_map: VelocityVolumeMap
    pitches: tuple[PitchPlan, ...]
    allocation: Allocation
    curve: tuple[RDCurvePoint, ...]
    method: Method
    energy_exponent: float
    reduction: ReductionSummary
    strategy: Literal["ungrouped"] = "ungrouped"

    @property
    def layers(self) -> VelocityLayers:
        """The one layer every key is written as, since a kept recording answers each of its dynamics."""
        return UNSPLIT

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
        """One stored sample per kept pitch, each serving only its own key (an identity note map).

        Every key answers every dynamic from the one recording kept for it, so the whole plan is written
        as the single layer :data:`~optisample.optimize.plans.strategy.FIRST_LAYER` names.
        """
        return tuple(
            SampleUnit(
                label=pitch_label(pitch.pitch),
                representative_key=pitch.representative_key,
                layer=FIRST_LAYER,
                keys=(pitch.pitch,),
                params=pitch.chosen.params,
                frames=pitch.chosen.frames,
                stored_bytes=pitch.chosen.stored_bytes,
                distortion=pitch.chosen.distortion,
                objective_share=pitch.objective_weight * pitch.chosen.distortion,
                hull_size=len(pitch.hull),
                weight=pitch.weight,
            )
            for pitch in self.pitches
        )
