"""Plan value objects for the pitch-zone grouping strategy: one stored sample per zone of keys."""

from __future__ import annotations

from dataclasses import dataclass

from optisample.dsp.surrogate import EncodingParams
from optisample.optimize.plans.budget import BudgetBreakdown, BudgetedPlanMixin
from optisample.optimize.velocity_map import VelocityVolumeMap


@dataclass(frozen=True)
class ZoneOption:
    """One way to realize a zone: which member is the stored representative and how it is encoded.

    ``distortion`` is the *usage-weighted, summed* reconstruction distortion over every pitch the
    zone covers (so it is directly comparable to the ungrouped objective), and ``stored_bytes``
    already includes the one 80-byte sample header the zone costs.
    """

    representative: int
    params: EncodingParams
    stored_bytes: int
    distortion: float
    frames: int


@dataclass(frozen=True)
class Zone:
    """A contiguous run of keys served by one stored sample, with the option the solver chose."""

    pitches: tuple[int, ...]
    representative: int
    representative_velocity: int
    weight: float  # total material usage (seconds) across the zone's pitches
    chosen: ZoneOption
    hull: tuple[ZoneOption, ...]


@dataclass(frozen=True)
class GroupingResult:
    """The solver's output: the chosen zones and the totals they add up to."""

    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float


@dataclass(frozen=True)
class GroupedInstrumentPlan(BudgetedPlanMixin):
    """A full grouped optimization: the velocity map, the zones, and the budget they fit within."""

    instrument_id: str
    budget: BudgetBreakdown
    velocity_map: VelocityVolumeMap
    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float

    @property
    def used_bytes(self) -> int:
        return self.total_bytes

    @property
    def pitches(self) -> tuple[int, ...]:
        """Every covered key, ascending (the zones already partition them in order)."""
        return tuple(pitch for zone in self.zones for pitch in zone.pitches)

    @property
    def total_weight(self) -> float:
        return sum(zone.weight for zone in self.zones)
