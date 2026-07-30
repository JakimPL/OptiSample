from dataclasses import dataclass
from typing import Final, Literal

from optisample.dsp.surrogate import EncodingParams
from optisample.keys import SampleKey
from optisample.music import note_name
from optisample.optimize.layers.bands import VelocityLayers
from optisample.optimize.plans.budget import BudgetBreakdown, BudgetedPlanMixin
from optisample.optimize.plans.strategy import SampleUnit
from optisample.optimize.reduce.summary import ReductionSummary
from optisample.optimize.velocity_map import VelocityVolumeMap


@dataclass(frozen=True)
class ZoneOption:
    """One way to realize a zone: which member is the stored representative and how it is encoded.

    ``distortion`` is the *usage-weighted, summed* reconstruction distortion over every pitch the
    zone covers (so it is directly comparable to the ungrouped objective), and ``stored_bytes``
    already includes the records the target format charges for the one sample the zone stores.
    """

    representative: int
    params: EncodingParams
    stored_bytes: int
    distortion: float
    frames: int


@dataclass(frozen=True)
class Zone:
    """A contiguous run of keys served by one stored sample, with the option the solver chose.

    ``layer`` is the velocity band the zone answers for, so a key played softly and loudly is covered by
    one zone in each layer and each of them stores the recording its own dynamics are nearest.
    """

    pitches: tuple[int, ...]
    layer: int
    representative_key: SampleKey
    weight: float  # total material usage (seconds) across the zone's pitches
    chosen: ZoneOption
    hull: tuple[ZoneOption, ...]

    @property
    def representative(self) -> int:
        """The pitch every key in the zone is repitched from (the stored recording's own pitch)."""
        return self.representative_key.pitch


@dataclass(frozen=True)
class GroupingResult:
    """The solver's output: the chosen zones and the totals they add up to."""

    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float


NO_RESERVE: Final = 0  # what a stored sample is charged beyond its own bytes while the cap leaves room


@dataclass(frozen=True)
class SampleReserve:
    """The cap on stored samples a solve was held to, and the charge per sample that held it there.

    A cap is met by pricing every stored sample above the bytes it occupies, which moves the partition
    toward fewer and wider zones. ``cap`` is the most samples the plan may keep, ``bytes_per_sample``
    the charge the search settled on, and ``objective_uncapped`` what the same budget scored when each
    sample paid its own bytes alone -- the three together state what meeting the cap was worth.
    """

    cap: int
    bytes_per_sample: int
    objective_uncapped: float

    @property
    def binding(self) -> bool:
        """Whether meeting the cap took a charge, which is when the cap decided what the plan stores."""
        return self.bytes_per_sample > NO_RESERVE


@dataclass(frozen=True)
class GroupedInstrumentPlan(BudgetedPlanMixin):
    """A full grouped optimization: the velocity map, the layers and zones, and the budget they fit within.

    ``layers`` is the velocity split the allocation settled on -- one band per stored instrument, in the
    order the zones are grouped by -- so a note's dynamic names the instrument it is played through.
    ``energy_exponent`` is how steeply each note's own energy scaled its distortion, which states what
    the ``objective`` means and so which other plans it may be compared with. ``reserve`` states the
    sample cap the solve was held to and what holding it there cost.
    """

    instrument_id: str
    budget: BudgetBreakdown
    velocity_map: VelocityVolumeMap
    layers: VelocityLayers
    zones: tuple[Zone, ...]
    total_bytes: int
    objective: float
    reserve: SampleReserve
    energy_exponent: float
    reduction: ReductionSummary
    strategy: Literal["grouped"] = "grouped"

    @property
    def used_bytes(self) -> int:
        return self.total_bytes

    @property
    def pitches(self) -> tuple[int, ...]:
        """Every covered key, ascending and named once however many layers store a recording for it."""
        return tuple(sorted({pitch for zone in self.zones for pitch in zone.pitches}))

    @property
    def total_weight(self) -> float:
        return sum(zone.weight for zone in self.zones)

    def sample_units(self) -> tuple[SampleUnit, ...]:
        """One stored sample per zone, every key the zone covers routed to its repitched representative."""
        return tuple(
            SampleUnit(
                label=f"zone{index:02d}_rep{zone.representative:03d}_{note_name(zone.representative)}",
                representative_key=zone.representative_key,
                layer=zone.layer,
                keys=zone.pitches,
                params=zone.chosen.params,
                frames=zone.chosen.frames,
                stored_bytes=zone.chosen.stored_bytes,
                distortion=zone.chosen.distortion,
                objective_share=zone.chosen.distortion,
                hull_size=len(zone.hull),
                weight=zone.weight,
            )
            for index, zone in enumerate(self.zones)
        )
