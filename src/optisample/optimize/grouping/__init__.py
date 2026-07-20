"""Pitch-zone grouping: cover a keyboard with fewer stored samples by repitching representatives.

IT stores one sample per key, but nothing forces every key to own a *distinct* sample -- a tracker
happily repitches one recording across a range of keys (that is what ``C5Speed`` + the note map are
for). So when the byte budget is tight we partition the used pitches into **contiguous zones**, store
one representative recording per zone, and let the player transpose it to the zone's other keys. Each
sample we drop saves its PCM *and* its 80-byte header; what we pay is repitching artifacts plus the
timbre a single recording cannot match across the zone.

The work splits across the subpackage:

* :mod:`.cost_model` enumerates each candidate zone's ``(representative, encoding)`` options;
* :mod:`.solve` runs the exact partition + allocation DP that picks the zones and their options;
* this module reuses the ungrouped pipeline's velocity map and per-pitch tasks, then wires the two
  together and wraps the result in a :class:`GroupedInstrumentPlan`.
"""

from __future__ import annotations

from dataclasses import dataclass

from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, kib_to_bytes
from optisample.model import InstrumentSpec
from optisample.optimize.grouping.cost_model import ZoneOption, build_zone_options, zone_hull
from optisample.optimize.grouping.solve import GroupingResult, Zone, solve_grouping
from optisample.optimize.orchestrate import (
    BudgetBreakdown,
    OptimizeSettings,
    load_instrument_audio,
    prepare_run,
)
from optisample.optimize.tasks import AudioMap
from optisample.optimize.velocity_map import VelocityVolumeMap


@dataclass(frozen=True)
class GroupedInstrumentPlan:
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
    def sample_budget_bytes(self) -> int:
        return self.budget.sample_bytes

    @property
    def module_budget_bytes(self) -> int:
        return self.budget.module_bytes

    @property
    def module_bytes(self) -> int:
        return self.used_bytes + FILE_HEADER_BYTES + INSTRUMENT_HEADER_BYTES

    @property
    def pitches(self) -> tuple[int, ...]:
        """Every covered key, ascending (the zones already partition them in order)."""
        return tuple(pitch for zone in self.zones for pitch in zone.pitches)

    @property
    def total_weight(self) -> float:
        return sum(zone.weight for zone in self.zones)


def optimize_instrument_grouped(
    instrument: InstrumentSpec, audio: AudioMap, sample_rate: int, settings: OptimizeSettings
) -> GroupedInstrumentPlan:
    """Optimize one instrument with pitch-zone grouping and return a structured plan.

    Reuses the ungrouped pipeline's velocity map and per-pitch tasks; only the allocation changes
    (zones of repitched representatives instead of one sample per key). ``settings.method`` is not
    used -- grouping always solves partition and allocation together with the exact DP.
    """
    velocity_map, ctx, tasks = prepare_run(instrument, audio, sample_rate, settings)
    options = build_zone_options(tasks, ctx)

    module_budget = kib_to_bytes(instrument.budget_kb)
    sample_budget = module_budget - FILE_HEADER_BYTES - INSTRUMENT_HEADER_BYTES
    result = solve_grouping(tasks, options, sample_budget)

    return GroupedInstrumentPlan(
        instrument_id=instrument.id,
        budget=BudgetBreakdown(module_bytes=module_budget, sample_bytes=sample_budget),
        velocity_map=velocity_map,
        zones=result.zones,
        total_bytes=result.total_bytes,
        objective=result.objective,
    )


def run_instrument_grouped(instrument: InstrumentSpec, settings: OptimizeSettings) -> GroupedInstrumentPlan:
    """Load an instrument's recordings from disk and optimize it with grouping."""
    audio, sample_rate = load_instrument_audio(instrument)
    return optimize_instrument_grouped(instrument, audio, sample_rate, settings)


__all__ = [
    "GroupedInstrumentPlan",
    "GroupingResult",
    "Zone",
    "ZoneOption",
    "build_zone_options",
    "optimize_instrument_grouped",
    "run_instrument_grouped",
    "solve_grouping",
    "zone_hull",
]
