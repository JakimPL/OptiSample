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

from optisample.model import InstrumentSpec
from optisample.optimize.grouping.cost_model import build_zone_options, zone_hull
from optisample.optimize.grouping.solve import solve_grouping
from optisample.optimize.orchestrate import OptimizeSettings, load_instrument_audio, prepare_run
from optisample.optimize.plans import GroupedInstrumentPlan, split_budget
from optisample.optimize.tasks import AudioMap


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

    budget = split_budget(instrument.budget_kb)
    result = solve_grouping(tasks, options, budget.sample_bytes)

    return GroupedInstrumentPlan(
        instrument_id=instrument.id,
        budget=budget,
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
    "build_zone_options",
    "optimize_instrument_grouped",
    "run_instrument_grouped",
    "solve_grouping",
    "zone_hull",
]
