from optisample.model import InstrumentSpec
from optisample.optimize.grouping.cost_model import build_zone_options
from optisample.optimize.grouping.solve import solve_grouping
from optisample.optimize.orchestrate import prepare_run
from optisample.optimize.orchestrate.audio import load_instrument_audio
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, split_budget
from optisample.optimize.tasks import AudioMap


def optimize_instrument_grouped(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Optimize one instrument with pitch-zone grouping and return a structured plan.

    Reuses the ungrouped pipeline's velocity map and per-pitch tasks; the allocation switches to
    zones of repitched representatives. Grouping always solves partition and allocation together with
    the exact DP, so ``settings.method`` applies only to the ungrouped solver.
    """
    velocity_map, context, tasks = prepare_run(
        instrument,
        audio,
        sample_rate,
        settings,
    )
    options = build_zone_options(tasks, context)

    budget = split_budget(instrument.budget_kb, settings.target.storage)
    result = solve_grouping(tasks, options, budget.sample_bytes)

    return GroupedInstrumentPlan(
        instrument_id=instrument.id,
        budget=budget,
        velocity_map=velocity_map,
        zones=result.zones,
        total_bytes=result.total_bytes,
        objective=result.objective,
    )


def run_instrument_grouped(
    instrument: InstrumentSpec,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Load an instrument's recordings from disk and optimize it with grouping."""
    audio, sample_rate = load_instrument_audio(instrument)
    return optimize_instrument_grouped(instrument, audio, sample_rate, settings)
