from optisample.model import InstrumentSpec
from optisample.optimize.grouping.cost_model import build_zone_options
from optisample.optimize.grouping.solve import solve_grouping
from optisample.optimize.orchestrate import RunInputs, prepare_run
from optisample.optimize.orchestrate.audio import load_run_audio
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, split_budget
from optisample.optimize.tasks import AudioMap


def allocate_instrument_grouped(
    instrument: InstrumentSpec,
    inputs: RunInputs,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Score every candidate pitch zone over the prepared tasks and solve partition and allocation together.

    Reads back the run :func:`~optisample.optimize.orchestrate.prepare_run` built, which is the same
    velocity map and pitch tasks the ungrouped solver allocates over -- that shared derivation is what
    makes the two strategies' objectives comparable. Grouping always solves with the exact DP, so
    ``settings.method`` applies to the ungrouped solver alone.
    """
    options = build_zone_options(
        inputs.tasks,
        inputs.context,
        workers=settings.workers,
        progress=settings.progress,
    )

    budget = split_budget(instrument.budget_kb, settings.target.storage)
    result = solve_grouping(inputs.tasks, options, budget.sample_bytes)

    return GroupedInstrumentPlan(
        instrument_id=instrument.id,
        budget=budget,
        velocity_map=inputs.velocity_map,
        zones=result.zones,
        total_bytes=result.total_bytes,
        objective=result.objective,
        reduction=inputs.reduction,
    )


def optimize_instrument_grouped(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Optimize one instrument with pitch-zone grouping and return a structured plan."""
    inputs = prepare_run(instrument, audio, sample_rate, settings)
    return allocate_instrument_grouped(instrument, inputs, settings)


def run_instrument_grouped(
    instrument: InstrumentSpec,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Load an instrument's recordings from disk and optimize it with grouping."""
    audio, sample_rate = load_run_audio(instrument, settings)
    return optimize_instrument_grouped(instrument, audio, sample_rate, settings)
