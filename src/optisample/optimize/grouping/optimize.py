from optisample.model import InstrumentSpec
from optisample.optimize.layers.allocate import allocate_layers
from optisample.optimize.orchestrate import RunInputs, prepare_run
from optisample.optimize.orchestrate.audio import load_run_audio
from optisample.optimize.orchestrate.looping import run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan
from optisample.optimize.tasks import StoredRecordings


def allocate_instrument_grouped(
    instrument: InstrumentSpec,
    inputs: RunInputs,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Choose the instrument's velocity layers and solve the pitch partition and allocation across them.

    Reads back the run :func:`~optisample.optimize.orchestrate.prepare_run` built, which is the same
    velocity map and recordings the ungrouped solver allocates over -- that shared derivation is what
    makes the two strategies' objectives comparable. Grouping always solves with the exact DP, so
    ``settings.method`` applies to the ungrouped solver alone.
    """
    allocation = allocate_layers(instrument, inputs, settings)

    return GroupedInstrumentPlan(
        instrument_id=instrument.id,
        budget=allocation.budget,
        velocity_map=inputs.velocity_map,
        layers=allocation.layers,
        zones=allocation.zones,
        total_bytes=allocation.total_bytes,
        objective=allocation.objective,
        reserve=allocation.reserve,
        energy_exponent=settings.energy_exponent,
        reduction=inputs.reduction,
    )


def optimize_instrument_grouped(
    instrument: InstrumentSpec,
    recordings: StoredRecordings,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Optimize one instrument with pitch-zone grouping and return a structured plan."""
    inputs = prepare_run(instrument, recordings, settings)
    return allocate_instrument_grouped(instrument, inputs, settings)


def run_instrument_grouped(
    instrument: InstrumentSpec,
    settings: OptimizeSettings,
) -> GroupedInstrumentPlan:
    """Load an instrument's recordings from disk, settle its loops, and optimize it with grouping."""
    looped = run_loops(load_run_audio(instrument, settings), settings)
    return optimize_instrument_grouped(looped.loaded.instrument, looped.recordings, settings)
