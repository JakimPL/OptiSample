import numpy as np

from optisample.model import InstrumentSpec
from optisample.optimize.knapsack import rd_curve
from optisample.optimize.orchestrate.audio import load_instrument_audio
from optisample.optimize.orchestrate.cost_model import build_items
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.orchestrate.solve import solve_allocation
from optisample.optimize.plans import InstrumentPlan, per_key_bytes, split_budget
from optisample.optimize.tasks import AudioMap, EvalContext, PitchTask, build_tasks
from optisample.optimize.velocity_map import (
    VelocityVolumeMap,
    derive_velocity_map,
    loudness_by_velocity,
)


def prepare_run(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    settings: OptimizeSettings,
) -> tuple[VelocityVolumeMap, EvalContext, list[PitchTask]]:
    """Build the shared inputs both optimizers need: the velocity map, scoring context and pitch tasks.

    Kept in one place so the ungrouped solver and pitch-zone grouping derive the map and tasks
    identically (the grouping objective must be comparable to the ungrouped one). The context also
    carries the byte scale the reductions aim at -- the sample budget split evenly across the pitches
    the material plays -- so both strategies narrow their stored grids around the same target.
    """
    velocity_map = derive_velocity_map(
        loudness_by_velocity([(key.velocity, signal) for key, signal in audio.items()], sample_rate),
        settings.velocity,
    )
    tasks = build_tasks(instrument, audio, velocity_map, settings.reduce)
    budget = split_budget(instrument.budget_kb, settings.target.storage)
    context = EvalContext(
        sample_rate=sample_rate,
        composite=settings.composite,
        rng=np.random.default_rng(settings.seed),
        sweep=settings.sweep,
        encode=settings.encode,
        storage=settings.target.storage,
        bandwidth=settings.reduce.bandwidth,
        byte_target=per_key_bytes(budget, len(tasks)),
    )
    return velocity_map, context, tasks


def optimize_instrument(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    settings: OptimizeSettings,
) -> InstrumentPlan:
    """Optimize one instrument's byte budget end to end and return a structured plan."""
    velocity_map, context, tasks = prepare_run(instrument, audio, sample_rate, settings)
    items, hulls = build_items(tasks, context)

    budget = split_budget(instrument.budget_kb, settings.target.storage)
    allocation, pitches = solve_allocation(tasks, items, hulls, budget.sample_bytes, settings.method)

    return InstrumentPlan(
        instrument_id=instrument.id,
        budget=budget,
        velocity_map=velocity_map,
        pitches=pitches,
        allocation=allocation,
        curve=tuple(rd_curve(items)),
        method=settings.method,
    )


def run_instrument(instrument: InstrumentSpec, settings: OptimizeSettings) -> InstrumentPlan:
    """Load an instrument's recordings from disk and optimize it."""
    audio, sample_rate = load_instrument_audio(instrument, settings.reduce.dedupe, settings.encode.loop)
    return optimize_instrument(instrument, audio, sample_rate, settings)


__all__ = [
    "optimize_instrument",
    "prepare_run",
    "run_instrument",
]
