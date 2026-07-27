from dataclasses import dataclass

import numpy as np

from optisample.model import InstrumentSpec
from optisample.optimize.knapsack import rd_curve
from optisample.optimize.orchestrate.audio import load_run_audio
from optisample.optimize.orchestrate.cost_model import build_items
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.orchestrate.solve import solve_allocation
from optisample.optimize.orchestrate.staging import staged_encode
from optisample.optimize.plans import InstrumentPlan, per_key_bytes, split_budget
from optisample.optimize.reduce.grids import GridContext
from optisample.optimize.reduce.summary import (
    ReductionInputs,
    ReductionSummary,
    summarize_reduction,
)
from optisample.optimize.tasks import AudioMap, EvalContext, PitchTask, build_tasks
from optisample.optimize.velocity_map import (
    VelocityVolumeMap,
    derive_velocity_map,
    loudness_by_velocity,
)


@dataclass(frozen=True)
class RunInputs:
    """Everything one instrument's allocation runs off, built once and shared by both strategies.

    ``reduction`` is the pre-optimization stage's own outcome: what the recorded grid, the material and
    the encoding grid came down to, and the shortlist each pitch is swept over. Both strategies carry it
    onto their plan, so the report and the artifacts state the search space the allocation was given.
    """

    velocity_map: VelocityVolumeMap
    context: EvalContext
    tasks: tuple[PitchTask, ...]
    reduction: ReductionSummary


def prepare_run(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    settings: OptimizeSettings,
) -> RunInputs:
    """Build the shared inputs both optimizers need: the velocity map, scoring context and pitch tasks.

    Kept in one place so the ungrouped solver and pitch-zone grouping derive the map and tasks
    identically (the grouping objective must be comparable to the ungrouped one). The context also
    carries the byte scale the reductions aim at -- the sample budget split evenly across the pitches
    the material plays -- so both strategies narrow their stored grids around the same target. The
    bandwidth pre-pass then runs here, once, and the sweep reads its shortlist back.

    The narrowing context is built first and the scoring context from it, so the pre-pass and the sweep
    that reads its shortlist back price and measure against one set of values. Gain staging is settled
    here too (:func:`~optisample.optimize.orchestrate.staging.staged_encode`), so every encode the run
    makes stores its clip the way the export will write it.
    """
    velocity_map = derive_velocity_map(
        loudness_by_velocity([(key.velocity, signal) for key, signal in audio.items()], sample_rate),
        settings.velocity,
    )
    tasks = build_tasks(instrument, audio, velocity_map, settings.reduce)
    budget = split_budget(instrument.budget_kb, settings.target.storage)
    grid = GridContext(
        sample_rate=sample_rate,
        metrics=settings.metrics,
        encode=staged_encode(settings.encode, settings.target, audio),
        storage=settings.target.storage,
        sweep=settings.sweep,
        bandwidth=settings.reduce.bandwidth,
        byte_target=per_key_bytes(budget, len(tasks)),
    )
    context = EvalContext(
        sample_rate=grid.sample_rate,
        composite=grid.composite,
        rng=np.random.default_rng(settings.seed),
        sweep=grid.sweep,
        encode=grid.encode,
        storage=grid.storage,
        bandwidth=grid.bandwidth,
        grouping=settings.reduce.grouping,
        byte_target=grid.byte_target,
        seed=settings.seed,
    )
    reduction = summarize_reduction(
        instrument,
        tasks,
        audio,
        ReductionInputs(
            dedupe=settings.reduce.dedupe,
            loop=grid.encode.loop,
            context=grid,
            workers=settings.workers,
            progress=settings.progress,
        ),
    )
    return RunInputs(velocity_map=velocity_map, context=context, tasks=tuple(tasks), reduction=reduction)


def allocate_instrument(
    instrument: InstrumentSpec,
    inputs: RunInputs,
    settings: OptimizeSettings,
) -> InstrumentPlan:
    """Sweep the prepared pitch tasks and solve the byte budget across them.

    Takes the run :func:`prepare_run` already built, so an instrument allocated under both strategies
    pays for the velocity map, the pitch tasks and the bandwidth pre-pass once between them.
    """
    items, hulls = build_items(inputs.tasks, inputs.reduction.shortlists(), inputs.context, settings.progress)

    budget = split_budget(instrument.budget_kb, settings.target.storage)
    allocation, pitches = solve_allocation(inputs.tasks, items, hulls, budget.sample_bytes, settings.method)

    return InstrumentPlan(
        instrument_id=instrument.id,
        budget=budget,
        velocity_map=inputs.velocity_map,
        pitches=pitches,
        allocation=allocation,
        curve=tuple(rd_curve(items)),
        method=settings.method,
        reduction=inputs.reduction,
    )


def optimize_instrument(
    instrument: InstrumentSpec,
    audio: AudioMap,
    sample_rate: int,
    settings: OptimizeSettings,
) -> InstrumentPlan:
    """Optimize one instrument's byte budget end to end and return a structured plan."""
    inputs = prepare_run(instrument, audio, sample_rate, settings)
    return allocate_instrument(instrument, inputs, settings)


def run_instrument(instrument: InstrumentSpec, settings: OptimizeSettings) -> InstrumentPlan:
    """Load an instrument's recordings from disk and optimize it."""
    audio, sample_rate = load_run_audio(instrument, settings)
    return optimize_instrument(instrument, audio, sample_rate, settings)


__all__ = [
    "RunInputs",
    "allocate_instrument",
    "optimize_instrument",
    "prepare_run",
    "run_instrument",
]
