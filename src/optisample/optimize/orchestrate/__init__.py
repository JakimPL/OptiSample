"""End-to-end budget optimization of a single instrument, one stored sample per key (no grouping).

IT stores one sample per key and has no velocity layers, so even without pitch-zone grouping the
velocity axis must collapse: for each pitch the material uses we store one recording (the loudest
velocity actually played there, kept peak-normalized) and reproduce every other dynamic through the
velocity->volume map. Distortion is the loudness-normalized composite, so the volume map fixes level
exactly and is not "paid for" here -- what remains is encoding loss plus the velocity-timbre a single
recording cannot cover across a key's dynamics.

The work splits across the subpackage:

* :mod:`.cost_model` sweeps each pitch's rate x depth encoding grid into one knapsack item plus its
  lower-convex-hull configs;
* :mod:`.solve` runs the MCKP allocation under the byte budget and attaches the chosen config to each
  pitch;
* this module builds the shared velocity map and per-pitch tasks (reused as-is by pitch-zone
  grouping), then wires the pieces into an :class:`~optisample.optimize.plans.InstrumentPlan`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig, VelocityConfig
from optisample.dsp.resample import resample_to
from optisample.io.audio import read_wav
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity
from optisample.model import InstrumentSpec
from optisample.optimize.knapsack import rd_curve
from optisample.optimize.orchestrate.cost_model import build_items
from optisample.optimize.orchestrate.solve import Method, solve_allocation
from optisample.optimize.plans import InstrumentPlan, split_budget
from optisample.optimize.tasks import AudioMap, EvalContext, PitchTask, build_tasks
from optisample.optimize.velocity_map import VelocityVolumeMap, derive_velocity_map, loudness_by_velocity


@dataclass(frozen=True)
class OptimizeSettings:
    """Knobs for one optimization run (bundled to keep the call site small).

    Carries the config the run needs -- the encoding sweep grid, the encode config, the prebuilt
    composite fidelity, the velocity-map shaping and the solver method -- all sourced from config at
    the entry point. ``seed`` drives the dither RNG (not a tuning knob, so it keeps a code default).
    """

    sweep: SweepConfig
    encode: EncodeConfig
    composite: CompositeFidelity
    velocity: VelocityConfig
    method: Method
    seed: int = 0


def prepare_run(
    instrument: InstrumentSpec, audio: AudioMap, sample_rate: int, settings: OptimizeSettings
) -> tuple[VelocityVolumeMap, EvalContext, list[PitchTask]]:
    """Build the shared inputs both optimizers need: the velocity map, scoring context and pitch tasks.

    Kept in one place so the ungrouped solver and pitch-zone grouping derive the map and tasks
    identically (the grouping objective must be comparable to the ungrouped one).
    """
    velocity_map = derive_velocity_map(
        loudness_by_velocity([(velocity, signal) for (_, velocity), signal in audio.items()], sample_rate),
        settings.velocity,
    )
    ctx = EvalContext(
        sample_rate=sample_rate,
        velocity_map=velocity_map,
        composite=settings.composite,
        rng=np.random.default_rng(settings.seed),
        sweep=settings.sweep,
        encode=settings.encode,
    )
    return velocity_map, ctx, build_tasks(instrument, audio)


def optimize_instrument(
    instrument: InstrumentSpec, audio: AudioMap, sample_rate: int, settings: OptimizeSettings
) -> InstrumentPlan:
    """Optimize one instrument's byte budget end to end and return a structured plan."""
    velocity_map, ctx, tasks = prepare_run(instrument, audio, sample_rate, settings)
    items, hulls = build_items(tasks, ctx)

    budget = split_budget(instrument.budget_kb)
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


def load_instrument_audio(instrument: InstrumentSpec) -> tuple[dict[tuple[int, int], Signal], int]:
    """Read every recording of ``instrument`` into ``(pitch, velocity) -> signal`` at a common rate."""
    audio: dict[tuple[int, int], Signal] = {}
    sample_rate = 0
    for sample in instrument.samples:
        data, rate = read_wav(sample.file)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sample_rate == 0:
            sample_rate = rate
        elif rate != sample_rate:
            data = resample_to(data, rate, sample_rate)
        audio[(sample.pitch, sample.velocity)] = np.asarray(data, dtype=np.float64)
    return audio, sample_rate


def run_instrument(instrument: InstrumentSpec, settings: OptimizeSettings) -> InstrumentPlan:
    """Load an instrument's recordings from disk and optimize it."""
    audio, sample_rate = load_instrument_audio(instrument)
    return optimize_instrument(instrument, audio, sample_rate, settings)


__all__ = [
    "OptimizeSettings",
    "load_instrument_audio",
    "optimize_instrument",
    "prepare_run",
    "run_instrument",
]
