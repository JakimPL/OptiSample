"""End-to-end budget optimization of a single instrument (no pitch grouping, no loop yet).

IT stores one sample per key and has no velocity layers, so even without pitch-zone grouping the
velocity axis must collapse: for each pitch used by the material we store one recording (the loudest
velocity actually played there, kept peak-normalized) and reproduce every other dynamic through the
velocity->volume map. The optimizer then:

1. builds the instrument-wide velocity->volume map from the recordings' loudness;
2. for each pitch, sweeps encoding configs (rate x depth) and scores the *reconstruction* distortion
   -- source note vs. render(stored, volume=map(velocity), duration) -- weighted by material usage;
3. picks one config per pitch with the MCKP solver under the instrument's byte budget;
4. emits a structured plan and a human-readable report.

Grouping (pitch zones, k-medoids representatives) is P5; loops and envelopes are P6; the IT writer is
P4. Distortion is the loudness-normalized composite, so the volume map fixes level exactly and is not
"paid for" here -- what remains is encoding loss plus the velocity-timbre the single sample can't cover.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig, VelocityConfig
from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate import EncodeContext, EncodingParams, encode
from optisample.io.audio import read_wav
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity
from optisample.model import InstrumentSpec
from optisample.optimize.knapsack import Allocation, KnapsackItem, rd_curve, solve_exact, solve_lagrangian
from optisample.optimize.operating_points import OperatingPoint, lower_convex_hull, sweep_rates
from optisample.optimize.plans import InstrumentPlan, PitchPlan, split_budget
from optisample.optimize.tasks import AudioMap, EvalContext, PitchTask, build_tasks, score_reconstruction
from optisample.optimize.velocity_map import VelocityVolumeMap, derive_velocity_map, loudness_by_velocity

Method = Literal["exact", "lagrangian"]


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


def _evaluate_config(task: PitchTask, ctx: EvalContext, params: EncodingParams) -> OperatingPoint:
    """Encode the pitch's own representative, then score reconstruction against every event at it."""
    encode_ctx = EncodeContext(root_pitch=task.pitch, config=ctx.encode, rng=ctx.rng)
    stored = encode(task.representative, ctx.sample_rate, params, encode_ctx)
    distortion = score_reconstruction(stored, task, ctx)
    return OperatingPoint(params=params, stored_bytes=stored.stored_bytes, distortion=distortion, frames=stored.frames)


def _pitch_points(task: PitchTask, ctx: EvalContext) -> list[OperatingPoint]:
    """Sweep the rate x depth grid for one pitch, trimming storage to its longest note."""
    rates = sweep_rates(ctx.sweep, ctx.sample_rate)
    trim_s = task.max_duration_s
    points: list[OperatingPoint] = []
    for loop in ctx.sweep.loops:
        for depth in ctx.sweep.depths:
            for rate in rates:
                params = EncodingParams(
                    target_rate=rate,
                    depth_bits=depth,
                    trim_s=trim_s,
                    dither=ctx.sweep.dither,
                    noise_shaping=ctx.sweep.noise_shaping,
                    loop=loop,
                )
                points.append(_evaluate_config(task, ctx, params))
    return points


def _build_items(
    tasks: Sequence[PitchTask], ctx: EvalContext
) -> tuple[tuple[KnapsackItem, ...], dict[int, tuple[OperatingPoint, ...]]]:
    """Turn each pitch task into a knapsack item plus its lower-convex-hull configs."""
    items: list[KnapsackItem] = []
    hulls: dict[int, tuple[OperatingPoint, ...]] = {}
    for task in tasks:
        points = tuple(_pitch_points(task, ctx))
        hulls[task.pitch] = tuple(lower_convex_hull(points))
        items.append(KnapsackItem(key=str(task.pitch), weight=task.weight, points=points))
    return tuple(items), hulls


def _pitch_plans(
    tasks: Sequence[PitchTask], allocation: Allocation, hulls: Mapping[int, tuple[OperatingPoint, ...]]
) -> tuple[PitchPlan, ...]:
    """Attach the solver's chosen config to each pitch task."""
    chosen = {selection.key: selection.point for selection in allocation.selections}
    return tuple(
        PitchPlan(
            pitch=task.pitch,
            weight=task.weight,
            representative_velocity=task.representative_velocity,
            chosen=chosen[str(task.pitch)],
            hull=hulls[task.pitch],
        )
        for task in tasks
    )


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
    items, hulls = _build_items(tasks, ctx)

    budget = split_budget(instrument.budget_kb)
    solve = solve_exact if settings.method == "exact" else solve_lagrangian
    allocation = solve(items, budget.sample_bytes)

    return InstrumentPlan(
        instrument_id=instrument.id,
        budget=budget,
        velocity_map=velocity_map,
        pitches=_pitch_plans(tasks, allocation, hulls),
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
