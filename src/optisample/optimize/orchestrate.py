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
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from optisample.dsp.resample import resample_to
from optisample.dsp.surrogate import EncodingParams, encode, render
from optisample.io.audio import read_wav
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, default_composite, evaluate
from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, bytes_to_kib, kib_to_bytes
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.knapsack import (
    Allocation,
    KnapsackItem,
    RDCurvePoint,
    rd_curve,
    solve_exact,
    solve_lagrangian,
)
from optisample.optimize.operating_points import OperatingPoint, SweepGrid, default_rates, lower_convex_hull
from optisample.optimize.velocity_map import VelocityVolumeMap, derive_velocity_map, loudness_by_velocity

AudioMap = Mapping[tuple[int, int], Signal]
Method = Literal["exact", "lagrangian"]

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_CURVE_ROWS = 6


@dataclass(frozen=True)
class OptimizeSettings:
    """Knobs for one optimization run (bundled to keep the call site small)."""

    grid: SweepGrid = SweepGrid()
    method: Method = "exact"
    composite: CompositeFidelity | None = None
    seed: int = 0


@dataclass(frozen=True)
class PitchPlan:
    """The outcome for one pitch: its representative recording and the config the solver chose."""

    pitch: int
    weight: float
    representative_velocity: int
    chosen: OperatingPoint
    hull: tuple[OperatingPoint, ...]


@dataclass(frozen=True)
class BudgetBreakdown:
    """The byte budget split into the whole module and the part left for sample PCM + headers."""

    module_bytes: int
    sample_bytes: int  # module_bytes minus the file + instrument header overhead


@dataclass(frozen=True)
class InstrumentPlan:
    """The full result for one instrument: the map, per-pitch choices, and the allocation."""

    instrument_id: str
    budget: BudgetBreakdown
    velocity_map: VelocityVolumeMap
    pitches: tuple[PitchPlan, ...]
    allocation: Allocation
    curve: tuple[RDCurvePoint, ...]
    method: str

    @property
    def module_budget_bytes(self) -> int:
        return self.budget.module_bytes

    @property
    def sample_budget_bytes(self) -> int:
        return self.budget.sample_bytes

    @property
    def used_bytes(self) -> int:
        return self.allocation.total_bytes

    @property
    def objective(self) -> float:
        return self.allocation.objective

    @property
    def module_bytes(self) -> int:
        return self.used_bytes + FILE_HEADER_BYTES + INSTRUMENT_HEADER_BYTES

    @property
    def total_weight(self) -> float:
        return sum(plan.weight for plan in self.pitches)


@dataclass(frozen=True)
class _Event:
    """A distinct (velocity, duration) the material plays at one pitch, with its ground-truth source."""

    velocity: int
    duration_s: float
    weight: float
    reference: Signal


@dataclass(frozen=True)
class _PitchTask:
    """Everything needed to score one pitch's configs."""

    pitch: int
    weight: float
    representative_velocity: int
    representative: Signal
    events: tuple[_Event, ...]


@dataclass(frozen=True)
class _EvalContext:
    """Shared scoring inputs (bundled to stay under the argument limit)."""

    sample_rate: int
    velocity_map: VelocityVolumeMap
    composite: CompositeFidelity
    rng: np.random.Generator
    grid: SweepGrid = field(default_factory=SweepGrid)


def _nearest_velocity(available: Sequence[int], target: int) -> int:
    """Recorded velocity closest to ``target`` (ties favour the louder recording)."""
    return min(available, key=lambda velocity: (abs(velocity - target), -velocity))


def _merge_events(events: Sequence[NoteEvent]) -> list[tuple[int, float, float]]:
    """Collapse events sharing a (velocity, duration) into ``(velocity, duration, summed weight)``."""
    weights: dict[tuple[int, float], float] = {}
    for event in events:
        key = (event.velocity, event.duration_s)
        weights[key] = weights.get(key, 0.0) + event.weight
    return [(velocity, duration, weight) for (velocity, duration), weight in weights.items()]


def _build_tasks(instrument: InstrumentSpec, audio: AudioMap) -> list[_PitchTask]:
    """Group the material by pitch and attach each pitch's representative recording and references."""
    material = instrument.material or []
    by_pitch: dict[int, list[NoteEvent]] = {}
    for event in material:
        by_pitch.setdefault(event.pitch, []).append(event)
    velocities_at: dict[int, list[int]] = {}
    for pitch, velocity in audio:
        velocities_at.setdefault(pitch, []).append(velocity)

    tasks: list[_PitchTask] = []
    for pitch, events in sorted(by_pitch.items()):
        available = sorted(velocities_at.get(pitch, []))
        if not available:
            raise ValueError(f"instrument {instrument.id!r} has no recorded sample for pitch {pitch}")
        representative_velocity = _nearest_velocity(available, max(event.velocity for event in events))
        built = tuple(
            _Event(velocity, duration, weight, audio[(pitch, _nearest_velocity(available, velocity))])
            for velocity, duration, weight in _merge_events(events)
        )
        weight = sum(event.weight for event in built)
        tasks.append(_PitchTask(pitch, weight, representative_velocity, audio[(pitch, representative_velocity)], built))
    return tasks


def _evaluate_config(task: _PitchTask, ctx: _EvalContext, params: EncodingParams) -> OperatingPoint:
    """Encode the representative once, then score reconstruction against every event at this pitch."""
    stored = encode(task.representative, ctx.sample_rate, params, root_pitch=task.pitch, rng=ctx.rng)
    total = 0.0
    for event in task.events:
        volume = ctx.velocity_map.volume(event.velocity)
        candidate = render(stored, ctx.sample_rate, pitch=task.pitch, volume=volume, duration_s=event.duration_s)
        reference = event.reference[: max(0, int(round(event.duration_s * ctx.sample_rate)))]
        total += event.weight * evaluate(reference, candidate, ctx.sample_rate, ctx.composite).fidelity
    distortion = total / task.weight if task.weight > 0.0 else 0.0
    return OperatingPoint(params=params, stored_bytes=stored.stored_bytes, distortion=distortion, frames=stored.frames)


def _pitch_points(task: _PitchTask, ctx: _EvalContext) -> list[OperatingPoint]:
    """Sweep the rate x depth grid for one pitch, trimming storage to its longest note."""
    rates = ctx.grid.rates if ctx.grid.rates is not None else default_rates(ctx.sample_rate)
    trim_s = max(event.duration_s for event in task.events)
    points: list[OperatingPoint] = []
    for depth in ctx.grid.depths:
        for rate in rates:
            params = EncodingParams(
                target_rate=rate,
                depth_bits=depth,
                trim_s=trim_s,
                dither=ctx.grid.dither,
                noise_shaping=ctx.grid.noise_shaping,
            )
            points.append(_evaluate_config(task, ctx, params))
    return points


def _build_items(
    tasks: Sequence[_PitchTask], ctx: _EvalContext
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
    tasks: Sequence[_PitchTask], allocation: Allocation, hulls: Mapping[int, tuple[OperatingPoint, ...]]
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


def optimize_instrument(
    instrument: InstrumentSpec, audio: AudioMap, sample_rate: int, settings: OptimizeSettings = OptimizeSettings()
) -> InstrumentPlan:
    """Optimize one instrument's byte budget end to end and return a structured plan."""
    velocity_map = derive_velocity_map(
        loudness_by_velocity([(velocity, signal) for (_, velocity), signal in audio.items()], sample_rate)
    )
    ctx = _EvalContext(
        sample_rate=sample_rate,
        velocity_map=velocity_map,
        composite=settings.composite if settings.composite is not None else default_composite(),
        rng=np.random.default_rng(settings.seed),
        grid=settings.grid,
    )
    tasks = _build_tasks(instrument, audio)
    items, hulls = _build_items(tasks, ctx)

    module_budget = kib_to_bytes(instrument.budget_kb)
    sample_budget = module_budget - FILE_HEADER_BYTES - INSTRUMENT_HEADER_BYTES
    solve = solve_exact if settings.method == "exact" else solve_lagrangian
    allocation = solve(items, sample_budget)

    return InstrumentPlan(
        instrument_id=instrument.id,
        budget=BudgetBreakdown(module_bytes=module_budget, sample_bytes=sample_budget),
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


def run_instrument(instrument: InstrumentSpec, settings: OptimizeSettings = OptimizeSettings()) -> InstrumentPlan:
    """Load an instrument's recordings from disk and optimize it."""
    audio, sample_rate = load_instrument_audio(instrument)
    return optimize_instrument(instrument, audio, sample_rate, settings)


def _note_name(pitch: int) -> str:
    """MIDI note number -> scientific pitch name (60 -> ``C4``)."""
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _format_header(plan: InstrumentPlan) -> str:
    overhead = FILE_HEADER_BYTES + INSTRUMENT_HEADER_BYTES
    used = plan.used_bytes
    fraction = used / plan.sample_budget_bytes if plan.sample_budget_bytes > 0 else float("nan")
    headroom = plan.sample_budget_bytes - used
    return "\n".join(
        (
            f"Instrument {plan.instrument_id!r} - budget solver (method: {plan.method})",
            "=" * 70,
            f"Budget:    {bytes_to_kib(plan.module_budget_bytes):7.1f} KiB module  ->  "
            f"{bytes_to_kib(plan.sample_budget_bytes):7.1f} KiB samples  "
            f"(overhead {overhead} B: file {FILE_HEADER_BYTES} + instrument {INSTRUMENT_HEADER_BYTES})",
            f"Used:      {bytes_to_kib(used):7.1f} KiB samples  ({fraction:6.1%} of budget, "
            f"{bytes_to_kib(headroom):.1f} KiB free)  ->  {bytes_to_kib(plan.module_bytes):.1f} KiB module",
            f"Objective: {plan.objective:8.4f}  (sum of weight x distortion over "
            f"{len(plan.pitches)} pitches, {plan.total_weight:.1f} s of material)",
        )
    )


def _format_pitches(plan: InstrumentPlan) -> str:
    lines = [
        "Per-pitch allocation",
        "-" * 70,
        f"{'pitch':>5}  {'note':>4}  {'weight(s)':>9}  {'rep.vel':>7}  "
        f"{'rate(Hz)':>8}  {'depth':>5}  {'size(KiB)':>9}  {'distortion':>10}  {'hull':>4}",
    ]
    for pitch in plan.pitches:
        point = pitch.chosen
        lines.append(
            f"{pitch.pitch:>5}  {_note_name(pitch.pitch):>4}  {pitch.weight:>9.1f}  "
            f"{pitch.representative_velocity:>7}  {point.params.target_rate:>8}  {point.params.depth_bits:>5}  "
            f"{bytes_to_kib(point.stored_bytes):>9.1f}  {point.distortion:>10.4f}  {len(pitch.hull):>4}"
        )
    return "\n".join(lines)


def _format_velocity_map(plan: InstrumentPlan) -> str:
    anchors = plan.velocity_map.anchors
    reference_volume = max((anchor.volume for anchor in anchors), default=0)
    lines = ["Velocity->volume map (loudness-matched, 0-64; loudest velocity -> 64)", "-" * 70]
    for anchor in anchors:
        marker = "  [reference]" if anchor.volume == reference_volume else ""
        lines.append(
            f"  vel {anchor.velocity:>3}  ->  vol {anchor.volume:>2}   ({anchor.loudness_lufs:6.1f} LUFS){marker}"
        )
    lines.append("  (full 0-127 map interpolated between these anchors)")
    return "\n".join(lines)


def _format_curve(plan: InstrumentPlan) -> str:
    curve = plan.curve
    lines = ["Budget->quality curve (Lagrangian sweep over the RD hulls)", "-" * 70]
    step = max(1, (len(curve) - 1) // (_CURVE_ROWS - 1)) if len(curve) > 1 else 1
    shown = list(range(0, len(curve), step))
    if shown and shown[-1] != len(curve) - 1:
        shown.append(len(curve) - 1)
    for position in shown:
        point = curve[position]
        fits = "" if point.total_bytes > plan.sample_budget_bytes else "  <= budget"
        lines.append(f"  {bytes_to_kib(point.total_bytes):7.1f} KiB  ->  objective {point.objective:8.4f}{fits}")
    return "\n".join(lines)


def format_report(plan: InstrumentPlan) -> str:
    """Render a human-readable summary of an instrument optimization."""
    sections = (_format_header(plan), _format_pitches(plan), _format_velocity_map(plan), _format_curve(plan))
    return "\n\n".join(sections) + "\n"
