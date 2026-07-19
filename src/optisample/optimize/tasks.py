"""Shared per-pitch material tasks and the reconstruction-scoring primitive.

Both the ungrouped optimizer (one sample per pitch, :mod:`optisample.optimize.orchestrate`) and
pitch-zone grouping (:mod:`optisample.optimize.grouping`) need the same inputs: the material grouped
by pitch, each pitch's representative recording (the loudest velocity actually played there, kept
peak-normalized) and its per-``(velocity, duration)`` ground-truth references, plus a way to score
how well a *stored* sample reconstructs a pitch's notes.

The two callers differ only in *which* stored sample scores a pitch. Ungrouped, the stored sample is
that pitch's own recording (no transpose). Grouped, it is a *different* pitch's recording, repitched
to cover this one -- which is exactly what :func:`optisample.dsp.surrogate.render` does when asked for
``pitch=task.pitch`` from a sample whose ``root_pitch`` differs. :func:`score_reconstruction` is
therefore agnostic to the stored sample's origin: it renders it at the task's pitch and compares to
the task's references.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from optisample.dsp.surrogate import StoredSample, render
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, evaluate
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.operating_points import SweepGrid
from optisample.optimize.velocity_map import VelocityVolumeMap

AudioMap = Mapping[tuple[int, int], Signal]


@dataclass(frozen=True)
class Event:
    """A distinct (velocity, duration) the material plays at one pitch, with its ground-truth source."""

    velocity: int
    duration_s: float
    weight: float
    reference: Signal


@dataclass(frozen=True)
class PitchTask:
    """Everything needed to score one pitch's configs: its representative recording and references."""

    pitch: int
    weight: float
    representative_velocity: int
    representative: Signal
    events: tuple[Event, ...]

    @property
    def max_duration_s(self) -> float:
        """Longest note the material holds at this pitch (the storage trim before any transpose)."""
        return max(event.duration_s for event in self.events)


@dataclass(frozen=True)
class EvalContext:
    """Shared scoring inputs (bundled to stay under the argument limit)."""

    sample_rate: int
    velocity_map: VelocityVolumeMap
    composite: CompositeFidelity
    rng: np.random.Generator
    grid: SweepGrid = field(default_factory=SweepGrid)


def nearest_velocity(available: Sequence[int], target: int) -> int:
    """Recorded velocity closest to ``target`` (ties favour the louder recording)."""
    return min(available, key=lambda velocity: (abs(velocity - target), -velocity))


def merge_events(events: Sequence[NoteEvent]) -> list[tuple[int, float, float]]:
    """Collapse events sharing a (velocity, duration) into ``(velocity, duration, summed weight)``."""
    weights: dict[tuple[int, float], float] = {}
    for event in events:
        key = (event.velocity, event.duration_s)
        weights[key] = weights.get(key, 0.0) + event.weight
    return [(velocity, duration, weight) for (velocity, duration), weight in weights.items()]


def build_tasks(instrument: InstrumentSpec, audio: AudioMap) -> list[PitchTask]:
    """Group the material by pitch and attach each pitch's representative recording and references.

    Returned tasks are ordered by pitch -- the order the pitch-zone partitioning DP segments over.
    """
    material = instrument.material or []
    by_pitch: dict[int, list[NoteEvent]] = {}
    for event in material:
        by_pitch.setdefault(event.pitch, []).append(event)
    velocities_at: dict[int, list[int]] = {}
    for pitch, velocity in audio:
        velocities_at.setdefault(pitch, []).append(velocity)

    tasks: list[PitchTask] = []
    for pitch, events in sorted(by_pitch.items()):
        available = sorted(velocities_at.get(pitch, []))
        if not available:
            raise ValueError(f"instrument {instrument.id!r} has no recorded sample for pitch {pitch}")
        representative_velocity = nearest_velocity(available, max(event.velocity for event in events))
        built = tuple(
            Event(velocity, duration, weight, audio[(pitch, nearest_velocity(available, velocity))])
            for velocity, duration, weight in merge_events(events)
        )
        weight = sum(event.weight for event in built)
        tasks.append(PitchTask(pitch, weight, representative_velocity, audio[(pitch, representative_velocity)], built))
    return tasks


def score_reconstruction(stored: StoredSample, task: PitchTask, ctx: EvalContext) -> float:
    """Weighted mean distortion of reconstructing ``task``'s notes from ``stored`` (repitched to its key).

    ``stored`` is rendered at ``task.pitch`` -- a transpose of ``task.pitch - stored.root_pitch``
    semitones, zero when ``stored`` is this pitch's own recording -- scaled by each event's mapped
    volume and fitted to its duration, then compared to that event's source note. The result is
    normalized per second of material so it can be reweighted by usage at the call site.
    """
    total = 0.0
    for event in task.events:
        volume = ctx.velocity_map.volume(event.velocity)
        candidate = render(stored, ctx.sample_rate, pitch=task.pitch, volume=volume, duration_s=event.duration_s)
        reference = event.reference[: max(0, int(round(event.duration_s * ctx.sample_rate)))]
        total += event.weight * evaluate(reference, candidate, ctx.sample_rate, ctx.composite).fidelity
    return total / task.weight if task.weight > 0.0 else 0.0
