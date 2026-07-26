from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.dsp.surrogate import StoredSample, render
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, QualityReport, evaluate
from optisample.model import InstrumentSpec, NoteEvent
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
    sweep: SweepConfig
    encode: EncodeConfig


def nearest_velocity(available: Sequence[int], target: int) -> int:
    """Recorded velocity closest to ``target`` (ties favour the louder recording)."""
    return min(available, key=lambda velocity: (abs(velocity - target), -velocity))


@dataclass(frozen=True)
class MergedEvent:
    """A distinct (velocity, duration) the material plays at a pitch, with its summed usage weight."""

    velocity: int
    duration_s: float
    weight: float


def merge_events(events: Sequence[NoteEvent]) -> list[MergedEvent]:
    """Collapse events sharing a (velocity, duration) into one :class:`MergedEvent` with summed weight."""
    weights: dict[tuple[int, float], float] = {}
    for event in events:
        key = (event.velocity, event.duration_s)
        weights[key] = weights.get(key, 0.0) + event.weight

    return [
        MergedEvent(
            velocity,
            duration,
            weight,
        )
        for (velocity, duration), weight in weights.items()
    ]


def _group_events_by_pitch(material: Sequence[NoteEvent]) -> dict[int, list[NoteEvent]]:
    """Bucket the material's note events by the pitch they play."""
    by_pitch: dict[int, list[NoteEvent]] = {}
    for event in material:
        by_pitch.setdefault(event.pitch, []).append(event)

    return by_pitch


def _recorded_velocities(audio: AudioMap) -> dict[int, list[int]]:
    """The velocities actually recorded at each pitch (``audio``'s keys, grouped by pitch)."""
    velocities_at: dict[int, list[int]] = {}
    for pitch, velocity in audio:
        velocities_at.setdefault(pitch, []).append(velocity)

    return velocities_at


def _build_pitch_task(
    pitch: int,
    events: Sequence[NoteEvent],
    available: Sequence[int],
    audio: AudioMap,
) -> PitchTask:
    """Assemble one pitch's task: its representative recording and per-(velocity, duration) references.

    Each distinct ``(velocity, duration)`` the material plays becomes an :class:`Event` referenced by the
    recording at the nearest available velocity; the representative is the recording nearest the loudest
    velocity played here (the sample the zone stores when this pitch is chosen representative).
    """
    representative_velocity = nearest_velocity(available, max(event.velocity for event in events))
    built = tuple(
        Event(
            merged.velocity,
            merged.duration_s,
            merged.weight,
            audio[(pitch, nearest_velocity(available, merged.velocity))],
        )
        for merged in merge_events(events)
    )
    weight = sum(event.weight for event in built)
    return PitchTask(
        pitch,
        weight,
        representative_velocity,
        audio[(pitch, representative_velocity)],
        built,
    )


def build_tasks(
    instrument: InstrumentSpec,
    audio: AudioMap,
) -> list[PitchTask]:
    """Group the material by pitch and attach each pitch's representative recording and references.

    Returned tasks are ordered by pitch -- the order the pitch-zone partitioning DP segments over.
    """
    by_pitch = _group_events_by_pitch(instrument.material or [])
    velocities_at = _recorded_velocities(audio)

    tasks: list[PitchTask] = []
    for pitch, events in sorted(by_pitch.items()):
        available = sorted(velocities_at.get(pitch, []))
        if not available:
            raise ValueError(f"instrument {instrument.id!r} has no recorded sample for pitch {pitch}")

        tasks.append(_build_pitch_task(pitch, events, available, audio))

    return tasks


@dataclass(frozen=True)
class EventScore:
    """One event scored: the note, the volume it mapped to, and the fidelity report of its reconstruction."""

    event: Event
    volume: int
    report: QualityReport

    @property
    def weighted_fidelity(self) -> float:
        """This event's contribution to the objective: its usage weight times its distortion."""
        return self.event.weight * self.report.fidelity


def score_events(
    stored: StoredSample,
    task: PitchTask,
    context: EvalContext,
) -> Iterator[EventScore]:
    """Reconstruct each of ``task``'s notes from ``stored`` and score it, one :class:`EventScore` per event.

    ``stored`` is rendered at ``task.pitch`` -- a transpose of ``task.pitch - stored.root_pitch``
    semitones, zero when ``stored`` is this pitch's own recording -- scaled by each event's mapped volume
    and fitted to its duration, then compared to that event's source note. This is the shared scorer both
    the objective and ``metrics.json`` consume.
    """
    for event in task.events:
        volume = context.velocity_map.volume(event.velocity)
        candidate = render(
            stored,
            context.sample_rate,
            pitch=task.pitch,
            volume=volume,
            duration_s=event.duration_s,
        )
        reference = event.reference[: seconds_to_frames(event.duration_s, context.sample_rate)]
        yield EventScore(
            event=event,
            volume=volume,
            report=evaluate(
                reference,
                candidate,
                context.sample_rate,
                context.composite,
            ),
        )


def score_reconstruction(
    stored: StoredSample,
    task: PitchTask,
    context: EvalContext,
) -> float:
    """Weighted mean distortion of reconstructing ``task``'s notes from ``stored`` (repitched to its key).

    The weighted sum of :func:`score_events` normalized per unit of material weight, so it can be
    reweighted by usage at the call site.
    """
    total = sum(score.weighted_fidelity for score in score_events(stored, task, context))
    return total / task.weight if task.weight > 0.0 else 0.0
