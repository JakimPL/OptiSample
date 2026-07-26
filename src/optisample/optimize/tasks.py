from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import Representatives
from optisample.dsp.surrogate import StoredSample, render
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, QualityReport, evaluate
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.module.storage import Storage

AudioMap = Mapping[SampleKey, Signal]


@dataclass(frozen=True)
class Event:
    """A distinct (velocity, duration) the material plays at one pitch, with its ground-truth source."""

    velocity: int
    duration_s: float
    weight: float
    reference: Signal


@dataclass(frozen=True)
class PitchTask:
    """Everything needed to score one pitch's configs: its representative recording and references.

    ``candidates`` lists the survivors at this pitch that may be stored as its sample, ``representative``
    being the recording of the first-listed one. Under the default ``nearest_loudest`` policy that is the
    single recording nearest the loudest velocity played here; ``all`` offers every survivor, so a
    timbral variant can compete for the slot.
    """

    pitch: int
    weight: float
    representative_key: SampleKey
    representative: Signal
    candidates: tuple[SampleKey, ...]
    events: tuple[Event, ...]

    @property
    def max_duration_s(self) -> float:
        """Longest note the material holds at this pitch (the storage trim before any transpose)."""
        return max(event.duration_s for event in self.events)


@dataclass(frozen=True)
class EvalContext:
    """Shared scoring inputs (bundled to stay under the argument limit).

    ``storage`` is the target format's cost table, so every operating point the sweep produces is
    priced in the bytes the written module will actually spend on it.
    """

    sample_rate: int
    velocity_map: VelocityVolumeMap
    composite: CompositeFidelity
    rng: np.random.Generator
    sweep: SweepConfig
    encode: EncodeConfig
    storage: Storage


def nearest_key(available: Sequence[SampleKey], velocity: int) -> SampleKey:
    """Recorded key whose velocity is closest to ``velocity``.

    Ties favour the louder recording, then the lowest CC bucket, so a pitch with several timbral
    variants at one velocity still resolves to the same reference on every run.
    """
    return min(available, key=lambda key: (abs(key.velocity - velocity), -key.velocity, key.cc))


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


def _keys_by_pitch(audio: AudioMap) -> dict[int, list[SampleKey]]:
    """The identities that survived dedup at each pitch (``audio``'s keys, grouped by pitch)."""
    keys_at: dict[int, list[SampleKey]] = {}
    for key in audio:
        keys_at.setdefault(key.pitch, []).append(key)

    return keys_at


def _candidate_keys(
    available: Sequence[SampleKey],
    representative_key: SampleKey,
    representatives: Representatives,
) -> tuple[SampleKey, ...]:
    """The survivors at a pitch that may be stored as its sample, the representative always first."""
    if representatives is Representatives.NEAREST_LOUDEST:
        return (representative_key,)

    return (representative_key, *(key for key in available if key != representative_key))


def _build_pitch_task(
    pitch: int,
    events: Sequence[NoteEvent],
    available: Sequence[SampleKey],
    audio: AudioMap,
    representatives: Representatives,
) -> PitchTask:
    """Assemble one pitch's task: its representative recording and per-(velocity, duration) references.

    Each distinct ``(velocity, duration)`` the material plays becomes an :class:`Event` referenced by the
    recording at the nearest available velocity; the representative is the recording nearest the loudest
    velocity played here (the sample the zone stores when this pitch is chosen representative).
    """
    representative_key = nearest_key(available, max(event.velocity for event in events))
    built = tuple(
        Event(
            merged.velocity,
            merged.duration_s,
            merged.weight,
            audio[nearest_key(available, merged.velocity)],
        )
        for merged in merge_events(events)
    )
    weight = sum(event.weight for event in built)
    return PitchTask(
        pitch,
        weight,
        representative_key,
        audio[representative_key],
        _candidate_keys(available, representative_key, representatives),
        built,
    )


def build_tasks(
    instrument: InstrumentSpec,
    audio: AudioMap,
    representatives: Representatives,
) -> list[PitchTask]:
    """Group the material by pitch and attach each pitch's representative recording and references.

    Returned tasks are ordered by pitch -- the order the pitch-zone partitioning DP segments over.
    ``representatives`` decides how many of a pitch's survivors are offered as candidate samples.

    Raises:
        ValueError: when the material plays a pitch the recorded grid has no sample for.
    """
    by_pitch = _group_events_by_pitch(instrument.material or [])
    keys_at = _keys_by_pitch(audio)

    tasks: list[PitchTask] = []
    for pitch, events in sorted(by_pitch.items()):
        available = sorted(keys_at.get(pitch, []))
        if not available:
            raise ValueError(f"instrument {instrument.id!r} has no recorded sample for pitch {pitch}")

        tasks.append(_build_pitch_task(pitch, events, available, audio, representatives))

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
