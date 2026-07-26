from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import ReduceConfig, Representatives
from optisample.dsp.surrogate import StoredSample, render
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, QualityReport, evaluate
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.reduce.events import merge_events
from optisample.optimize.reduce.keys import SampleKey, nearest_key
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.module.storage import Storage

AudioMap = Mapping[SampleKey, Signal]


@dataclass(frozen=True)
class Event:
    """A class of notes the material plays at one pitch that score alike, with its ground-truth source.

    ``volume`` is the note volume every member renders at and ``duration_s`` the length they are all
    scored over, the two things the reconstruction depends on. ``velocity`` labels the class by the
    loudest note in it, and ``weight`` is their combined playing time.
    """

    velocity: int
    volume: int
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
    composite: CompositeFidelity
    rng: np.random.Generator
    sweep: SweepConfig
    encode: EncodeConfig
    storage: Storage


@dataclass(frozen=True)
class _TaskInputs:
    """The instrument-wide inputs every pitch task is built from (bundled to stay under the limit)."""

    audio: AudioMap
    velocity_map: VelocityVolumeMap
    reduce: ReduceConfig


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
    inputs: _TaskInputs,
) -> PitchTask:
    """Assemble one pitch's task: its representative recording and the note classes scored against it.

    The notes played here collapse into the classes that reconstruct alike
    (:func:`~optisample.optimize.reduce.events.merge_events`), each attached to the recording it is
    compared against. The representative is the recording nearest the loudest velocity played here --
    the sample a zone stores when this pitch is chosen to represent it.
    """
    representative_key = nearest_key(available, max(event.velocity for event in events))
    scored = tuple(
        Event(
            merged.velocity,
            merged.volume,
            merged.duration_s,
            merged.weight,
            inputs.audio[merged.reference_key],
        )
        for merged in merge_events(events, available, inputs.velocity_map, inputs.reduce.events)
    )
    return PitchTask(
        pitch,
        sum(event.weight for event in scored),
        representative_key,
        inputs.audio[representative_key],
        _candidate_keys(available, representative_key, inputs.reduce.dedupe.representatives),
        scored,
    )


def build_tasks(
    instrument: InstrumentSpec,
    audio: AudioMap,
    velocity_map: VelocityVolumeMap,
    reduce: ReduceConfig,
) -> list[PitchTask]:
    """Group the material by pitch and attach each pitch's representative recording and note classes.

    Returned tasks are ordered by pitch -- the order the pitch-zone partitioning DP segments over.
    ``velocity_map`` fixes the volume each note renders at, which is one of the two things that decide
    whether two notes score alike; ``reduce`` supplies the rest of the reduction: how far duration
    bucketing widens a class, and how many of a pitch's survivors are offered as candidate samples.

    Raises:
        ValueError: when the material plays a pitch the recorded grid has no sample for.
    """
    by_pitch = _group_events_by_pitch(instrument.material or [])
    keys_at = _keys_by_pitch(audio)
    inputs = _TaskInputs(audio=audio, velocity_map=velocity_map, reduce=reduce)

    tasks: list[PitchTask] = []
    for pitch, events in sorted(by_pitch.items()):
        available = sorted(keys_at.get(pitch, []))
        if not available:
            raise ValueError(f"instrument {instrument.id!r} has no recorded sample for pitch {pitch}")

        tasks.append(_build_pitch_task(pitch, events, available, inputs))

    return tasks


@dataclass(frozen=True)
class EventScore:
    """One note class scored: the class and the fidelity report its reconstruction earned."""

    event: Event
    report: QualityReport

    @property
    def weighted_fidelity(self) -> float:
        """This class's contribution to the objective: its usage weight times its distortion."""
        return self.event.weight * self.report.fidelity


def score_events(
    stored: StoredSample,
    task: PitchTask,
    context: EvalContext,
) -> Iterator[EventScore]:
    """Reconstruct each of ``task``'s note classes from ``stored``, one :class:`EventScore` per class.

    ``stored`` is rendered at ``task.pitch`` -- a transpose of ``task.pitch - stored.root_pitch``
    semitones, zero when ``stored`` is this pitch's own recording -- scaled by each class's volume and
    fitted to its duration, then compared to that class's source note. This is the shared scorer both the
    objective and ``metrics.json`` consume.
    """
    for event in task.events:
        candidate = render(
            stored,
            context.sample_rate,
            pitch=task.pitch,
            volume=event.volume,
            duration_s=event.duration_s,
        )
        reference = event.reference[: seconds_to_frames(event.duration_s, context.sample_rate)]
        yield EventScore(
            event=event,
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
