from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig, ReduceConfig, Representatives, ZoneConfig
from optisample.dsp.surrogate import StoredSample, render
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, QualityReport, evaluate
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.reduce.events import MergedEvent, merge_events
from optisample.optimize.reduce.keys import SampleKey, nearest_key
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.module.storage import Storage

AudioMap = Mapping[SampleKey, Signal]


@dataclass(frozen=True)
class Event(MergedEvent):
    """A merged class of notes the material plays at one pitch, with its ground-truth source attached.

    Everything deciding how the class scores comes from
    :class:`~optisample.optimize.reduce.events.MergedEvent`; what the task layer adds is ``reference``,
    the recording :attr:`~optisample.optimize.reduce.events.MergedEvent.reference_key` names, resolved
    once here so a scorer reads the audio straight off the class it is scoring.
    """

    reference: Signal

    def scored_reference(self, sample_rate: int) -> Signal:
        """The stretch of the source note a reconstruction of this class is compared against.

        The class is scored over ``duration_s``, so the ground truth is the recording held for exactly
        that long. One accessor, so the objective, the A/B pair and the shortlist auditions all measure
        against the same span.
        """
        return self.reference[: seconds_to_frames(self.duration_s, sample_rate)]


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

    @property
    def scored_classes(self) -> int:
        """How many classes the notes played here collapsed into, which is what one encoding is scored on."""
        return len(self.events)

    @property
    def representative_event(self) -> Event:
        """The note class worth listening to at this pitch: the most-played one, ties going to the longest.

        Every artifact that renders one note per pitch renders this one, so the A/B pair and the
        shortlist auditions are heard at the dynamic and length the material spends the most time on.
        """
        return max(self.events, key=lambda event: (event.weight, event.duration_s))


def render_event(stored: StoredSample, event: Event, *, pitch: int, sample_rate: int) -> Signal:
    """The audio ``stored`` produces for one note class: repitched to ``pitch``, at its volume and length.

    The one reconstruction the objective, the A/B pair and the shortlist auditions all listen to, so a
    fidelity score and the WAV written beside it describe the same audio.
    """
    return render(stored, sample_rate, pitch=pitch, volume=event.volume, duration_s=event.duration_s)


@dataclass(frozen=True)
class EvalContext:
    """Shared scoring inputs (bundled to stay under the argument limit).

    ``storage`` is the target format's cost table, so every operating point the sweep produces is
    priced in the bytes the written module will actually spend on it. ``bandwidth`` and ``byte_target``
    are what narrows a stored grid before that sweep runs: the reduction's own knobs, and the share of
    the sample budget one key can expect once they split it evenly, which is the scale every demand a
    zone makes of a recording is built from. ``grouping`` bounds what pitch-zone grouping enumerates.

    The two dither sources sit side by side: ``rng`` is the one stream the ungrouped sweep's encodes
    draw from in the order it reaches them, and ``seed`` is the run entropy a stored sample scored
    across zones derives its own stream from, so that score reads the same wherever it appears.
    """

    sample_rate: int
    composite: CompositeFidelity
    rng: np.random.Generator
    sweep: SweepConfig
    encode: EncodeConfig
    storage: Storage
    bandwidth: BandwidthConfig
    grouping: ZoneConfig
    byte_target: int
    seed: int


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
            merged.reference_key,
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


def score_event(stored: StoredSample, event: Event, *, pitch: int, context: EvalContext) -> QualityReport:
    """Reconstruct one note class from ``stored`` sounded at ``pitch`` and score it against its source.

    The atom every reconstruction score is built from: one render and one composite evaluation, settled
    by the stored sample, the key it sounds at and the class alone. A caller scoring some of a pitch's
    classes therefore reads exactly the numbers the whole-pitch scorer reads for those classes.
    """
    candidate = render_event(stored, event, pitch=pitch, sample_rate=context.sample_rate)
    return evaluate(
        event.scored_reference(context.sample_rate),
        candidate,
        context.sample_rate,
        context.composite,
    )


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
        yield EventScore(event=event, report=score_event(stored, event, pitch=task.pitch, context=context))


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
