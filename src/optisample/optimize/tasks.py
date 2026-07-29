from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.codec import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig, ReduceConfig, Representatives, ZoneConfig
from optisample.dsp.surrogate import StoredSample, closed_reference, render
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, QualityReport, evaluate
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.reduce.events import MergedEvent, merge_events
from optisample.optimize.reduce.keys import SampleKey, nearest_key
from optisample.optimize.velocity_map import VelocityVolumeMap
from optisample.optimize.weighting import energy_weight
from trackmod.module.storage import Storage

AudioMap = Mapping[SampleKey, Signal]


def _scored_span(reference: Signal, duration_s: float, sample_rate: int) -> Signal:
    """The stretch of a recording one note class is measured over: the recording held for the note's length."""
    return reference[: seconds_to_frames(duration_s, sample_rate)]


@dataclass(frozen=True)
class Event(MergedEvent):
    """A merged class of notes the material plays at one pitch, with its ground-truth source attached.

    Everything deciding how the class scores comes from
    :class:`~optisample.optimize.reduce.events.MergedEvent`; what the task layer adds is ``reference``,
    the recording :attr:`~optisample.optimize.reduce.events.MergedEvent.reference_key` names, resolved
    once here so a scorer reads the audio straight off the class it is scoring, and ``energy_weight``,
    what the notes' own level makes their distortion worth
    (:func:`~optisample.optimize.weighting.energy_weight`).
    """

    reference: Signal
    energy_weight: float

    @property
    def objective_weight(self) -> float:
        """This class's share of the objective's weight: the time it plays for, scaled by how loud it plays.

        The one number an allocation weighs a class's distortion by, so playing time and level reach the
        objective through a single rule and either can be read back on its own.
        """
        return self.weight * self.energy_weight

    def scored_span(self, sample_rate: int) -> Signal:
        """The stretch of the source note a reconstruction of this class is measured over.

        The class is scored over ``duration_s``, so the ground truth runs for exactly that long. One
        accessor, so the objective, the A/B pair and the shortlist auditions all measure over the same
        stretch of recording.
        """
        return _scored_span(self.reference, self.duration_s, sample_rate)

    def scored_reference(self, stored: StoredSample, sample_rate: int, *, pitch: int) -> Signal:
        """The ground truth a reconstruction of this class from ``stored``, sounded at ``pitch``, is compared to.

        The scored stretch closed the way ``stored`` closes
        (:func:`~optisample.dsp.surrogate.render.closed_reference`), so the ramp a stored span stops on
        stands on both sides of the comparison and what is left between them is the codec.
        """
        return closed_reference(self.scored_span(sample_rate), stored, sample_rate, pitch=pitch)


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
    def objective_weight(self) -> float:
        """What this pitch carries of the objective's weight, summed over the classes played here.

        The multiplier an allocation prices this pitch's distortion by, so a key the material plays
        rarely or softly asks less of the budget than one it leans on.
        """
        return sum(event.objective_weight for event in self.events)

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
    priced in the bytes the written module will actually spend on it. ``bandwidth`` holds the
    reduction's own knobs, which narrow a stored grid before that sweep runs, and ``grouping`` bounds
    what pitch-zone grouping enumerates.

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
    seed: int


@dataclass(frozen=True)
class TaskInputs:
    """The instrument-wide inputs every pitch task is built from (bundled to stay under the limit).

    ``velocity_map`` fixes the volume each note renders at and ``reduce`` how far merging widens a class,
    which together decide what one scored class stands for. ``sample_rate`` and ``energy_exponent``
    settle what that class costs: the stretch of its recording it is scored over, and how steeply that
    stretch's energy scales the distortion measured on it.
    """

    audio: AudioMap
    velocity_map: VelocityVolumeMap
    reduce: ReduceConfig
    sample_rate: int
    energy_exponent: float


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


def _scored_event(merged: MergedEvent, inputs: TaskInputs) -> Event:
    """One merged class with the audio it is judged against, and what its level makes that judgement worth."""
    reference = inputs.audio[merged.reference_key]
    scored = _scored_span(reference, merged.duration_s, inputs.sample_rate)
    return Event(
        merged.reference_key,
        merged.velocity,
        merged.volume,
        merged.duration_s,
        merged.weight,
        reference,
        energy_weight(scored, inputs.energy_exponent),
    )


def _build_pitch_task(
    pitch: int,
    events: Sequence[NoteEvent],
    available: Sequence[SampleKey],
    inputs: TaskInputs,
) -> PitchTask:
    """Assemble one pitch's task: its representative recording and the note classes scored against it.

    The notes played here collapse into the classes that reconstruct alike
    (:func:`~optisample.optimize.reduce.events.merge_events`), each attached to the recording it is
    compared against. The representative is the recording nearest the loudest velocity played here --
    the sample a zone stores when this pitch is chosen to represent it.
    """
    representative_key = nearest_key(available, max(event.velocity for event in events))
    scored = tuple(
        _scored_event(merged, inputs)
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


def build_tasks(instrument: InstrumentSpec, inputs: TaskInputs) -> list[PitchTask]:
    """Group the material by pitch and attach each pitch's representative recording and note classes.

    Returned tasks are ordered by pitch -- the order the pitch-zone partitioning DP segments over. What
    each class stands for and what it costs both come off ``inputs``, so one bundle settles the whole
    reduction a task carries.

    Raises:
        ValueError: when the material plays a pitch the recorded grid has no sample for.
    """
    by_pitch = _group_events_by_pitch(instrument.material or [])
    keys_at = _keys_by_pitch(inputs.audio)

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
        """This class's contribution to the objective: its objective weight times its distortion."""
        return self.event.objective_weight * self.report.fidelity


def score_event(stored: StoredSample, event: Event, *, pitch: int, context: EvalContext) -> QualityReport:
    """Reconstruct one note class from ``stored`` sounded at ``pitch`` and score it against its source.

    The atom every reconstruction score is built from: one render and one composite evaluation, settled
    by the stored sample, the key it sounds at and the class alone. A caller scoring some of a pitch's
    classes therefore reads exactly the numbers the whole-pitch scorer reads for those classes.
    """
    candidate = render_event(stored, event, pitch=pitch, sample_rate=context.sample_rate)
    return evaluate(
        event.scored_reference(stored, context.sample_rate, pitch=pitch),
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


def weighted_distortion(task: PitchTask, fidelity: Callable[[Event], float]) -> float:
    """``task``'s distortion per unit of objective weight, gathered from a per-class ``fidelity``.

    The rule turning per-class scores into the number an allocation compares, held apart from how a
    class is scored so a caller measuring its classes some other way -- reading a score it already took
    for one -- still weights them the way the objective does. A key the material never plays, or plays
    only in silence, scores zero.
    """
    total = sum(event.objective_weight * fidelity(event) for event in task.events)
    return total / task.objective_weight if task.objective_weight > 0.0 else 0.0


def score_reconstruction(
    stored: StoredSample,
    task: PitchTask,
    context: EvalContext,
) -> float:
    """Weighted mean distortion of reconstructing ``task``'s notes from ``stored`` (repitched to its key).

    The weighted sum of each class's own score (:func:`score_event`) normalized per unit of material
    weight, so it can be reweighted by usage at the call site.
    """
    return weighted_distortion(
        task,
        lambda event: score_event(stored, event, pitch=task.pitch, context=context).fidelity,
    )
