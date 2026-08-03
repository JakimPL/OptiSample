from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.dsp.surrogate import EncodingParams
from optisample.keys import SampleKey
from optisample.metrics.base import Signal
from optisample.metrics.preprocess import integrated_loudness
from optisample.optimize.tasks import (
    EvalContext,
    Event,
    PitchTask,
    audition_sample,
    reconstruct,
    render_event,
)

_SILENT_LUFS: Final = -120.0  # the level a rendition holding digital silence is stated at


@dataclass(frozen=True)
class Rendition:
    """One encoding of one note class: what it stores, how loudly it plays, and what the composite made of it.

    The audio itself is left to be rebuilt on demand (:func:`rendered`), which is what lets a whole
    instrument's grid be priced in one pass and only the handful of encodings a listener is asked about
    reach memory as waveforms. ``loudness_lufs`` is kept from that pass because level is the confound a
    blinded preference is most exposed to and the composite quotients it out, so nothing downstream of the
    metric would otherwise know what a listener was hearing.
    """

    params: EncodingParams
    stored_bytes: int
    distortion: float
    loudness_lufs: float


@dataclass(frozen=True)
class ClipRenditions:
    """One note class priced under every encoding a listening set offers it, beside the run that rebuilds it.

    ``task`` and ``event`` are what each member is rendered from, so a set holds its own means of
    reproducing any encoding it names -- the dither runs off the surrogate's fixed seed, so a member
    rebuilt later is the very audio its distortion was read on.
    """

    task: PitchTask
    event: Event
    renditions: tuple[Rendition, ...]

    @property
    def pitch(self) -> int:
        """The key the class is played at, which is the pitch its representative was recorded at."""
        return self.task.pitch

    @property
    def key(self) -> SampleKey:
        """The recording stored as this pitch's sample, which every member of the set encodes."""
        return self.task.representative_key

    @property
    def velocity(self) -> int:
        """The dynamic the class stands for, which is the loudest note merged into it."""
        return self.event.velocity


def rendered(clip: ClipRenditions, rendition: Rendition, context: EvalContext) -> Signal:
    """The audio ``rendition`` produces for ``clip``, rebuilt from the encoding it names."""
    stored = audition_sample(clip.task, rendition.params, context)
    return render_event(stored, clip.event, pitch=clip.pitch, sample_rate=context.sample_rate)


def reference(clip: ClipRenditions, context: EvalContext) -> Signal:
    """The recording ``clip``'s encodings are judged against: the source held for the length it is scored over.

    The composite reads this same stretch closed the way each member closes it
    (:meth:`~optisample.optimize.tasks.Event.scored_reference`), so a stored span stopping early costs a
    listener what the ramp standing on both sides of the comparison spares the metric.
    """
    return clip.event.scored_span(context.sample_rate)


def _loudness(audio: Signal, sample_rate: int) -> float:
    """How loudly one rendition plays, held at :data:`_SILENT_LUFS` where it holds silence."""
    return max(integrated_loudness(audio, sample_rate), _SILENT_LUFS)


def _rendition(task: PitchTask, event: Event, params: EncodingParams, context: EvalContext) -> Rendition:
    """``task``'s class rebuilt under one encoding, priced in the bytes a written module spends on it."""
    stored = audition_sample(task, params, context)
    rebuilt = reconstruct(stored, event, pitch=task.pitch, context=context)
    return Rendition(
        params=params,
        stored_bytes=context.storage.sample_bytes(frames=stored.frames, depth=stored.depth),
        distortion=rebuilt.report.fidelity,
        loudness_lufs=_loudness(rebuilt.audio, context.sample_rate),
    )


def clip_renditions(
    task: PitchTask,
    encodings: Sequence[EncodingParams],
    context: EvalContext,
) -> ClipRenditions:
    """Price one pitch's representative note under every encoding a listening set asks about.

    The class priced is the one the material spends the most time on
    (:attr:`~optisample.optimize.tasks.PitchTask.representative_event`), which is the class the
    reduction's auditions and the plan's A/B pairs both already sound, so a pitch is heard at one
    dynamic and one length throughout the artifacts.
    """
    event = task.representative_event
    return ClipRenditions(
        task=task,
        event=event,
        renditions=tuple(_rendition(task, event, params, context) for params in encodings),
    )
