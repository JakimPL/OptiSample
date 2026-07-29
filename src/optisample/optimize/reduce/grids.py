from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol

from optisample.config.codec import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig
from optisample.dsp.loop import Loop, LoopQuality, loop_candidates, loop_quality
from optisample.dsp.surrogate import EncodingParams
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.base import Signal
from optisample.optimize.reduce.bandwidth import (
    ClipDemand,
    StoredFormat,
    stored_encodings,
    stored_format,
    useful_rate_hz,
)
from optisample.parallel import map_workers
from optisample.progress import ProgressSink

NARROW_LABEL: Final = "Narrowing stored grids"

_OWN_KEY: Final = 0  # a key sounding its own recording plays it at the pitch it was recorded at


class StoredClip(Protocol):
    """What narrowing reads about the sample one pitch stores: its recording and how long it is held.

    Stated as a protocol so the pre-pass measures the same pitch tasks the sweep scores, while the
    reduction subpackage stays a leaf the task layer builds on.
    """

    @property
    def pitch(self) -> int: ...

    @property
    def representative(self) -> Signal: ...

    @property
    def max_duration_s(self) -> float: ...


@dataclass(frozen=True)
class ClipRequest:
    """One pitch's stored clip as a worker receives it: the recording and the span held, alone.

    Narrowing reads a clip through :class:`StoredClip`, so a worker is handed exactly those three
    fields and the note classes scored against the clip stay in the calling process.
    """

    pitch: int
    representative: Signal
    max_duration_s: float


@dataclass(frozen=True)
class GridContext:
    """What settling a pitch's stored grid runs off, in the validated config a worker process can be handed.

    ``sample_rate`` is the rate the run measures at, ``encode`` the codec config a loop is measured under,
    ``sweep`` what a clip may be stored as, and ``bandwidth`` the knobs that read a recording's own band.
    Every field is a small validated value, so the whole context travels to a worker as it stands.
    """

    sample_rate: int
    encode: EncodeConfig
    sweep: SweepConfig
    bandwidth: BandwidthConfig


@dataclass(frozen=True)
class MeasuredLoop:
    """One loop candidate as the report states it: where it sits in the note, and how well it stands in.

    ``choice`` is the :attr:`~optisample.dsp.surrogate.EncodingParams.loop_choice` naming it. The bounds
    are seconds into the recording, so a loop is read where a listener hears it, and ``quality`` is what
    :func:`~optisample.dsp.loop.loop_quality` measures of it -- together, the case for or against the
    loop the allocation went on to buy.
    """

    choice: int
    start_s: float
    end_s: float
    quality: LoopQuality


@dataclass(frozen=True)
class NarrowedGrid:
    """What the reduction settled for the sample one pitch stores, and the band it settled it from.

    ``useful_rate_hz`` is the stored rate carrying everything the recording still contributes at its own
    key (see :func:`~optisample.optimize.reduce.bandwidth.useful_rate_hz`); ``stored`` is the format the
    ladder's lowest rung reaching it names; ``encodings`` are what the sweep then runs, that one format
    over each loop choice; and ``loops`` are the candidates those choices name, measured on the recording
    as it stands.
    """

    pitch: int
    useful_rate_hz: float
    stored: StoredFormat
    encodings: tuple[EncodingParams, ...]
    loops: tuple[MeasuredLoop, ...]


def _measured_loop(choice: int, loop: Loop, stored: Signal, context: GridContext) -> MeasuredLoop:
    """One candidate loop placed in the note by the second and measured for what storing it would cost."""
    return MeasuredLoop(
        choice=choice,
        start_s=loop.start / context.sample_rate,
        end_s=loop.end / context.sample_rate,
        quality=loop_quality(stored, loop, context.sample_rate, context.encode.loop),
    )


def _measured_loops(clip: StoredClip, context: GridContext) -> tuple[MeasuredLoop, ...]:
    """The loop candidates the sweep reaches for one clip, measured over the stretch it holds.

    Read on the recording at its own rate, which is the waveform a reader listens to and the one the
    audition folder holds. A stored copy at a reduced rate lays its candidates out on its own resampled
    waveform, where the same choice lands at the same place in the note, because resampling carries the
    material's period and the analysis window alike.
    """
    stored = clip.representative[: seconds_to_frames(clip.max_duration_s, context.sample_rate)]
    candidates = loop_candidates(stored, context.sample_rate, context.encode.loop)
    return tuple(
        _measured_loop(choice, loop, stored, context)
        for choice, loop in enumerate(candidates[: context.sweep.loop_choices])
    )


def narrow_grid(clip: StoredClip, context: GridContext) -> NarrowedGrid:
    """Settle one pitch's stored grid, at the demand a key sounding its own recording makes of it.

    Reads the clip and the context alone, so a pitch earns the same grid in whichever process and
    whichever order it is reached. That is the property :func:`narrow_grids` shares the pitches out on.

    The sample answers for its own key, so the band it is stored at is the one its recording occupies with
    nothing transposed away.
    """
    demand = ClipDemand(trim_s=clip.max_duration_s, delta_semitones=_OWN_KEY)
    stored = stored_format(clip.representative, demand, context)
    return NarrowedGrid(
        pitch=clip.pitch,
        useful_rate_hz=useful_rate_hz(clip.representative, demand, context.sample_rate, context.bandwidth),
        stored=stored,
        encodings=stored_encodings(stored, context.sweep, trim_s=demand.trim_s),
        loops=_measured_loops(clip, context),
    )


def _requested(clip: StoredClip) -> ClipRequest:
    """One clip reduced to the three fields narrowing reads, which is all a worker needs of it."""
    return ClipRequest(pitch=clip.pitch, representative=clip.representative, max_duration_s=clip.max_duration_s)


def narrow_grids(
    clips: Sequence[StoredClip],
    context: GridContext,
    *,
    workers: int,
    progress: ProgressSink,
) -> tuple[NarrowedGrid, ...]:
    """Narrow every pitch's stored grid, sharing the pitches across ``workers`` processes.

    Grids come back in ``clips`` order and hold what :func:`narrow_grid` decides for each pitch on its
    own, so a fanned-out pre-pass states the same reduction an in-process one does.
    """
    return tuple(
        map_workers(
            partial(narrow_grid, context=context),
            [_requested(clip) for clip in clips],
            workers=workers,
            label=NARROW_LABEL,
            progress=progress,
        )
    )
