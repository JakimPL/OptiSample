from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol

from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig
from optisample.dsp.surrogate import EncodingParams
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
    """What narrowing reads about the sample one pitch stores: its recording, how long it is held, its loops.

    Stated as a protocol so the pre-pass measures the same pitch tasks the sweep scores, while the
    reduction subpackage stays a leaf the task layer builds on.
    """

    @property
    def pitch(self) -> int: ...

    @property
    def representative(self) -> Signal: ...

    @property
    def max_duration_s(self) -> float: ...

    @property
    def offered_loops(self) -> int: ...


@dataclass(frozen=True)
class ClipRequest:
    """One pitch's stored clip as a worker receives it: the recording, the span held, and the loops it offers.

    Narrowing reads a clip through :class:`StoredClip`, so a worker is handed exactly those four
    fields and the note classes scored against the clip stay in the calling process.
    """

    pitch: int
    representative: Signal
    max_duration_s: float
    offered_loops: int


@dataclass(frozen=True)
class GridContext:
    """What settling a pitch's stored grid runs off, in the validated config a worker process can be handed.

    ``sample_rate`` is the rate the run measures at, ``sweep`` what a clip may be stored as, and
    ``bandwidth`` the knobs that read a recording's own band. Every field is a small validated value, so the
    whole context travels to a worker as it stands.
    """

    sample_rate: int
    sweep: SweepConfig
    bandwidth: BandwidthConfig


@dataclass(frozen=True)
class NarrowedGrid:
    """What the reduction settled for the sample one pitch stores, and the band it settled it from.

    ``useful_rate_hz`` is the stored rate carrying everything the recording still contributes at its own
    key (see :func:`~optisample.optimize.reduce.bandwidth.useful_rate_hz`); ``stored`` is the format the
    ladder's lowest rung reaching it names; and ``encodings`` are what the sweep then runs, that one format
    over each stored span the clip offers.
    """

    pitch: int
    useful_rate_hz: float
    stored: StoredFormat
    encodings: tuple[EncodingParams, ...]


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
        encodings=stored_encodings(
            stored,
            context.sweep,
            sample_rate=context.sample_rate,
            trim_s=demand.trim_s,
            loops=clip.offered_loops,
        ),
    )


def _requested(clip: StoredClip) -> ClipRequest:
    """One clip reduced to the four fields narrowing reads, which is all a worker needs of it."""
    return ClipRequest(
        pitch=clip.pitch,
        representative=clip.representative,
        max_duration_s=clip.max_duration_s,
        offered_loops=clip.offered_loops,
    )


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
