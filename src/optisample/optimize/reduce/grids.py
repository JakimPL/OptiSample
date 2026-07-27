from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol

from optisample.config.dsp import EncodeConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig
from optisample.dsp.surrogate import EncodingParams
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, build_composite
from optisample.optimize.reduce.bandwidth import ClipDemand, candidate_params, useful_rate_hz
from optisample.parallel import map_workers
from optisample.progress import ProgressSink
from trackmod.module.storage import Storage

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
    """What narrowing a stored grid runs off, in the validated config a worker process can be handed.

    ``sample_rate`` is the rate the run measures at, ``storage`` the cost table each candidate is priced
    against, ``sweep`` the grid enumerated and ``byte_target`` the share of the budget one stored sample
    can expect, which is what the shortlist is drawn around.

    The metrics config travels rather than the metric built from it, so the whole context is small
    validated values and every process narrowing a grid scores with a metric assembled the one way
    :func:`~optisample.metrics.composite.build_composite` assembles it.
    """

    sample_rate: int
    metrics: MetricsConfig
    encode: EncodeConfig
    storage: Storage
    sweep: SweepConfig
    bandwidth: BandwidthConfig
    byte_target: int

    @property
    def composite(self) -> CompositeFidelity:
        """The fidelity metric a candidate encoding is scored with, assembled from ``metrics``."""
        return build_composite(self.metrics)


@dataclass(frozen=True)
class NarrowedGrid:
    """The encodings the sweep runs for the sample one pitch stores, and the band that bounds them.

    ``useful_rate_hz`` is the stored rate carrying everything the recording still contributes at its own
    key (see :func:`~optisample.optimize.reduce.bandwidth.useful_rate_hz`); ``shortlist`` is what the
    frontier around the budget's per-key share leaves of the full grid.
    """

    pitch: int
    useful_rate_hz: float
    shortlist: tuple[EncodingParams, ...]


def narrow_grid(clip: StoredClip, context: GridContext) -> NarrowedGrid:
    """Narrow one pitch's stored grid, at the demand a key sounding its own recording makes of it.

    Reads the clip and the context alone, and the proxy encodes it prices run off the surrogate's own
    fixed dither seed, so a pitch earns the same grid in whichever process and whichever order it is
    reached. That is the property :func:`narrow_grids` shares the pitches out on.

    The sample answers for its own key alone, so what it may spend is the per-key share as it stands.
    """
    demand = ClipDemand(
        trim_s=clip.max_duration_s,
        delta_semitones=_OWN_KEY,
        byte_target=context.byte_target,
    )
    return NarrowedGrid(
        pitch=clip.pitch,
        useful_rate_hz=useful_rate_hz(clip.representative, demand, context.sample_rate, context.bandwidth),
        shortlist=candidate_params(clip.representative, demand, context),
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
