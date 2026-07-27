from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig
from optisample.dsp.spectral import content_edge_hz
from optisample.dsp.surrogate import EncodingParams, Signal
from optisample.dsp.timebase import seconds_to_frames
from optisample.metrics.composite import CompositeFidelity
from optisample.music import semitone_ratio
from optisample.optimize.operating_points import (
    OperatingPoint,
    SourceClip,
    SweepContext,
    evaluate_encoding,
    lower_convex_hull,
    sweep_param_grid,
)
from trackmod.module.storage import Storage

_RATE_PER_BANDWIDTH: Final = 2.0  # Nyquist: a stored rate carries content up to half of it
_PROXY_ROOT_PITCH: Final = 0  # the proxy plays the clip back at its own root, so the pitch naming it cancels


@dataclass(frozen=True)
class ClipDemand:
    """What one stored sample has to serve: the keys it covers, how long it is held, and its transpose.

    ``trim_s`` is the stored length the sweep trims to, which is what prices every candidate rate.
    ``delta_semitones`` is the largest upward transpose the sample plays at: zero while every key sounds
    its own recording, an octave once one representative also serves the key twelve semitones above it.
    ``key_count`` is how many keys the one sample stands for, which sets the share of the budget it may
    spend: a zone covering a dozen keys can afford a dozen times what a single key can.
    """

    trim_s: float
    delta_semitones: int
    key_count: int


class SweepInputs(Protocol):
    """What narrowing a stored grid reads from the run's shared scoring context.

    Stated as a protocol so the pre-pass and the full sweep run off one context: the rate the run
    measures at, the metric it scores with, the table it prices against and the grid it enumerates each
    have a single source, and the shortlist comes out in exactly the terms the allocation then uses.
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def composite(self) -> CompositeFidelity: ...

    @property
    def encode(self) -> EncodeConfig: ...

    @property
    def storage(self) -> Storage: ...

    @property
    def sweep(self) -> SweepConfig: ...

    @property
    def bandwidth(self) -> BandwidthConfig: ...

    @property
    def byte_target(self) -> int: ...


def _stored_span(clip: Signal, trim_s: float, sample_rate: int) -> Signal:
    """The stretch of ``clip`` a sample trimmed to ``trim_s`` keeps -- the part being priced and scored."""
    return np.asarray(clip[: seconds_to_frames(trim_s, sample_rate)], dtype=np.float64)


def useful_rate_hz(
    clip: Signal,
    demand: ClipDemand,
    sample_rate: int,
    config: BandwidthConfig,
) -> float:
    """The stored rate that carries everything ``clip`` still contributes at the keys it serves.

    Two bounds decide it, and the tighter one wins. The recording occupies a band of its own, ending
    where its spectrum falls away (:func:`~optisample.dsp.spectral.content_edge_hz`); playback scales
    that content and the stored cutoff by one factor, so a rate holding the whole band at the root holds
    it at every key. Playback separately stops mattering above ``ceiling_hz``, and a sample played
    ``delta_semitones`` up lifts its stored band by that interval, leaving audible only what sits under
    the ceiling scaled back down by it. Nyquist doubles the surviving frequency into the rate that
    holds it.

    The band is read over the stretch a sample trimmed to ``demand.trim_s`` keeps, which is the stretch
    the storage actually holds. Both bounds are deliberately generous, because a rate wrongly ruled out
    is a rate the allocation never gets to buy. The ceiling is itself capped at the analysis Nyquist,
    the band every fidelity score in the run is measured over.
    """
    stored = _stored_span(clip, demand.trim_s, sample_rate)
    ceiling_hz = min(config.ceiling_hz, sample_rate / _RATE_PER_BANDWIDTH)
    audible_hz = ceiling_hz * semitone_ratio(-demand.delta_semitones)
    content_hz = content_edge_hz(stored, sample_rate, config.content_floor_db, config.content_band_hz)
    return _RATE_PER_BANDWIDTH * min(content_hz, audible_hz)


def _priced_rates(rates: Sequence[int], useful_rate: float) -> list[int]:
    """Rates worth pricing: every one within ``useful_rate``, plus the cheapest one beyond it.

    A rate past the bound spends its extra bytes on a band that plays back inaudibly, so one of them is
    enough. Keeping the cheapest leaves a budget with room to spare the option of buying headroom above
    a bound that was, after all, measured.
    """
    within = [rate for rate in rates if rate <= useful_rate]
    beyond = [rate for rate in rates if rate > useful_rate]
    return within + ([min(beyond)] if beyond else [])


def _proxy_points(
    clip: Signal,
    demand: ClipDemand,
    grid: Sequence[EncodingParams],
    context: SweepInputs,
) -> list[OperatingPoint]:
    """Price and score each grid encoding by storing ``clip`` and playing it back at its own root.

    This is the cheap stand-in for the real sweep: it asks what one encoding costs and how much of the
    recording survives it, once per grid point, where scoring properly costs that much for every note
    the material plays. Bytes come from the encoder itself, so a shortlist drawn here is priced in the
    same currency the budget is. The dither runs off the surrogate's own fixed seed, so a clip earns the
    same shortlist however many other clips were narrowed before it.
    """
    useful_rate = useful_rate_hz(clip, demand, context.sample_rate, context.bandwidth)
    rates = set(_priced_rates(sorted({params.target_rate for params in grid}), useful_rate))
    source = SourceClip(
        signal=clip,
        sample_rate=context.sample_rate,
        root_pitch=_PROXY_ROOT_PITCH,
        duration_s=demand.trim_s,
    )
    sweep_context = SweepContext(composite=context.composite, encode=context.encode, storage=context.storage)
    return [evaluate_encoding(source, params, sweep_context) for params in grid if params.target_rate in rates]


def _shortlist(
    points: Sequence[OperatingPoint],
    config: BandwidthConfig,
    byte_target: int,
) -> set[EncodingParams]:
    """The ``candidates`` frontier vertices priced closest to what one stored sample can afford.

    Every hull vertex is the encoding some byte price makes optimal, so the ones priced around the
    budget's share of the keys it serves are the ones the allocation has a real chance of choosing;
    spending the sweep there is what the shortlist buys. Vertices sit at distinct byte costs, so price
    alone settles the ranking and the same clip always earns the same shortlist.
    """
    ranked = sorted(
        lower_convex_hull(points),
        key=lambda point: (abs(point.stored_bytes - byte_target), point.stored_bytes),
    )
    return {point.params for point in ranked[: config.candidates]}


def candidate_params(clip: Signal, demand: ClipDemand, context: SweepInputs) -> tuple[EncodingParams, ...]:
    """The encodings worth sweeping for one stored sample, in the sweep's own enumeration order.

    Scoring an encoding properly costs a render and a composite evaluation for every note the material
    plays, which the full grid affords only on a small instrument. This narrows the grid first. The
    recording's own band and the interval it is transposed by bound which rates buy anything at all
    (see :func:`useful_rate_hz`), and what survives that is priced and scored once apiece against the
    clip alone; the ``candidates`` vertices of the resulting frontier lying nearest what the keys it
    serves can afford are what the real sweep then runs.

    A ``candidates`` count reaching the whole grid returns the grid as it stands, which is the setting
    that reproduces an unnarrowed run.
    """
    grid = tuple(sweep_param_grid(context.sweep, context.sample_rate, trim_s=demand.trim_s))
    if context.bandwidth.candidates >= len(grid):
        return grid

    kept = _shortlist(
        _proxy_points(clip, demand, grid, context),
        context.bandwidth,
        context.byte_target * demand.key_count,
    )
    return tuple(params for params in grid if params in kept)
