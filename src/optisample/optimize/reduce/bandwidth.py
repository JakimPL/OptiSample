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
_UNTRANSPOSED: Final = 0  # the transpose a sample plays at while it serves the key it was recorded at


@dataclass(frozen=True)
class ClipDemand:
    """What one stored sample has to serve: how long it is held, its transpose, and what it may cost.

    ``trim_s`` is the stored length the sweep trims to, which is what prices every candidate rate.
    ``delta_semitones`` is the largest upward transpose the sample plays at: zero while every key sounds
    its own recording, an octave once one representative also serves the key twelve semitones above it.
    ``byte_target`` is the share of the budget this one sample may spend, which is the price the
    shortlist is ranked around: a zone covering a dozen keys carries a dozen keys' worth of it.
    """

    trim_s: float
    delta_semitones: int
    byte_target: int


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


def _stored_span(clip: Signal, trim_s: float, sample_rate: int) -> Signal:
    """The stretch of ``clip`` a sample trimmed to ``trim_s`` keeps -- the part being priced and scored."""
    return np.asarray(clip[: seconds_to_frames(trim_s, sample_rate)], dtype=np.float64)


def _content_hz(clip: Signal, trim_s: float, sample_rate: int, config: BandwidthConfig) -> float:
    """Where the spectrum of the stretch stored at ``trim_s`` falls away -- the recording's own band.

    Read over the stretch a sample trimmed to ``trim_s`` keeps, which is the stretch the storage
    actually holds, and so a property of the recording and that length alone.
    """
    stored = _stored_span(clip, trim_s, sample_rate)
    return content_edge_hz(stored, sample_rate, config.content_floor_db, config.content_band_hz)


def _audible_rate_hz(
    content_hz: float,
    delta_semitones: int,
    sample_rate: int,
    config: BandwidthConfig,
) -> float:
    """The stored rate holding a band of ``content_hz`` played ``delta_semitones`` above its root.

    Two bounds decide it, and the tighter one wins. The recording occupies ``content_hz`` of its own;
    playback scales that content and the stored cutoff by one factor, so a rate holding the whole band
    at the root holds it at every key. Playback separately stops mattering above ``ceiling_hz``, and a
    sample played ``delta_semitones`` up lifts its stored band by that interval, leaving audible only
    what sits under the ceiling scaled back down by it. Nyquist doubles the surviving frequency into the
    rate that holds it.

    Both bounds are deliberately generous, because a rate wrongly ruled out is a rate the allocation
    never gets to buy. The ceiling is itself capped at the analysis Nyquist, the band every fidelity
    score in the run is measured over. A wider transpose lowers the result, so the untransposed rate is
    the loosest bound any demand on a stored stretch can impose.
    """
    ceiling_hz = min(config.ceiling_hz, sample_rate / _RATE_PER_BANDWIDTH)
    audible_hz = ceiling_hz * semitone_ratio(-delta_semitones)
    return _RATE_PER_BANDWIDTH * min(content_hz, audible_hz)


def useful_rate_hz(
    clip: Signal,
    demand: ClipDemand,
    sample_rate: int,
    config: BandwidthConfig,
) -> float:
    """The stored rate that carries everything ``clip`` still contributes at the keys it serves.

    Measures the band the stored stretch occupies (:func:`_content_hz`) and reads the rate holding it at
    the transpose the demand asks for (:func:`_audible_rate_hz`).
    """
    return _audible_rate_hz(
        _content_hz(clip, demand.trim_s, sample_rate, config),
        demand.delta_semitones,
        sample_rate,
        config,
    )


def _priced_rates(rates: Sequence[int], useful_rate: float) -> list[int]:
    """Rates worth pricing: every one within ``useful_rate``, plus the cheapest one beyond it.

    A rate past the bound spends its extra bytes on a band that plays back inaudibly, so one of them is
    enough. Keeping the cheapest leaves a budget with room to spare the option of buying headroom above
    a bound that was, after all, measured.
    """
    within = [rate for rate in rates if rate <= useful_rate]
    beyond = [rate for rate in rates if rate > useful_rate]
    return within + ([min(beyond)] if beyond else [])


@dataclass(frozen=True)
class ProxyGrid:
    """One recording priced over the stored grid at one held length, ahead of any demand made on it.

    Pricing a grid entry stores the clip and plays it back at its own root, so what it costs and how much
    of the recording survives it follow from the recording and the length held. What a zone then asks of
    that sample arrives afterwards and enters as arithmetic over these points: the transpose it plays at
    rules out the rates that stop being audible, and the keys it answers for set the share of the budget
    the frontier is ranked around. So every demand on the same recording held the same length is
    answered from one grid.

    ``entries`` is the full grid in the sweep's own enumeration order, ``content_hz`` the band the stored
    stretch occupies, and ``points`` each entry the loosest bandwidth bound admits, priced and scored. A
    ``candidates`` count reaching the whole grid settles the shortlist as ``entries`` itself, which is
    known before anything is priced, so ``points`` stays empty.
    """

    entries: tuple[EncodingParams, ...]
    content_hz: float
    points: tuple[OperatingPoint, ...]


def _narrows(entries: Sequence[EncodingParams], config: BandwidthConfig) -> bool:
    """Whether a shortlist comes out shorter than the grid, which is when pricing the grid buys anything."""
    return config.candidates < len(entries)


def _rates_within(entries: Sequence[EncodingParams], useful_rate: float) -> set[int]:
    """The stored rates among ``entries`` worth pricing under ``useful_rate``."""
    return set(_priced_rates(sorted({params.target_rate for params in entries}), useful_rate))


def proxy_grid(clip: Signal, trim_s: float, context: SweepInputs) -> ProxyGrid:
    """Price and score ``clip`` held for ``trim_s`` across every grid entry a demand on it can reach.

    This is the cheap stand-in for the real sweep, and the whole of what narrowing measures: it asks what
    one encoding costs and how much of the recording survives it, once per grid entry, where scoring
    properly costs that much for every note the material plays. Bytes come from the encoder itself, so a
    shortlist drawn from these points is priced in the same currency the budget is. The dither runs off
    the surrogate's own fixed seed, so a clip earns the same grid however many other clips were priced
    before it.

    Entries are admitted under the bound an untransposed sample meets, which is the loosest one any
    demand imposes -- transposing a sample up only lowers the rate that stays audible. Every rate a zone
    may ask for is therefore already priced here, and :func:`narrowed_params` reaches each zone's own
    shortlist by filtering these points.
    """
    entries = tuple(sweep_param_grid(context.sweep, context.sample_rate, trim_s=trim_s))
    content_hz = _content_hz(clip, trim_s, context.sample_rate, context.bandwidth)
    if not _narrows(entries, context.bandwidth):
        return ProxyGrid(entries=entries, content_hz=content_hz, points=())

    widest = _audible_rate_hz(content_hz, _UNTRANSPOSED, context.sample_rate, context.bandwidth)
    rates = _rates_within(entries, widest)
    source = SourceClip(
        signal=clip,
        sample_rate=context.sample_rate,
        root_pitch=_PROXY_ROOT_PITCH,
        duration_s=trim_s,
    )
    sweep_context = SweepContext(composite=context.composite, encode=context.encode, storage=context.storage)
    return ProxyGrid(
        entries=entries,
        content_hz=content_hz,
        points=tuple(
            evaluate_encoding(source, params, sweep_context) for params in entries if params.target_rate in rates
        ),
    )


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


def narrowed_params(grid: ProxyGrid, demand: ClipDemand, context: SweepInputs) -> tuple[EncodingParams, ...]:
    """What ``demand`` leaves of an already-priced grid, in the sweep's own enumeration order.

    The demand enters here and nowhere else. The interval the sample is transposed by rules out the rates
    whose extra band plays back inaudibly, and the share of the budget it carries ranks the frontier;
    both read the points :func:`proxy_grid` already measured. So the second zone to ask the same
    recording for the same stored length is answered by arithmetic alone.

    A ``candidates`` count reaching the whole grid returns the grid as it stands, which is the setting
    that reproduces an unnarrowed run.
    """
    if not _narrows(grid.entries, context.bandwidth):
        return grid.entries

    useful_rate = _audible_rate_hz(
        grid.content_hz,
        demand.delta_semitones,
        context.sample_rate,
        context.bandwidth,
    )
    rates = _rates_within(grid.entries, useful_rate)
    kept = _shortlist(
        [point for point in grid.points if point.params.target_rate in rates],
        context.bandwidth,
        demand.byte_target,
    )
    return tuple(params for params in grid.entries if params in kept)


def candidate_params(clip: Signal, demand: ClipDemand, context: SweepInputs) -> tuple[EncodingParams, ...]:
    """The encodings worth sweeping for one stored sample, in the sweep's own enumeration order.

    Scoring an encoding properly costs a render and a composite evaluation for every note the material
    plays, which the full grid affords only on a small instrument. This narrows the grid first: the clip
    held for the demand's stored length is priced across the grid (:func:`proxy_grid`), and the demand
    then takes the ``candidates`` frontier vertices nearest the share of the budget it carries
    (:func:`narrowed_params`). The two run together for a caller with one demand per clip; a caller
    making several demands of one recording prices it once and narrows it repeatedly.
    """
    return narrowed_params(proxy_grid(clip, demand.trim_s, context), demand, context)
