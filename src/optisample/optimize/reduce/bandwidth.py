from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol

import numpy as np

from optisample.config.optimize import SweepConfig
from optisample.config.reduce import BandwidthConfig
from optisample.dsp.spectral import content_edge_hz
from optisample.dsp.surrogate import UNLOOPED, EncodingParams, Signal
from optisample.dsp.timebase import seconds_to_frames
from optisample.music import semitone_ratio
from optisample.optimize.operating_points import compresses, sweep_rates

_RATE_PER_BANDWIDTH: Final = 2.0  # Nyquist: a stored rate carries content up to half of it
_UNTRANSPOSED: Final = 0  # the transpose a sample plays at while it serves the key it was recorded at


@dataclass(frozen=True)
class ClipDemand:
    """What one stored sample has to serve: how long it is held, and the transpose it plays at.

    ``trim_s`` is the stored length the sweep trims to, which is the stretch the band is read over.
    ``delta_semitones`` is the largest upward transpose the sample plays at: zero while every key sounds
    its own recording, an octave once one representative also serves the key twelve semitones above it.
    """

    trim_s: float
    delta_semitones: int


class FormatInputs(Protocol):
    """What settling a stored format reads from the run's shared context: the run's rate and its rules.

    Stated as a protocol so the reduction and the sweep run off one context: the rate the run measures
    at, the ladder a stored rate is chosen from and the band knobs that choose the rung each have a
    single source, and the format comes out in exactly the terms the sweep then encodes under.
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def sweep(self) -> SweepConfig: ...

    @property
    def bandwidth(self) -> BandwidthConfig: ...


def _stored_span(clip: Signal, trim_s: float, sample_rate: int) -> Signal:
    """The stretch of ``clip`` a sample trimmed to ``trim_s`` keeps -- the part being stored and scored."""
    return np.asarray(clip[: seconds_to_frames(trim_s, sample_rate)], dtype=np.float64)


def clip_band_hz(clip: Signal, trim_s: float, sample_rate: int, config: BandwidthConfig) -> float:
    """Where the spectrum of the stretch stored at ``trim_s`` falls away -- the recording's own band.

    Read over the stretch a sample trimmed to ``trim_s`` keeps, which is the stretch the storage actually
    holds, and so a property of the recording and that length alone. Every demand made on the same
    recording held the same length reads the same band, so measuring it once answers all of them.
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

    Both bounds are deliberately generous, because a rate wrongly ruled out is a rate the run never gets
    to store. The ceiling is itself capped at the analysis Nyquist, the band every fidelity score in the
    run is measured over. A wider transpose lowers the result, so the untransposed rate is the loosest
    bound any demand on a stored stretch can impose.
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

    Measures the band the stored stretch occupies (:func:`clip_band_hz`) and reads the rate holding it at
    the transpose the demand asks for (:func:`_audible_rate_hz`).
    """
    return _audible_rate_hz(
        clip_band_hz(clip, demand.trim_s, sample_rate, config),
        demand.delta_semitones,
        sample_rate,
        config,
    )


@dataclass(frozen=True)
class StoredFormat:
    """The rate, depth and compression one sample is stored at, settled from the recording's own content.

    The reduction stage decides this and the allocation spends its bytes elsewhere -- on zone width,
    sample count, stored length and which loop -- so a plan pressed for room stores fewer, wider or
    shorter samples while each one it does store carries the band its recording asked for.
    """

    target_rate: int
    depth_bits: int
    compress: bool


def _lowest_rung(rates: Sequence[int], useful_rate: float) -> int:
    """The lowest rung of ``rates`` reaching ``useful_rate``, which is the cheapest rate carrying the band.

    Rounding up keeps the stored band as wide as the recording's own, so every frequency the material still
    plays survives storage. The clip's own rate is the ladder's top rung and every bound lands at or under
    it (:func:`_audible_rate_hz` caps at the analysis Nyquist), so a rung always answers.
    """
    return min(rate for rate in rates if rate >= useful_rate)


def format_from_band(content_hz: float, demand: ClipDemand, context: FormatInputs) -> StoredFormat:
    """The format a recording of band ``content_hz`` is stored at under ``demand``, by arithmetic alone.

    The demand enters here and nowhere else: the interval the sample is transposed by lowers the rate
    that stays audible, and the ladder's lowest rung reaching what remains is the rate the sample is kept
    at. So the second zone to ask the same recording for the same stored length is answered from the band
    already measured.
    """
    useful_rate = _audible_rate_hz(content_hz, demand.delta_semitones, context.sample_rate, context.bandwidth)
    depth_bits = context.sweep.depth
    return StoredFormat(
        target_rate=_lowest_rung(sweep_rates(context.sweep, context.sample_rate), useful_rate),
        depth_bits=depth_bits,
        compress=compresses(context.sweep, depth_bits),
    )


def stored_format(clip: Signal, demand: ClipDemand, context: FormatInputs) -> StoredFormat:
    """The format ``clip`` is stored at to serve ``demand``, read off the recording's own content.

    Measures the band the stored stretch occupies (:func:`clip_band_hz`) and settles the format around it
    (:func:`format_from_band`). A caller making several demands of one recording measures the band once
    and settles each demand's format from it.
    """
    return format_from_band(
        clip_band_hz(clip, demand.trim_s, context.sample_rate, context.bandwidth),
        demand,
        context,
    )


def stored_encodings(
    stored: StoredFormat,
    sweep: SweepConfig,
    *,
    trim_s: float | None,
    loops: int,
) -> tuple[EncodingParams, ...]:
    """Every encoding the sweep runs for a sample kept at ``stored``: the trimmed span, then each loop.

    The format is settled before the sweep starts and the loops are settled before it too, so what the
    sweep prices is how far the sample carries on past its attack: keeping the played span, against keeping
    the attack plus each of the ``loops`` regions the loop stage found. Those regions run from short to
    long, so the encodings after the first walk a sample's stored length from its cheapest to its most
    faithful and the hull picks the trades worth keeping. The trimmed span leads however many loops there
    are, so it sits at the same index for every clip and the per-pitch and per-zone sweeps score in one
    order.
    """
    trimmed = EncodingParams(
        target_rate=stored.target_rate,
        depth_bits=stored.depth_bits,
        trim_s=trim_s,
        dither=sweep.dither,
        noise_shaping=sweep.noise_shaping,
        loop_index=UNLOOPED,
        compress=stored.compress,
    )
    return (trimmed, *(replace(trimmed, loop_index=index) for index in range(loops)))
