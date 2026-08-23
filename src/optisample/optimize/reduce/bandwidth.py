from collections.abc import Sequence
from dataclasses import dataclass, field, replace
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


def _stored_span(clip: Signal, trim_s: float | None, sample_rate: int) -> Signal:
    """The stretch of ``clip`` a sample trimmed to ``trim_s`` keeps -- the part being stored and scored.

    A trim of ``None`` keeps the recording whole, which is the span an untrimmed encoding stores.
    """
    if trim_s is None:
        return np.asarray(clip, dtype=np.float64)

    return np.asarray(clip[: seconds_to_frames(trim_s, sample_rate)], dtype=np.float64)


def clip_band_hz(clip: Signal, trim_s: float, sample_rate: int, config: BandwidthConfig) -> float:
    """Where the spectrum of the stretch stored at ``trim_s`` falls away -- the recording's own band.

    Read over the stretch a sample trimmed to ``trim_s`` keeps, which is the stretch the storage actually
    holds, and so a property of the recording and that length alone. Every demand made on the same
    recording held the same length reads the same band, so measuring it once answers all of them.
    """
    stored = _stored_span(clip, trim_s, sample_rate)
    return content_edge_hz(stored, sample_rate, config.content_floor_db, config.content_band_hz)


def clip_reach_hz(clip: Signal, trim_s: float | None, sample_rate: int, config: BandwidthConfig) -> float:
    """How far up the spectrum the stretch stored at ``trim_s`` still reaches, read at the deeper floor.

    The same measurement :func:`clip_band_hz` makes, taken at ``discard_floor_db``, which counts content
    quiet enough that the settled rung is free to leave it out. What sits between the two readings is the
    band a stored sample gives up, and :func:`discard_surcharge` is what prices it.
    """
    stored = _stored_span(clip, trim_s, sample_rate)
    return content_edge_hz(stored, sample_rate, config.discard_floor_db, config.content_band_hz)


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
    """The rate one sample is stored at, settled from the recording's own content, and the depths offered.

    The reduction settles the rate from the recording's own band, so every sample a plan stores carries
    the spectrum its material asked for whatever else the budget presses on. ``depths`` are the grids the
    sweep then prices that rate at, deepest first: a shallow copy costs half the frames of a deep one, so
    how finely a waveform is stored is a trade the objective makes against zone width, sample count and
    stored length rather than a decision taken before it.
    """

    target_rate: int
    depths: tuple[int, ...]


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
    return StoredFormat(
        target_rate=_lowest_rung(sweep_rates(context.sweep, context.sample_rate), useful_rate),
        depths=context.sweep.depths,
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


def discarded_octaves(reach_hz: float, target_rate: int) -> float:
    """Octaves of the band a recording reaches that a sample stored at ``target_rate`` leaves out.

    Counted in octaves so the charge follows the span of spectrum given up rather than its energy: a
    harmonic recording keeps nearly all of its energy in the first few partials, which leaves a share of
    energy reading almost the same for a rung carrying the whole spectrum as for one carrying a fraction
    of it. Octaves separate those two, and separate a bass whose material genuinely ends low -- and so
    gives up nothing at a cheap rung -- from a brass whose does not.
    """
    stored_edge_hz = target_rate / _RATE_PER_BANDWIDTH
    if reach_hz <= stored_edge_hz:
        return 0.0

    return float(np.log2(reach_hz / stored_edge_hz))


def discard_surcharge(reach_hz: float, target_rate: int, config: BandwidthConfig) -> float:
    """What storing a recording reaching ``reach_hz`` at ``target_rate`` costs the objective beyond its metrics.

    The composite metrics read a narrowed sample as close to its reference, so the bytes a wider rung
    costs buy little the objective can see. This is where a run states the worth of that band by ear:
    ``discard_penalty`` per octave given up (:func:`discarded_octaves`), added to the distortion the
    metrics measured, so the allocation weighs a wider rung against everything else the same bytes buy.
    """
    return config.discard_penalty * discarded_octaves(reach_hz, target_rate)


@dataclass
class DiscardPricer:
    """What each stored span of one recording gives up in band, measured once per length asked about.

    A recording's reach depends on the stretch stored of it and nothing else, so a sweep offering many
    encodings of one recording reads each distinct length once and prices every rung offered from it.
    Both the per-pitch sweep and the per-zone one charge through here, so a rung costs the objective the
    same wherever it is scored.
    """

    clip: Signal
    sample_rate: int
    config: BandwidthConfig
    reaches: dict[float | None, float] = field(default_factory=dict, init=False)

    def surcharge(self, params: EncodingParams) -> float:
        """What the objective is charged for the band ``params`` leaves out of this recording.

        A penalty of zero prices every rung on the metrics alone, which is answered before any spectrum
        is read so a run that states no worth by ear pays nothing to ask.
        """
        if self.config.discard_penalty == 0.0:
            return 0.0

        if params.trim_s not in self.reaches:
            self.reaches[params.trim_s] = clip_reach_hz(self.clip, params.trim_s, self.sample_rate, self.config)

        return discard_surcharge(self.reaches[params.trim_s], params.target_rate, self.config)


def _offered_rates(settled_rate: int, sweep: SweepConfig, sample_rate: int) -> tuple[int, ...]:
    """``settled_rate`` and the ``rate_headroom`` rungs of the sweep's own ladder nearest above it."""
    above = [rate for rate in reversed(sweep_rates(sweep, sample_rate)) if rate > settled_rate]
    return (settled_rate, *above[: sweep.rate_headroom])


def _dynamics(sweep: SweepConfig, depth: int) -> tuple[bool, ...]:
    """Whether a span stored at ``depth`` is offered both plain and compressed, or plain alone."""
    return (False, True) if compresses(sweep, depth) else (False,)


def stored_encodings(
    stored: StoredFormat,
    sweep: SweepConfig,
    *,
    sample_rate: int,
    trim_s: float | None,
    loops: int,
) -> tuple[EncodingParams, ...]:
    """Every encoding the sweep runs for a sample kept at ``stored``: the trimmed span, then each loop.

    The loops are settled before the sweep starts, so what the sweep prices is how far the sample carries
    on past its attack -- keeping the played span, against keeping the attack plus each of the ``loops``
    regions the loop stage found -- and how much of its spectrum it keeps. Those regions run from short to
    long, so the encodings walk a sample's stored length from its cheapest to its most faithful and the
    hull picks the trades worth keeping.

    ``stored`` names the rung the recording's own content asks for, and each span is offered there and at
    the rungs above it the headroom reaches (:func:`_offered_rates`). That is what lets a byte buy band:
    the wider rungs cost more and win where :func:`discard_surcharge` prices the spectrum they keep above
    what the same bytes buy in length or in another sample.

    Each of those is offered at every depth the sweep names, and a depth shallow enough for the dynamics
    stage to buy headroom is offered both plain and compressed, which is what makes both the grid a
    waveform is stored on and the compression ahead of it axes the run prices rather than steps it takes
    on the way past. A depth deep enough to carry the material outright offers each span once.

    Spans lead, then rates, then depths: every encoding of the trimmed span stands before the first loop's,
    so the trimmed span occupies the opening positions for every clip and the per-pitch and per-zone sweeps
    score in one order, with the settled rung and the deepest grid leading each span.
    """
    plain = EncodingParams(
        target_rate=stored.target_rate,
        depth_bits=stored.depths[0],
        trim_s=trim_s,
        dither=sweep.dither,
        noise_shaping=sweep.noise_shaping,
        loop_index=UNLOOPED,
        compress=False,
    )
    return tuple(
        replace(plain, loop_index=span, target_rate=rate, depth_bits=depth, compress=compress)
        for span in (UNLOOPED, *range(loops))
        for rate in _offered_rates(stored.target_rate, sweep, sample_rate)
        for depth in stored.depths
        for compress in _dynamics(sweep, depth)
    )
