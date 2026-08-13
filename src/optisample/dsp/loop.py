from dataclasses import dataclass
from enum import StrEnum, unique
from math import ceil, floor
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.loop import GeometryConfig, LoopConfig, PhaseConfig, SeamConfig
from optisample.config.spectral import StftParams
from optisample.dsp.envelope import LevelReading, local_level_over
from optisample.dsp.level import gain_to_db
from optisample.dsp.loopability import FrontierBounds, LoopReach, holds_a_round, loop_frontier
from optisample.dsp.resample import resampled_frame_count
from optisample.dsp.series import autocorrelation, refined_lag
from optisample.dsp.similarity import FrameSeries, frame_series, settling_frame
from optisample.dsp.spectral import split_bands, stft_magnitude
from optisample.dsp.timebase import seconds_to_frames
from optisample.music import semitone_ratio

Signal = NDArray[np.float64]

_MIN_STEADY_FRAMES: Final = 8
_MIN_LOOP_FRAMES: Final = 2  # frames a wrap needs to name two distinct ends
_SEAM_WINDOW_SHARE: Final = 100  # share of a loop read on each side of the wrap as the motion it lands in
_QUALITY_FFT: Final = 1024  # window the loop region and the stretch it stands for are compared over
_QUALITY_HOP: Final = 512
_AMPLITUDE_DB: Final = 20.0  # decibels per decade of amplitude
_SPECTRUM_FLOOR: Final = 1e-10
_STEP_FLOOR: Final = 1e-12
_CORRELATION_FLOOR: Final = 1e-24  # product of two norms a silent stretch reaches, which correlates with nothing
_MAX_LEVEL_GAIN: Final = 4.0  # +12 dB, the most holding a region at one level asks of the material
_MAX_SEAM_GAIN: Final = 2.0  # +6 dB, the most holding a blend at one level asks of material that cancels


@dataclass(frozen=True)
class Loop:
    """A forward loop over the half-open frame range ``[start, end)`` (playback wraps ``end`` -> ``start``)."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@unique
class Material(StrEnum):
    """What a recording holds too little of for any loop to be placed in it.

    Each value names the reading that came up short, so a recording stored over the span it plays says
    which knob would have to move for it to offer a loop at all. ``STEADY`` and ``ROUND`` are both length:
    the recording settles too late or ends too soon to hold one round of
    :func:`shortest_loop_frames`, which is what a floor on how long a note must sound answers for.
    ``PERIOD`` is material whose pitch reads too loosely to wrap, ``MOVEMENT`` material whose every round
    lands on a sound the recording has left behind, and ``PHASE`` a round the geometry named that the
    waveform had no room to land on.
    """

    STEADY = "steady"
    PERIOD = "period"
    ROUND = "round"
    MOVEMENT = "movement"
    PHASE = "phase"


@dataclass(frozen=True)
class LoopSearch:
    """Every loop worth offering for one recording, and what stood in the way where there are none.

    ``lacking`` is stated exactly where ``candidates`` is empty, so a settlement reports the reason a
    recording offers nothing rather than leaving a caller to infer it from an absence.
    """

    candidates: tuple[Loop, ...]
    lacking: Material | None


@dataclass(frozen=True)
class LoopQuality:
    """How well a loop stands in for the material it replaces: the wrap it makes, and the timbre it holds.

    ``seam_step`` reads the jump at the wrap in units of the loop's own typical frame-to-frame motion, so
    1.0 is a wrap as smooth as the waveform already moves and a large value is the click heard once per
    round. ``level_drift_db`` reads how far the region's own level falls across it, which is the gain
    holding it at one level asks of the material. ``spectral_distance`` is the log-spectral distance in
    decibels between the loop region and the stretch it plays in place of, measured over spectra normalized
    to unit sum so it reads the timbre a loop holds on to while the material moved on.
    """

    seam_step: float
    level_drift_db: float
    spectral_distance: float


@dataclass(frozen=True)
class _SteadyRegion:
    """The stretch a loop may be placed in, the period its material repeats at, and how it reads as timbre.

    ``series`` is the recording read as timbre frames, which is both where the window's opening was found
    and what the rounds inside it are looked for over, so one reading serves the whole placement. The two
    radii a round is landed within are read off the same period, so what ``phase`` asks for in periods
    arrives in frames of this recording.
    """

    attack: int
    tail: int
    period: float
    series: FrameSeries
    phase: PhaseConfig

    @property
    def snap_radius(self) -> int:
        """How far either side of a named start an ascending zero crossing is looked for."""
        return round(self.period * self.phase.snap_periods)

    @property
    def match_radius(self) -> int:
        """How far either side of a whole count of periods a matched end is searched."""
        return round(self.period * self.phase.match_periods)


def _searched_lags(sample_rate: int, config: GeometryConfig, root_hz: float) -> tuple[int, int]:
    """The lags a recording played at ``root_hz`` has its period searched over, widest lag last.

    A tuning fork's worth of leeway either side of the played pitch covers the tuning a set was recorded at
    and the stretch a piano's own strings carry, while staying well inside the octave -- which is what keeps
    the peak found the note's own period rather than two or three of them, the reading a search over a whole
    band lands on wherever the even harmonics run strong.
    """
    spread = semitone_ratio(config.detune_semitones)
    return max(1, floor(sample_rate / (root_hz * spread))), ceil(sample_rate * spread / root_hz)


def _estimate_period(
    signal: Signal,
    sample_rate: int,
    config: GeometryConfig,
    root_hz: float,
) -> float | None:
    """Fundamental period in frames, from the strongest autocorrelation peak near the pitch the note was played at.

    The pitch is known from the key the recording sounds, so the search runs over the lags that pitch makes
    (:func:`_searched_lags`) and the peak found there is read between frames
    (:func:`~optisample.dsp.series.refined_lag`).

    Returns ``None`` when the signal spans fewer than ``_MIN_STEADY_FRAMES``, holds less than one period of
    its own pitch, or peaks below ``config.min_correlation`` -- material a loop has no purchase on, either
    because there is too little of it to read or because it repeats too loosely to wrap.
    """
    if signal.size < _MIN_STEADY_FRAMES:
        return None

    correlation = autocorrelation(signal)
    low, widest = _searched_lags(sample_rate, config, root_hz)
    high = min(signal.size - 1, widest)
    if high < low:
        return None

    lag = int(np.argmax(correlation[low : high + 1])) + low
    if correlation[lag] < config.min_correlation:
        return None

    return refined_lag(correlation, lag)


def _snap_ascending_zero(signal: Signal, index: int, radius: int) -> int:
    """Nearest ascending zero crossing (``-`` -> ``+``) to ``index`` within ``radius`` (else ``index``)."""
    low = max(1, index - radius)
    high = min(signal.size - 1, index + radius)
    best, best_distance = index, radius + 1
    for candidate in range(low, high + 1):
        if signal[candidate - 1] <= 0.0 < signal[candidate] and abs(candidate - index) < best_distance:
            best, best_distance = candidate, abs(candidate - index)

    return best


def _steady_tail(signal: Signal, sample_rate: int, config: GeometryConfig) -> int:
    """The frame the steady window closes at, which leaves the release the recording ends on alone."""
    return signal.size - seconds_to_frames(config.tail_skip_s, sample_rate)


def _settled_frame(series: FrameSeries, sample_rate: int, config: LoopConfig) -> int:
    """The frame the steady window opens at: where this recording's own material settles.

    The onset is read off the recording (:func:`~optisample.dsp.similarity.settling_frame`), so a note
    whose partials are still dying away at 300 ms opens its window there while one that holds its sound
    from the moment it begins opens at the start. Two bounds hold that reading to a window a loop can be
    taken from: ``min_fade_s`` is the material a wrap has to blend into, which is the earliest a region may
    begin, and ``max_attack_s`` is the longest the search waits, so material whose shape keeps moving still
    offers the rest of itself.
    """
    earliest = seconds_to_frames(config.seam.min_fade_s, sample_rate)
    latest = seconds_to_frames(config.geometry.max_attack_s, sample_rate)
    return min(max(settling_frame(series, config.features), earliest), latest)


def _steady_region(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
    root_hz: float,
) -> _SteadyRegion | Material:
    """The window a loop may be placed in, together with the period it repeats at.

    Material that declines as it rings is placed in just as steady material is: the region is held at one
    level (:func:`level_loop`) and brought down outside the PCM by
    :class:`~optisample.dsp.decay.LinearDecay`, so a struck note is stored as attack plus loop and declines
    from there.

    Answers with the reading that came up short for material a loop has no purchase on:
    :attr:`Material.STEADY` for a window shorter than ``_MIN_STEADY_FRAMES``, and
    :attr:`Material.PERIOD` for one carrying no reliable period.
    """
    series = frame_series(signal, sample_rate, config.features, root_hz)
    attack = _settled_frame(series, sample_rate, config)
    tail = _steady_tail(signal, sample_rate, config.geometry)
    if tail - attack < _MIN_STEADY_FRAMES:
        return Material.STEADY

    estimated = seconds_to_frames(config.geometry.max_estimation_s, sample_rate)
    window = signal[attack : min(tail, attack + estimated)]
    period = _estimate_period(window, sample_rate, config.geometry, root_hz)
    if period is None:
        return Material.PERIOD

    return _SteadyRegion(attack=attack, tail=tail, period=period, series=series, phase=config.phase)


def shortest_loop_frames(period: float, sample_rate: int, config: GeometryConfig) -> int:
    """The shortest loop worth offering: whole periods covering ``min_periods``, ``min_loop_s``, and a spectrum.

    Rounding the period count up makes ``min_loop_s`` a floor every stored loop clears, which is what
    keeps a loop long enough to carry the material's own movement instead of buzzing at its rate. The
    whole periods land on a frame boundary, which is where a loop's bounds live.

    A third floor holds the region at one analysis window or longer, so :func:`_spectral_distance` reads a
    spectrum off every candidate offered and the timbre gate judges each of them on a measurement
    of its own. That is what lets ``min_loop_s`` be set as short as the material allows: the gates stay the
    constraint at any floor, because a loop the gates could only wave through is never offered.
    """
    periods = max(
        config.min_periods,
        ceil(config.min_loop_s * sample_rate / period),
        ceil(_QUALITY_FFT / period),
    )
    return round(periods * period)


def _fitted_length(start: int, wanted: int, region: _SteadyRegion, shortest: int) -> int | None:
    """``wanted`` frames from ``start``, shortened to the whole periods the steady region has room for.

    Returns ``None`` where the room left holds less than ``shortest``, which is the floor
    :func:`shortest_loop_frames` sets.
    """
    if start + wanted <= region.tail:
        return wanted

    periods = int((region.tail - start) // region.period)
    fitted = min(round(periods * region.period), region.tail - start)
    return fitted if fitted >= shortest else None


def _normalized_correlation(first: Signal, second: Signal) -> float:
    """How alike two equal-length stretches are past the level each sits at: ``1.0`` for one shape twice.

    Silence correlates with nothing, so a stretch whose norm reaches ``_CORRELATION_FLOOR`` reads ``0.0``.
    """
    scale = float(np.linalg.norm(first)) * float(np.linalg.norm(second))
    if scale <= _CORRELATION_FLOOR:
        return 0.0

    return float(np.dot(first, second) / scale)


def _matched_end(signal: Signal, start: int, wanted: int, region: _SteadyRegion, shortest: int) -> int:
    """The end near ``start + wanted`` whose approach best matches the approach to ``start``.

    A wrap carries the frames before the end onto the frames before the start, so the end worth keeping is
    the one already shaped like that approach: correlating the two stretches over one period lands the
    material in phase where a whole count of periods only comes close, since a period read off a finite
    window carries an error every period of a loop compounds.

    The search runs :attr:`_SteadyRegion.match_radius` either side of the whole count and stays inside the
    steady region and no shorter than ``shortest``, so a matched end names a loop the geometry accepts.
    Where the room before ``start`` holds less than two frames to correlate over, the whole count stands.
    """
    window = min(round(region.period), start, wanted)
    ideal = start + wanted
    if window < _MIN_LOOP_FRAMES:
        return ideal

    approaching_start = signal[start - window : start]
    radius = region.match_radius
    ends = range(max(start + shortest, ideal - radius), min(region.tail, ideal + radius) + 1)
    return max(ends, key=lambda end: _normalized_correlation(signal[end - window : end], approaching_start))


def _placed(signal: Signal, reach: LoopReach, region: _SteadyRegion, shortest: int) -> Loop | None:
    """The loop a frontier reach names, landed on the phase the waveform itself makes.

    A reach is read off frames of the timbre series, each spanning several periods of the note, so both
    bounds are brought onto the waveform: the start snaps to the nearest ascending zero crossing within
    :attr:`_SteadyRegion.snap_radius`, which lands the wrap mid-slope in phase for the seam crossfade to
    smooth over, and the end is matched to the phase the start approaches on (:func:`_matched_end`) once
    the length is fitted to the room the region has.

    Returns ``None`` where the room past the snapped start holds less than the shortest accepted loop.
    """
    start = _snap_ascending_zero(signal, reach.start, radius=region.snap_radius)
    fitted = _fitted_length(start, max(reach.length, shortest), region, shortest)
    if fitted is None:
        return None

    return Loop(start, _matched_end(signal, start, fitted, region, shortest))


def _frontier_bounds(sample_rate: int, region: _SteadyRegion, config: LoopConfig, shortest: int) -> FrontierBounds:
    """The stretch rounds are looked for over: from where the material settled, as far as it is worth reading.

    ``max_reach_s`` past the settled frame is both the longest round offered and how much of a recording
    one search reads, which holds a note that rings for a minute to the same work as one that rings for a
    second -- the rounds past it store so much of the recording that keeping the played span costs less.
    What each round stands in for reaches the whole way to the tail regardless, so a long note prices its
    rounds against everything it goes on to do.
    """
    reach = seconds_to_frames(config.frontier.max_reach_s, sample_rate)
    return FrontierBounds(
        opens=region.attack,
        reach=min(region.tail, region.attack + reach),
        ends=region.tail,
        shortest=shortest,
    )


def _lacking(series: FrameSeries, bounds: FrontierBounds, config: LoopConfig, reaches: int) -> Material:
    """Which reading left a recording with no loop to place, read in the order the search takes them.

    Room comes first because a window holding less than one round is measured no wraps at all, then the
    wraps themselves, and last the landing: a reach the frontier named that the waveform had no room to
    land on (:func:`_placed`).
    """
    if not holds_a_round(series, bounds, config.features):
        return Material.ROUND

    return Material.MOVEMENT if reaches == 0 else Material.PHASE


def loop_search(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
    root_hz: float,
) -> LoopSearch:
    """Every forward loop worth offering for ``signal``, the frontier a settlement prices.

    Which rounds are on offer is read off the recording's own self-similarity
    (:func:`~optisample.dsp.loopability.loop_frontier`): each length is placed where the material wraps
    onto itself most closely, and the lower convex hull of what those placements store against how far the
    sound moves across their wraps is the frontier. Storing a candidate keeps ``[0, loop.end)`` -- the
    attack plus one round -- and the sustain tail past it is where the bytes are saved, so the frontier is
    a rate axis a budget reads directly.

    Each reach is then landed on the waveform (:func:`_placed`): a start at an ascending zero crossing, a
    length of about a whole number of periods of the material (read off the pitch it was played at by
    :func:`_estimate_period`), and an end matched to the phase the start approaches on. Every candidate
    lies inside the steady region, between the frame the recording's material settles at
    (:func:`_settled_frame`) and the tail skip, and clears the shortest loop worth storing
    (:func:`shortest_loop_frames`).

    Material a loop has no purchase on offers no candidate and states which reading came up short
    (:class:`Material`), so a recording stored over the span it plays says why.
    """
    region = _steady_region(signal, sample_rate, config, root_hz)
    if isinstance(region, Material):
        return LoopSearch(candidates=(), lacking=region)

    shortest = shortest_loop_frames(region.period, sample_rate, config.geometry)
    bounds = _frontier_bounds(sample_rate, region, config, shortest)
    reaches = loop_frontier(region.series, bounds, config.features, config.frontier)
    found: list[Loop] = []
    for reach in reaches:
        placed = _placed(signal, reach, region, shortest)
        if placed is not None:
            found.append(placed)

    if not found:
        return LoopSearch(candidates=(), lacking=_lacking(region.series, bounds, config, len(reaches)))

    return LoopSearch(candidates=tuple(dict.fromkeys(found)), lacking=None)


def loop_at_rate(loop: Loop, orig_rate: int, target_rate: int, *, frames: int) -> Loop | None:
    """``loop`` scaled onto a copy of the same material stored at ``target_rate`` and ``frames`` long.

    Both bounds scale by the rate ratio (:func:`~optisample.dsp.resample.resampled_frame_count`), which is
    the same map the resampler puts the material through, so the scaled loop covers the same stretch of the
    recording. The end is held inside ``frames`` so it names a frame the copy holds.

    Scaling rounds each bound to a whole frame, leaving the wrap up to half a frame off the whole periods it
    spanned at ``orig_rate``. That is a fraction of a cycle, which :func:`crossfade_loop` blends into a ripple
    rather than a step. Returns ``None`` where the scaled bounds leave less than ``_MIN_LOOP_FRAMES``, which
    is a copy stored at too low a rate for this loop to survive onto.
    """
    start = resampled_frame_count(loop.start, orig_rate, target_rate)
    end = min(resampled_frame_count(loop.end, orig_rate, target_rate), frames)
    if end - start < _MIN_LOOP_FRAMES:
        return None

    return Loop(start=start, end=end)


def seam_frames(loop: Loop, sample_rate: int, config: SeamConfig) -> int:
    """The stretch of ``loop`` the wrap is blended over, in frames at ``sample_rate``.

    A blend stated as a share of the loop's own length covers the same fraction of every round, so a short
    loop is blended over a stretch as long a part of it as a long one is, and the floor in seconds keeps a
    very short loop's blend long enough to carry the material's motion. The material preceding ``loop.start``
    is what the blend reaches for and the region is what it rewrites, so both bound it.
    """
    wanted = max(round(config.fade_share * loop.length), round(config.min_fade_s * sample_rate))
    return min(wanted, loop.start, loop.length)


def _seam_weights(correlation: float, progress: Signal) -> tuple[Signal, Signal]:
    """The weights carrying a stretch onto one ``correlation`` alike at the level the two of them held.

    Turning the blend through a quarter circle -- a cosine holding the stretch it is leaving, a sine
    reaching for the one it returns to -- leaves it carrying ``1 + correlation * sin(2 * angle)`` of the
    level the material had, so dividing both sides by the root of that holds the level exactly the whole
    way across. The divisor rests at 1 where the angle reaches either end, so the blend opens on the
    material it held and closes on the material it reaches for.

    Material that came round in phase reads 1.0 and is weighted so the two sides sum to one; material whose
    partials arrived elsewhere reads 0.0 and is weighted so their squares do; material that came round
    inverted reads below 0.0, where the two sides take each other out and the weighting lifts them to make
    up the difference. That lift reaches ``_MAX_SEAM_GAIN``, which is what a band cancelling outright is
    left holding, and it is the reading a struck string's upper partials land on often enough to hear.

    Returns the weighting on the stretch being held and on the one being reached for.
    """
    angle = 0.5 * np.pi * progress
    carried = 1.0 + correlation * np.sin(2.0 * angle)
    gain = 1.0 / np.sqrt(np.maximum(carried, 1.0 / _MAX_SEAM_GAIN**2))
    return np.cos(angle) * gain, np.sin(angle) * gain


def _banded_seam(signal: Signal, loop: Loop, fade: int, sample_rate: int, config: SeamConfig) -> tuple[Signal, Signal]:
    """The two stretches a wrap joins, each separated into the bands the blend weighs: ``(n_bands, fade)``.

    Each split is read over the material surrounding its stretch as well -- as much again on each side as
    the stretch itself, where the recording holds it -- so what reaches the edges of each band is the
    neighbouring material the recording actually made there. Both are read over the same span, which is what
    the tighter of the two has room for, so one bank serves them and the bands line up to be weighed against
    each other. The bands sum back to their stretch frame by frame
    (:func:`~optisample.dsp.spectral.split_bands`), so weighting them apart still carries the whole of it.

    Returns the stretch approaching ``loop.end`` and the stretch preceding ``loop.start``.
    """
    lead = min(fade, loop.start - fade)
    trail = min(fade, signal.size - loop.end)

    def banded(stop: int) -> Signal:
        context = signal[stop - fade - lead : stop + trail]
        bands = split_bands(context, sample_rate, config.crossovers_hz, config.crossover_octaves)
        return np.asarray(bands[:, lead : lead + fade], dtype=np.float64)

    return banded(loop.end), banded(loop.start)


def _band_ramps(holding: Signal, reaching: Signal) -> tuple[Signal, Signal]:
    """The ramps that carry each band of ``holding`` onto its own band of ``reaching`` at the level it had.

    Every band is weighted by the likeness it measures on its own (:func:`_seam_weights`), so a band that
    came round in phase is summed, one whose partials arrived elsewhere has its squares summed, and one that
    came round inverted is lifted to make up what its two sides take out of each other.

    Returns the weighting on each side, ``(n_bands, fade)`` apiece.
    """
    progress = np.linspace(0.0, 1.0, holding.shape[1], endpoint=True)
    weighted = [_seam_weights(_normalized_correlation(one, other), progress) for one, other in zip(holding, reaching)]
    return np.stack([held for held, _ in weighted]), np.stack([reached for _, reached in weighted])


def crossfade_loop(signal: Signal, loop: Loop, sample_rate: int, config: SeamConfig) -> Signal:
    """Blend the seam so the loop wraps smoothly; returns a copy with the end of the region rewritten.

    The frames before ``loop.end`` are ramped from themselves toward the frames that precede ``loop.start``,
    so ``signal[end - 1]`` lands on ``signal[start - 1]`` -- making the wrap into ``signal[start]``
    continuous. Both stretches are read band by band (:func:`_banded_seam`) and each band carries the
    weighting law its own likeness names (:func:`_band_ramps`), so every part of the spectrum comes through
    the blend at the level it had. That is what a string's own tuning asks for: one round of a loop returns
    the fundamental to the phase it left while the partials above it, spaced a little wider than whole
    multiples of it, arrive where their own spacing puts them, so the likeness that holds one of them holds
    the rest at a notch.

    A loop the room before its start holds no frames for is returned as it stands, which is a wrap the
    material makes on its own.
    """
    fade = seam_frames(loop, sample_rate, config)
    if fade <= 0:
        return np.asarray(signal, dtype=np.float64)

    approaching_end, preceding_start = _banded_seam(signal, loop, fade, sample_rate, config)
    holding_end, toward_start = _band_ramps(approaching_end, preceding_start)
    out = np.array(signal, dtype=np.float64)
    blended = holding_end * approaching_end + toward_start * preceding_start
    out[loop.end - fade : loop.end] = np.sum(blended, axis=0)
    return out


def level_loop(signal: Signal, loop: Loop, reading: LevelReading) -> Signal:
    """``signal`` with its loop region held at the level that region starts on; returns a copy.

    A region taken from material that declines as it rings falls from ``loop.start`` to ``loop.end``, so a
    player wrapping it steps the level back up once per round and a held note pulses at the loop's rate.
    Dividing the region by the level its own material holds
    (:func:`~optisample.dsp.envelope.local_level_over`) holds it at one amplitude, pinned at the level it
    starts on so the attack runs into the region continuously and the decline the region gives up is what a
    fitted ramp restores (:func:`~optisample.dsp.decay.fit_linear_decay`). The reading follows the material
    frame by frame, so a region that swells and falls again comes out as flat as one that only falls.

    The gain stays under ``_MAX_LEVEL_GAIN``, so a region ringing its way down to silence is lifted only as
    far as flattening it holds its own noise floor down, and what a region falling further keeps is the
    part of its decline that gain reaches.
    """
    level = local_level_over(signal, reading, start=loop.start, end=loop.end)
    out = np.array(signal, dtype=np.float64)
    out[loop.start : loop.end] *= np.clip(level[0] / level, 0.0, _MAX_LEVEL_GAIN)
    return out


def prepare_loop(signal: Signal, loop: Loop, sample_rate: int, seam: SeamConfig, reading: LevelReading) -> Signal:
    """``signal`` with its loop region ready to wrap: held at one level, then blended at the seam.

    Levelling runs first, so the two stretches the blend joins sit at the same amplitude and the blend is
    left to join phase alone. Every stretch that measures, stores or plays a loop passes through here, which
    is what makes a report, an audition and a stored sample wrap the same waveform.
    """
    return crossfade_loop(level_loop(signal, loop, reading), loop, sample_rate, seam)


def _seam_step(signal: Signal, loop: Loop) -> float:
    """The jump the wrap makes, in units of the frame-to-frame motion the waveform makes right there.

    The step is read against the motion in the stretches on either side of the wrap rather than across the
    whole region, because the wrap is heard against the waveform it lands in. Material that declines as it
    rings moves less at the end of a region than at its start, so an average taken over the whole region
    would report a step in units of motion the wrap never sits next to.
    """
    region = signal[loop.start : loop.end]
    if region.size < 2:
        return 0.0

    window = max(_MIN_LOOP_FRAMES, region.size // _SEAM_WINDOW_SHARE)
    nearby = np.concatenate([region[:window], region[-window:]])
    typical = float(np.mean(np.abs(np.diff(nearby))))
    return float(abs(region[0] - region[-1])) / max(typical, _STEP_FLOOR)


def _level_drift_db(signal: Signal, loop: Loop, reading: LevelReading) -> float:
    """How far the loop region's own level falls across it, in decibels, positive where it declines.

    This is what holding the region at one level costs: the gain levelling asks of the material by the far
    end of the region. It is read off the region as the recording made it, which is the fall a listener would
    have heard step back up once per round. A region holding its level reads ``0.0``, and one that rises
    across itself reads a negative fall, which is levelling holding it back to the level it starts on.
    """
    level = local_level_over(signal, reading, start=loop.start, end=loop.end)
    return gain_to_db(float(level[0])) - gain_to_db(float(level[-1]))


def _spectral_shape(signal: Signal) -> Signal:
    """Frame-averaged magnitude spectrum of ``signal``, normalized to unit sum so it reads shape alone."""
    magnitude = stft_magnitude(signal, StftParams(n_fft=_QUALITY_FFT, hop_length=_QUALITY_HOP)).mean(axis=0)
    return np.asarray(magnitude / max(float(np.sum(magnitude)), _SPECTRUM_FLOOR), dtype=np.float64)


def _spectral_distance(region: Signal, material: Signal) -> float:
    """Root-mean-square log-spectral distance in decibels between two stretches of the same recording.

    Both stretches carry a spectrum once they hold an analysis window each. Every candidate offered
    clears that on the region side (:func:`shortest_loop_frames`), so what remains is a loop
    reaching so far into its recording that the material past it holds less than a window -- a stretch the
    loop gives up nothing by standing in for, which reads 0.0.
    """
    if region.size < _QUALITY_FFT or material.size < _QUALITY_FFT:
        return 0.0

    difference = _AMPLITUDE_DB * np.log10(
        (_spectral_shape(region) + _SPECTRUM_FLOOR) / (_spectral_shape(material) + _SPECTRUM_FLOOR)
    )
    return float(np.sqrt(np.mean(difference**2)))


def loop_quality(
    signal: Signal,
    loop: Loop,
    sample_rate: int,
    config: LoopConfig,
    reading: LevelReading,
) -> LoopQuality:
    """Measure what storing ``loop`` costs: the seam it wraps on, the level it holds, and the timbre it keeps.

    The seam and the timbre are read off the prepared region (:func:`prepare_loop`), which is the waveform a
    player wraps, so both state what storing this loop actually sounds like. The material a loop stands in
    for is the steady region past its end -- the stretch a looped sample stops storing -- so a loop taken
    from a part of the note that has moved on in timbre reports the distance. A loop reaching the end of the
    steady region stands in for less than one analysis window and reports a distance of 0.0. The drift is
    read off the recording as it stands, which is the fall levelling had to flatten.
    """
    prepared = prepare_loop(signal, loop, sample_rate, config.seam, reading)
    tail = _steady_tail(signal, sample_rate, config.geometry)
    return LoopQuality(
        seam_step=_seam_step(prepared, loop),
        level_drift_db=_level_drift_db(signal, loop, reading),
        spectral_distance=_spectral_distance(prepared[loop.start : loop.end], signal[loop.end : tail]),
    )
