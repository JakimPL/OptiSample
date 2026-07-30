from dataclasses import dataclass
from math import ceil, log2
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.loop import GeometryConfig, SeamConfig
from optisample.config.spectral import StftParams
from optisample.dsp.levels import gain_to_db, level_trend
from optisample.dsp.resample import resampled_frame_count
from optisample.dsp.spectral import stft_magnitude

Signal = NDArray[np.float64]

_MIN_STEADY_FRAMES: Final = 8
_MIN_LOOP_FRAMES: Final = 2  # frames a wrap needs to name two distinct ends
_SEAM_WINDOW_SHARE: Final = 100  # share of a loop read on each side of the wrap as the motion it lands in
_QUALITY_FFT: Final = 1024  # window the loop region and the stretch it stands for are compared over
_QUALITY_HOP: Final = 512
_AMPLITUDE_DB: Final = 20.0  # decibels per decade of amplitude
_SPECTRUM_FLOOR: Final = 1e-10
_STEP_FLOOR: Final = 1e-12
_PEAK_CURVATURE: Final = 1e-12  # concavity a peak holds for a parabola to read a lag between frames
_MATCH_SHARE: Final = 2  # share of a period the matched end is searched on either side of a whole count
_CORRELATION_FLOOR: Final = 1e-24  # product of two norms a silent stretch reaches, which correlates with nothing
_LEVEL_FLOOR: Final = 1e-12  # the level a silent stretch reads as, which leaves a ratio against it finite
_MAX_LEVEL_GAIN: Final = 4.0  # +12 dB, the most holding a region at one level asks of the material


@dataclass(frozen=True)
class Loop:
    """A forward loop over the half-open frame range ``[start, end)`` (playback wraps ``end`` -> ``start``)."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


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
    """The stretch a loop may be placed in, and the period its material repeats at."""

    attack: int
    tail: int
    period: float


def _autocorrelation(signal: Signal) -> Signal:
    """Unbiased-enough autocorrelation via FFT (lags ``0..n-1``), normalized so lag 0 == 1."""
    centered = signal - float(np.mean(signal))
    length = centered.size
    size = 1 << int(np.ceil(np.log2(2 * length)))
    spectrum = np.fft.rfft(centered, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:length]
    if correlation[0] <= 0.0:
        return np.zeros(length, dtype=np.float64)
    return np.asarray(correlation / correlation[0], dtype=np.float64)


def _refined_lag(correlation: Signal, lag: int) -> float:
    """The lag the autocorrelation peaks at, read between frames by the parabola through its top three.

    A peak read to the nearest frame sits up to half a frame from the material's own cycle, and a loop spans
    several periods, so that error accumulates into a wrap landing part-way through a cycle. Fitting a
    parabola to the peak and its two neighbours states the lag to a fraction of a frame, which is what lets
    a whole number of periods mean what it says at every rate a recording is analysed at.

    A peak on the first or last lag searched, or one the neighbours make no concave top with, is read at its
    own frame -- there is no parabola through it to place the lag between frames.
    """
    if lag <= 0 or lag >= correlation.size - 1:
        return float(lag)

    before, peak, after = float(correlation[lag - 1]), float(correlation[lag]), float(correlation[lag + 1])
    curvature = before - 2.0 * peak + after
    if curvature > -_PEAK_CURVATURE:
        return float(lag)

    return float(lag) + float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))


def _estimate_period(
    signal: Signal,
    sample_rate: int,
    config: GeometryConfig,
) -> float | None:
    """Fundamental period in frames from the strongest autocorrelation peak in the pitched band.

    Searches lags from ``sample_rate / max_hz`` (the shortest period the band admits) to
    ``sample_rate / min_hz`` (the longest), and reads the peak found between frames
    (:func:`_refined_lag`). Returns ``None`` when the signal spans fewer than ``_MIN_STEADY_FRAMES``, the
    band holds no lag at this rate, or the strongest peak stays below ``config.min_correlation`` -- each the
    mark of material too aperiodic to loop.
    """
    if signal.size < _MIN_STEADY_FRAMES:
        return None

    correlation = _autocorrelation(signal)
    low = max(1, int(sample_rate / config.max_hz))
    high = min(signal.size - 1, int(sample_rate / config.min_hz))
    if high <= low:
        return None

    lag = int(np.argmax(correlation[low : high + 1])) + low
    if correlation[lag] < config.min_correlation:
        return None

    return _refined_lag(correlation, lag)


def _snap_ascending_zero(signal: Signal, index: int, radius: int) -> int:
    """Nearest ascending zero crossing (``-`` -> ``+``) to ``index`` within ``radius`` (else ``index``)."""
    low = max(1, index - radius)
    high = min(signal.size - 1, index + radius)
    best, best_distance = index, radius + 1
    for candidate in range(low, high + 1):
        if signal[candidate - 1] <= 0.0 < signal[candidate] and abs(candidate - index) < best_distance:
            best, best_distance = candidate, abs(candidate - index)

    return best


def _steady_bounds(
    signal: Signal,
    sample_rate: int,
    config: GeometryConfig,
) -> tuple[int, int]:
    """The ``[attack, tail)`` frame window to analyse: past the onset transient, before the release."""
    attack = int(config.attack_skip_s * sample_rate)
    tail = signal.size - int(config.tail_skip_s * sample_rate)
    return attack, tail


def _steady_region(
    signal: Signal,
    sample_rate: int,
    config: GeometryConfig,
) -> _SteadyRegion | None:
    """The window a loop may be placed in, together with the period it repeats at.

    Material that declines as it rings is placed in just as steady material is: the region is held at one
    level (:func:`level_loop`) and brought down outside the PCM by
    :class:`~optisample.dsp.decay.LinearDecay`, so a struck note is stored as attack plus loop and declines
    from there.

    Returns ``None`` for material a loop has no purchase on: a steady window shorter than
    ``_MIN_STEADY_FRAMES``, or one carrying no reliable period.
    """
    attack, tail = _steady_bounds(signal, sample_rate, config)
    if tail - attack < _MIN_STEADY_FRAMES:
        return None

    window = signal[attack : min(tail, attack + int(config.max_estimation_s * sample_rate))]
    period = _estimate_period(window, sample_rate, config)
    if period is None:
        return None

    return _SteadyRegion(attack=attack, tail=tail, period=period)


def shortest_loop_frames(period: float, sample_rate: int, config: GeometryConfig) -> int:
    """The shortest loop the config accepts: whole periods covering ``min_periods`` and ``min_loop_s``.

    Rounding the period count up makes ``min_loop_s`` a floor every stored loop clears, which is what
    keeps a loop long enough to carry the material's own movement instead of buzzing at its rate. The
    whole periods land on a frame boundary, which is where a loop's bounds live.
    """
    periods = max(config.min_periods, ceil(config.min_loop_s * sample_rate / period))
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

    The search runs half a period either side of the whole count -- far enough to reach any phase, near
    enough to keep the count the geometry laid out -- and stays inside the steady region and no shorter than
    ``shortest``, so a matched end names a loop the geometry accepts. Where the room before ``start`` holds
    less than two frames to correlate over, the whole count stands.
    """
    window = min(round(region.period), start, wanted)
    ideal = start + wanted
    if window < _MIN_LOOP_FRAMES:
        return ideal

    approaching_start = signal[start - window : start]
    radius = round(region.period / _MATCH_SHARE)
    ends = range(max(start + shortest, ideal - radius), min(region.tail, ideal + radius) + 1)
    return max(ends, key=lambda end: _normalized_correlation(signal[end - window : end], approaching_start))


def _placements(signal: Signal, region: _SteadyRegion, shortest: int, config: GeometryConfig) -> list[int]:
    """Loop starts spread evenly through the room the steady region has, snapped to ascending zeros.

    The first placement sits at the attack skip and the last as late as the shortest accepted loop still
    fits, so ``placements`` readings span the whole stretch a loop may be taken from. Snapping each to an
    ascending zero crossing lands the wrap mid-slope in phase, which is what the seam crossfade then
    smooths over.
    """
    room = max(0, region.tail - region.attack - shortest)
    spacing = room / (config.placements - 1) if config.placements > 1 else 0.0
    starts = [
        _snap_ascending_zero(signal, region.attack + round(index * spacing), radius=round(region.period))
        for index in range(config.placements)
    ]
    return list(dict.fromkeys(starts))


def loop_candidates(
    signal: Signal,
    sample_rate: int,
    config: GeometryConfig,
) -> tuple[Loop, ...]:
    """Every forward loop worth offering for ``signal``, the ladder a settlement chooses from.

    Each candidate begins at an ascending zero crossing, spans about a whole number of periods of the
    material (found by :func:`_estimate_period`) with its end matched to the phase the start approaches on
    (:func:`_matched_end`), and lies inside the steady region between the attack skip and the tail skip.
    Placement runs outermost and length innermost, so the ladder covers both axes: the loop the attack leads
    into at each accepted length, then the same at placements further into the note.

    Storing a candidate keeps ``[0, loop.end)`` -- the attack plus one loop region -- and the sustain
    tail past it is where the bytes are saved. Lengths are the multiples of the shortest accepted loop
    (:func:`shortest_loop_frames`) that ``config.length_multiples`` asks for, each shortened to the whole
    periods the region has room for, so a longer loop carries more of the material's own movement where
    the note is long enough to hold it.

    Returns an empty tuple for material a loop has no purchase on: a steady region too short to analyse,
    one carrying no reliable period, or one with room for less than the shortest accepted loop.
    """
    region = _steady_region(signal, sample_rate, config)
    if region is None:
        return ()

    shortest = shortest_loop_frames(region.period, sample_rate, config)
    found: list[Loop] = []
    for start in _placements(signal, region, shortest, config):
        for multiple in config.length_multiples:
            fitted = _fitted_length(start, shortest * multiple, region, shortest)
            if fitted is not None:
                found.append(Loop(start, _matched_end(signal, start, fitted, region, shortest)))

    return tuple(dict.fromkeys(found))


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


def _fade_exponent(correlation: float) -> float:
    """The weighting law that holds the level of a blend of two stretches ``correlation`` alike.

    Weighting each side of a blend by ``x ** exponent`` leaves it carrying
    ``end**2 + start**2 + 2 * correlation * end * start`` of the level the material had, and solving that
    for one at the middle of the blend gives ``(1 + log2(1 + correlation)) / 2``, which holds across the
    rest of the blend to a fraction of a decibel. Material that stayed in phase reads 1.0 and is weighted
    so the two sides sum to one; material whose partials have drifted apart reads 0.0 and is weighted so
    their squares do; anything between lands between. Reading the law off the material is what lets one
    blend serve a tone that repeats exactly and a piano that has moved on by the time a loop comes round.
    """
    return 0.5 * (1.0 + log2(1.0 + max(correlation, 0.0)))


def crossfade_loop(signal: Signal, loop: Loop, sample_rate: int, config: SeamConfig) -> Signal:
    """Blend the seam so the loop wraps smoothly; returns a copy with the end of the region rewritten.

    The frames before ``loop.end`` are ramped from themselves toward the frames that precede ``loop.start``,
    so ``signal[end - 1]`` lands on ``signal[start - 1]`` -- making the wrap into ``signal[start]``
    continuous. How alike the two stretches measure is what sets the weighting law (:func:`_fade_exponent`),
    so the blend holds the level the material had all the way across.

    A loop the room before its start holds no frames for is returned as it stands, which is a wrap the
    material makes on its own.
    """
    fade = seam_frames(loop, sample_rate, config)
    if fade <= 0:
        return np.asarray(signal, dtype=np.float64)

    approaching_end = signal[loop.end - fade : loop.end]
    preceding_start = signal[loop.start - fade : loop.start]
    exponent = _fade_exponent(_normalized_correlation(approaching_end, preceding_start))
    progress = np.linspace(0.0, 1.0, fade, endpoint=True)
    toward_start = progress**exponent
    holding_end = (1.0 - progress) ** exponent
    out = np.array(signal, dtype=np.float64)
    out[loop.end - fade : loop.end] = holding_end * approaching_end + toward_start * preceding_start
    return out


def level_loop(signal: Signal, loop: Loop, sample_rate: int) -> Signal:
    """``signal`` with its loop region held at the level that region starts on; returns a copy.

    A region taken from material that declines as it rings falls from ``loop.start`` to ``loop.end``, so a
    player wrapping it steps the level back up once per round and a held note pulses at the loop's rate.
    Dividing the region by the line its own level readings make
    (:func:`~optisample.dsp.levels.level_trend`) holds it at one amplitude, pinned at the level it starts on
    so the attack runs into the region continuously and the decline the region gives up is what a fitted ramp
    restores (:func:`~optisample.dsp.decay.fit_linear_decay`).

    The gain stays under ``_MAX_LEVEL_GAIN``, so a region whose trend reads its way down to silence is lifted
    only so far; how far flattening reaches is what ``max_level_drift_db`` bounds. A region too short for a
    level line, or one starting from silence, is returned as it stands.
    """
    region = np.asarray(signal[loop.start : loop.end], dtype=np.float64)
    trend = level_trend(region, sample_rate)
    if trend is None or trend.at(0.0) <= _LEVEL_FLOOR:
        return np.asarray(signal, dtype=np.float64)

    seconds = np.arange(region.size, dtype=np.float64) / sample_rate
    gain = np.clip(trend.at(0.0) / np.maximum(trend.at(seconds), _LEVEL_FLOOR), 0.0, _MAX_LEVEL_GAIN)
    out = np.array(signal, dtype=np.float64)
    out[loop.start : loop.end] = region * gain
    return out


def prepare_loop(signal: Signal, loop: Loop, sample_rate: int, config: SeamConfig) -> Signal:
    """``signal`` with its loop region ready to wrap: held at one level, then blended at the seam.

    Levelling runs first, so the two stretches the blend joins sit at the same amplitude and the blend is
    left to join phase alone. Every stretch that measures, stores or plays a loop passes through here, which
    is what makes a report, an audition and a stored sample wrap the same waveform.
    """
    return crossfade_loop(level_loop(signal, loop, sample_rate), loop, sample_rate, config)


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


def _level_drift_db(signal: Signal, loop: Loop, sample_rate: int) -> float:
    """How far the loop region's own level falls across it, in decibels, positive where it declines.

    This is what holding the region at one level costs: the gain levelling asks of the material by the far
    end of the region. It is read off the region as the recording made it, which is the fall a listener would
    have heard step back up once per round. A region holding its level reads ``0.0``, and so does one too
    short for a line to be drawn through its readings.
    """
    region = np.asarray(signal[loop.start : loop.end], dtype=np.float64)
    trend = level_trend(region, sample_rate)
    if trend is None:
        return 0.0

    return gain_to_db(trend.at(0.0)) - gain_to_db(trend.at(region.size / sample_rate))


def _spectral_shape(signal: Signal) -> Signal:
    """Frame-averaged magnitude spectrum of ``signal``, normalized to unit sum so it reads shape alone."""
    magnitude = stft_magnitude(signal, StftParams(n_fft=_QUALITY_FFT, hop_length=_QUALITY_HOP)).mean(axis=0)
    return np.asarray(magnitude / max(float(np.sum(magnitude)), _SPECTRUM_FLOOR), dtype=np.float64)


def _spectral_distance(region: Signal, material: Signal) -> float:
    """Root-mean-square log-spectral distance in decibels between two stretches of the same recording.

    A stretch shorter than one analysis window carries too little for a spectrum to be read off, so it
    reports a distance of 0.0.
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
    geometry: GeometryConfig,
    seam: SeamConfig,
) -> LoopQuality:
    """Measure what storing ``loop`` costs: the seam it wraps on, the level it holds, and the timbre it keeps.

    The seam and the timbre are read off the prepared region (:func:`prepare_loop`), which is the waveform a
    player wraps, so both state what storing this loop actually sounds like. The material a loop stands in
    for is the steady region past its end -- the stretch a looped sample stops storing -- so a loop taken
    from a part of the note that has moved on in timbre reports the distance. A loop reaching the end of the
    steady region stands in for less than one analysis window and reports a distance of 0.0. The drift is
    read off the recording as it stands, which is the fall levelling had to flatten.
    """
    prepared = prepare_loop(signal, loop, sample_rate, seam)
    _, tail = _steady_bounds(signal, sample_rate, geometry)
    return LoopQuality(
        seam_step=_seam_step(prepared, loop),
        level_drift_db=_level_drift_db(signal, loop, sample_rate),
        spectral_distance=_spectral_distance(prepared[loop.start : loop.end], signal[loop.end : tail]),
    )
