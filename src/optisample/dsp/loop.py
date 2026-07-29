from dataclasses import dataclass
from math import ceil
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.codec import LoopConfig
from optisample.config.spectral import StftParams
from optisample.dsp.spectral import stft_magnitude

Signal = NDArray[np.float64]

_MIN_STEADY_FRAMES: Final = 8
_QUALITY_FFT: Final = 1024  # window the loop region and the stretch it stands for are compared over
_QUALITY_HOP: Final = 512
_AMPLITUDE_DB: Final = 20.0  # decibels per decade of amplitude
_SPECTRUM_FLOOR: Final = 1e-10
_STEP_FLOOR: Final = 1e-12


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
    round. ``spectral_distance`` is the log-spectral distance in decibels between the loop region and the
    stretch it plays in place of, measured over spectra normalized to unit sum so it reads the timbre a
    loop holds on to while the material moved on.
    """

    seam_step: float
    spectral_distance: float


@dataclass(frozen=True)
class _SteadyRegion:
    """The stretch a loop may be placed in, and the period its material repeats at."""

    attack: int
    tail: int
    period: int


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


def _estimate_period(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
) -> int | None:
    """Fundamental period in frames from the strongest autocorrelation peak in the pitched band.

    Searches lags from ``sample_rate / max_hz`` (the shortest period the band admits) to
    ``sample_rate / min_hz`` (the longest). Returns ``None`` when the signal spans fewer than
    ``_MIN_STEADY_FRAMES``, the band holds no lag at this rate, or the strongest peak stays below
    ``config.min_correlation`` -- each the mark of material too aperiodic to loop.
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

    return lag


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
    config: LoopConfig,
) -> tuple[int, int]:
    """The ``[attack, tail)`` frame window to analyse: past the onset transient, before the release."""
    attack = int(config.attack_skip_s * sample_rate)
    tail = signal.size - int(config.tail_skip_s * sample_rate)
    return attack, tail


def _steady_region(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
) -> _SteadyRegion | None:
    """The window a loop may be placed in, together with the period it repeats at.

    Material that declines as it rings is placed in just as steady material is: the level a loop settles
    on is brought down outside the PCM by :class:`~optisample.dsp.decay.LinearDecay`, so a struck note is
    stored as attack plus loop and declines from there.

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


def shortest_loop_frames(period: int, sample_rate: int, config: LoopConfig) -> int:
    """The shortest loop the config accepts: whole periods covering ``min_periods`` and ``min_loop_s``.

    Rounding the period count up makes ``min_loop_s`` a floor every stored loop clears, which is what
    keeps a loop long enough to carry the material's own movement instead of buzzing at its rate.
    """
    periods = max(config.min_periods, ceil(config.min_loop_s * sample_rate / period))
    return periods * period


def _fitted_length(start: int, wanted: int, region: _SteadyRegion, shortest: int) -> int | None:
    """``wanted`` frames from ``start``, shortened to the whole periods the steady region has room for.

    Returns ``None`` where the room left holds less than ``shortest``, which is the floor
    :func:`shortest_loop_frames` sets.
    """
    if start + wanted <= region.tail:
        return wanted

    fitted = ((region.tail - start) // region.period) * region.period
    return fitted if fitted >= shortest else None


def _placements(signal: Signal, region: _SteadyRegion, shortest: int, config: LoopConfig) -> list[int]:
    """Loop starts spread evenly through the room the steady region has, snapped to ascending zeros.

    The first placement sits at the attack skip and the last as late as the shortest accepted loop still
    fits, so ``placements`` readings span the whole stretch a loop may be taken from. Snapping each to an
    ascending zero crossing lands the wrap mid-slope in phase, which is what the seam crossfade then
    smooths over.
    """
    room = max(0, region.tail - region.attack - shortest)
    spacing = room / (config.placements - 1) if config.placements > 1 else 0.0
    starts = [
        _snap_ascending_zero(signal, region.attack + round(index * spacing), radius=region.period)
        for index in range(config.placements)
    ]
    return list(dict.fromkeys(starts))


def loop_candidates(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
) -> tuple[Loop, ...]:
    """Every forward loop worth offering for ``signal``, ordered so the first is the default choice.

    Each candidate spans a whole number of periods of the material (found by :func:`_estimate_period`),
    begins at an ascending zero crossing, and lies inside the steady region between the attack skip and
    the tail skip. Placement runs outermost and length innermost, so a caller taking the first few
    candidates sees both axes early: the loop the attack leads into at each accepted length, then the
    same at placements further into the note. Candidate 0 is therefore the shortest accepted loop right
    after the attack, which is the one :func:`detect_loop` answers with.

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
                found.append(Loop(start, start + fitted))

    return tuple(dict.fromkeys(found))


def detect_loop(
    signal: Signal,
    sample_rate: int,
    config: LoopConfig,
) -> Loop | None:
    """The loop to store when one is wanted and nothing chooses among the alternatives.

    Answers with the first of :func:`loop_candidates` -- the shortest accepted loop placed right after
    the attack -- so a caller needing one loop has the cheapest one the material supports. Returns
    ``None`` where the material supports none.
    """
    candidates = loop_candidates(signal, sample_rate, config)
    return candidates[0] if candidates else None


def crossfade_loop(signal: Signal, loop: Loop, *, fade_len: int) -> Signal:
    """Blend the seam so the loop wraps smoothly; returns a copy with ``[end-fade, end)`` rewritten.

    The last ``fade`` frames before ``loop.end`` are ramped from themselves toward the frames that
    precede ``loop.start``, so ``signal[end-1]`` lands on ``signal[start-1]`` -- making the wrap into
    ``signal[start]`` continuous. Needs ``loop.start >= fade``; a smaller fade is used otherwise.
    """
    fade = min(fade_len, loop.start, loop.length)
    if fade <= 0:
        return np.asarray(signal, dtype=np.float64)

    out = np.array(signal, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, fade, endpoint=True)
    approaching_end = signal[loop.end - fade : loop.end]
    preceding_start = signal[loop.start - fade : loop.start]
    out[loop.end - fade : loop.end] = (1.0 - ramp) * approaching_end + ramp * preceding_start
    return out


def _seam_step(signal: Signal, loop: Loop) -> float:
    """The jump the wrap makes, in units of the typical frame-to-frame step inside the loop region."""
    region = signal[loop.start : loop.end]
    if region.size < 2:
        return 0.0

    typical = float(np.mean(np.abs(np.diff(region))))
    return float(abs(region[0] - region[-1])) / max(typical, _STEP_FLOOR)


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
    config: LoopConfig,
) -> LoopQuality:
    """Measure what storing ``loop`` costs: the seam it wraps on, and the timbre it settles into.

    The seam is read after :func:`crossfade_loop` has blended it, which is the waveform a player wraps.
    The material a loop stands in for is the steady region past its end -- the stretch a looped sample
    stops storing -- so a loop taken from a part of the note that has moved on in timbre reports the
    distance. A loop reaching the end of the steady region stands in for less than one analysis window
    and reports a distance of 0.0.
    """
    faded = crossfade_loop(signal, loop, fade_len=round(config.crossfade_s * sample_rate))
    _, tail = _steady_bounds(signal, sample_rate, config)
    return LoopQuality(
        seam_step=_seam_step(faded, loop),
        spectral_distance=_spectral_distance(faded[loop.start : loop.end], signal[loop.end : tail]),
    )
