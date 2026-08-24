from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.timebase import seconds_to_frames

Signal = NDArray[np.float64]

_QUIET: Final = 1e-12  # amplitude a reading treats as silence, which has no onset to place
_START: Final = 0  # the frame a silent recording places its onset at
_SHARE: Final = 0.1  # share of its own peak a recording is placed at, which is where the attack begins
_RISE_LOW: Final = 0.1  # share of the attack's peak a rise is timed from
_RISE_HIGH: Final = 0.9  # share of it the rise is timed to
_SMOOTH_S: Final = 0.0005  # stretch the amplitude is averaged over, one period of the top of the band
_PRE_S: Final = 0.01  # stretch ahead of an onset an attack is read with
_SPAN_S: Final = 0.06  # stretch past it the attack is read over


@dataclass(frozen=True)
class OnsetReading:
    """How an attack is placed and how much of it is read: the thresholds and the stretches around it.

    ``share`` places the onset at the frame a recording first reaches that much of its own peak, and
    ``rise_low`` and ``rise_high`` bound the stretch a rise is timed across. ``smooth_s`` is what the
    amplitude is averaged over first, short enough to leave a transient standing and long enough for one
    period of the waveform to average away. ``pre_s`` and ``span_s`` are how far either side of the onset
    the attack is read, which is the stretch a listener places a sound by.
    """

    share: float
    rise_low: float
    rise_high: float
    smooth_s: float
    pre_s: float
    span_s: float


def smoothed_amplitude(signal: Signal, sample_rate: int, smooth_s: float) -> Signal:
    """The amplitude ``signal`` holds, averaged over ``smooth_s``, which is what an attack is read off."""
    window = max(1, seconds_to_frames(smooth_s, sample_rate))
    kernel = np.ones(window, dtype=np.float64) / window
    return np.asarray(np.convolve(np.abs(signal), kernel, mode="same"), dtype=np.float64)


def onset_frame(signal: Signal, sample_rate: int, reading: OnsetReading) -> int:
    """The frame ``signal`` first reaches ``reading.share`` of its own peak at, which is where it begins.

    Read off each signal separately, so a stored copy the codec delayed is placed by its own attack and a
    comparison between the two stays about the shape of that attack rather than the delay.
    """
    amplitude = smoothed_amplitude(signal, sample_rate, reading.smooth_s)
    peak = float(amplitude.max()) if amplitude.size else 0.0
    if peak <= _QUIET:
        return _START

    reached = np.flatnonzero(amplitude >= reading.share * peak)
    return int(reached[0]) if reached.size else _START


def onset_window(signal: Signal, sample_rate: int, reading: OnsetReading) -> Signal:
    """The stretch around ``signal``'s own onset an attack is read over: a little before it, and the rise."""
    at = onset_frame(signal, sample_rate, reading)
    low = max(_START, at - seconds_to_frames(reading.pre_s, sample_rate))
    return np.asarray(
        signal[low : low + seconds_to_frames(reading.pre_s + reading.span_s, sample_rate)],
        dtype=np.float64,
    )


def attack_seconds(signal: Signal, sample_rate: int, reading: OnsetReading) -> float:
    """How long ``signal`` takes to rise across its attack, in seconds.

    Timed from where the smoothed amplitude first reaches ``rise_low`` of the attack's own peak to where
    it first reaches ``rise_high``, both inside the window the onset places
    (:func:`onset_window`), which keeps the reading on the attack rather than on anything the recording
    does later. A recording holding no level to speak of reads ``0.0``.

    This is the reading that says whether a level curve can carry a recording at all: a curve is written on
    a tick grid, so an attack running shorter than a tick is a level event the grid has no room to state.
    What it has to separate is therefore the brief from the unhurried, and it is bounded by the window it
    is taken over: material still rising at the end of that window reads as the whole of it.
    """
    amplitude = smoothed_amplitude(onset_window(signal, sample_rate, reading), sample_rate, reading.smooth_s)
    peak = float(amplitude.max()) if amplitude.size else 0.0
    if peak <= _QUIET:
        return 0.0

    low = np.flatnonzero(amplitude >= reading.rise_low * peak)
    high = np.flatnonzero(amplitude >= reading.rise_high * peak)
    if not low.size or not high.size:
        return 0.0

    return max(0, int(high[0]) - int(low[0])) / sample_rate


def crest_db(signal: Signal, sample_rate: int, reading: OnsetReading) -> float:
    """How far an attack's peak stands above the power it carries, in decibels.

    A transient is a peak a frame-averaged spectrum spreads out, so the crest states whether a stored copy
    still holds one. Being a ratio taken inside one window, it reads the same at any level.
    """
    window = onset_window(signal, sample_rate, reading)
    power = float(np.sqrt(np.mean(np.square(window)))) if window.size else 0.0
    peak = float(np.max(np.abs(window))) if window.size else 0.0
    return 20.0 * float(np.log10(max(peak, _QUIET) / max(power, _QUIET)))


ATTACK_READING: Final = OnsetReading(
    share=_SHARE,
    rise_low=_RISE_LOW,
    rise_high=_RISE_HIGH,
    smooth_s=_SMOOTH_S,
    pre_s=_PRE_S,
    span_s=_SPAN_S,
)
