"""Loop-point detection and seam crossfade, so a short stored sample can sustain a long note.

A decaying one-shot must store every frame it will ever play, but a *sustained* tone is nearly
periodic in its steady region: store the attack plus a few periods of that region and a tracker can
repeat (loop) them for as long as the note is held. The bytes saved are the entire sustain tail; what
we pay is that the loop is static (it cannot evolve the way the real sustain does) plus whatever
discontinuity the loop seam introduces -- which a crossfade hides.

Detection has three parts:

* **Sustain check** first: a loop repeats forever, so only *level* material may be looped. If the
  steady region's energy decays (a struck piano note), detection declines and the caller stores the
  sample whole.
* **Period estimate** by FFT autocorrelation over the steady region (O(n log n), so it is cheap even
  on multi-second recordings). A weak autocorrelation peak (not periodic enough) also declines.
* **Loop region** = an integer number of periods ending near the end of the steady region, its start
  snapped to an ascending zero crossing so the wrap lands mid-slope in phase.
* **Crossfade** across the seam: the frames approaching the loop end are blended toward the frames
  that precede the loop start, so ``signal[end-1]`` meets ``signal[start-1]`` and the wrap into
  ``signal[start]`` is as smooth as the original signal already was there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.dsp import LoopConfig

Signal = NDArray[np.float64]

_MIN_STEADY_FRAMES: Final = 8  # a steady region shorter than this cannot be analysed for a period.
_MIN_SUSTAIN_FRAMES: Final = 6  # below this the decay check has too few frames to judge sustain.
_ENERGY_THIRDS: Final = 3  # the sustain check compares the region's first third against its last third.


@dataclass(frozen=True)
class Loop:
    """A forward loop over the half-open frame range ``[start, end)`` (playback wraps ``end`` -> ``start``)."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _autocorrelation(signal: Signal) -> Signal:
    """Unbiased-enough autocorrelation via FFT (lags ``0..n-1``), normalized so lag 0 == 1."""
    centered = signal - float(np.mean(signal))
    n = centered.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(centered, size)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    if corr[0] <= 0.0:
        return np.zeros(n, dtype=np.float64)
    return np.asarray(corr / corr[0], dtype=np.float64)


def _estimate_period(signal: Signal, sample_rate: int, config: LoopConfig) -> int | None:
    """Fundamental period in frames from the strongest autocorrelation peak in the pitched band.

    Searches lags from ``sample_rate / max_hz`` (the shortest period the band admits) to
    ``sample_rate / min_hz`` (the longest). Returns ``None`` when the signal spans fewer than
    ``_MIN_STEADY_FRAMES``, the band holds no lag at this rate, or the strongest peak stays below
    ``config.min_correlation`` -- each the mark of material too aperiodic to loop.
    """
    if signal.size < _MIN_STEADY_FRAMES:
        return None
    corr = _autocorrelation(signal)
    low = max(1, int(sample_rate / config.max_hz))
    high = min(signal.size - 1, int(sample_rate / config.min_hz))
    if high <= low:
        return None
    lag = int(np.argmax(corr[low : high + 1])) + low
    if corr[lag] < config.min_correlation:
        return None
    return lag


def _is_sustained(region: Signal, decay_ratio: float) -> bool:
    """Whether ``region`` holds a level (loopable) tone rather than a decaying one.

    A looped sample repeats its region forever, so looping a decay (a struck piano note) would make
    it ring at a constant level instead of dying away -- wrong. We compare the energy of the region's
    last third to its first third; a sustain stays roughly level, a decay drops well below it.
    """
    if region.size < _MIN_SUSTAIN_FRAMES:
        return False
    third = region.size // _ENERGY_THIRDS
    early = float(np.sqrt(np.mean(region[:third] ** 2)))
    late = float(np.sqrt(np.mean(region[-third:] ** 2)))
    return early > 0.0 and late / early >= decay_ratio


def _snap_ascending_zero(signal: Signal, index: int, radius: int) -> int:
    """Nearest ascending zero crossing (``-`` -> ``+``) to ``index`` within ``radius`` (else ``index``)."""
    low = max(1, index - radius)
    high = min(signal.size - 1, index + radius)
    best, best_distance = index, radius + 1
    for candidate in range(low, high + 1):
        if signal[candidate - 1] <= 0.0 < signal[candidate] and abs(candidate - index) < best_distance:
            best, best_distance = candidate, abs(candidate - index)
    return best


def _steady_region(signal: Signal, sample_rate: int, config: LoopConfig) -> tuple[int, int]:
    """The ``[attack, tail)`` frame window to analyse: past the onset transient, before the release."""
    attack = int(config.attack_skip_s * sample_rate)
    tail = signal.size - int(config.tail_skip_s * sample_rate)
    return attack, tail


def _loop_length(period: int, sample_rate: int, config: LoopConfig) -> int:
    """The wanted loop length: whole periods covering at least ``min_periods`` and ``min_loop_s`` worth."""
    wanted = max(config.min_periods * period, int(round(config.min_loop_s * sample_rate)))
    return max(config.min_periods, int(round(wanted / period))) * period


def _fit_loop_to_region(start: int, loop_len: int, tail: int, period: int, config: LoopConfig) -> int | None:
    """Shrink ``loop_len`` to the whole periods that fit before ``tail``; ``None`` if too few remain."""
    if start + loop_len <= tail:
        return loop_len
    fitted = ((tail - start) // period) * period
    return fitted if fitted >= config.min_periods * period else None


def detect_loop(signal: Signal, sample_rate: int, config: LoopConfig) -> Loop | None:
    """Find a forward loop in the steady region of ``signal``, or ``None`` if it cannot be looped.

    Analyses the steady region between the attack skip (past the onset transient) and the tail skip
    (before the release). The region must sustain a level tone -- a loop repeats its content forever, so
    a decaying region would ring on at a constant level -- and carry a fundamental period, found by
    :func:`_estimate_period`. The loop spans a whole number of periods (at least ``min_periods`` and
    ``min_loop_s`` worth), begins at an ascending zero crossing just after the attack so the wrap lands
    mid-slope in phase, and keeps only ``[0, loop.end)`` -- storing the attack plus one loop region while
    dropping the sustain tail, which is where the bytes are saved. A region shorter than the wanted loop
    uses as many whole periods as fit.

    Returns ``None`` when the steady region is too short to analyse, decays instead of sustaining, has no
    reliable period, or leaves room for fewer than ``min_periods`` whole periods.
    """
    total = signal.size
    attack, tail = _steady_region(signal, sample_rate, config)
    if tail - attack < _MIN_STEADY_FRAMES:
        return None
    if not _is_sustained(signal[attack:tail], config.sustain_decay_ratio):
        return None
    window = signal[attack : min(tail, attack + int(config.max_estimation_s * sample_rate))]
    period = _estimate_period(window, sample_rate, config)
    if period is None:
        return None

    start = _snap_ascending_zero(signal, attack, radius=period)
    loop_len = _fit_loop_to_region(start, _loop_length(period, sample_rate, config), tail, period, config)
    if loop_len is None:
        return None
    end = start + loop_len
    if end > total or end <= start:
        return None
    return Loop(start, end)


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
