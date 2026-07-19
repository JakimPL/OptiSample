"""Loop-point detection and seam crossfade, so a short stored sample can sustain a long note.

A decaying one-shot must store every frame it will ever play, but a *sustained* tone is nearly
periodic in its steady region: store the attack plus a few periods of that region and a tracker can
repeat (loop) them for as long as the note is held. The bytes saved are the entire sustain tail; what
we pay is that the loop is static (it cannot evolve the way the real sustain does) plus whatever
discontinuity the loop seam introduces -- which a crossfade hides.

Detection has three parts:

* **Period estimate** by FFT autocorrelation over the steady region (O(n log n), so it is cheap even
  on multi-second recordings). A weak autocorrelation peak means the material is not periodic enough
  to loop (e.g. a decaying piano), and detection declines -- the caller then stores the sample whole.
* **Loop region** = an integer number of periods ending near the end of the steady region, its start
  snapped to an ascending zero crossing so the wrap lands mid-slope in phase.
* **Crossfade** across the seam: the frames approaching the loop end are blended toward the frames
  that precede the loop start, so ``signal[end-1]`` meets ``signal[start-1]`` and the wrap into
  ``signal[start]`` is as smooth as the original signal already was there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]

_MIN_HZ = 40.0  # lowest fundamental we look for (below this, a "loop" would be longer than most sustains)
_MAX_HZ = 2_000.0
_MIN_CORRELATION = 0.3  # peak autocorrelation below this = not periodic enough to loop
_DEFAULT_MIN_PERIODS = 3  # a loop shorter than this beats/buzzes audibly
_DEFAULT_MIN_LOOP_S = 0.05
_ATTACK_SKIP_S = 0.05  # skip the onset when estimating the period and placing the loop
_TAIL_SKIP_S = 0.02
_MAX_ESTIMATION_S = 1.0  # cap the autocorrelation window; a second of steady tone is plenty
DEFAULT_CROSSFADE_S = 0.01


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


def _estimate_period(signal: Signal, sample_rate: int) -> int | None:
    """Fundamental period in frames from the strongest autocorrelation peak in the pitched band."""
    if signal.size < 8:
        return None
    corr = _autocorrelation(signal)
    low = max(1, int(sample_rate / _MAX_HZ))
    high = min(signal.size - 1, int(sample_rate / _MIN_HZ))
    if high <= low:
        return None
    lag = int(np.argmax(corr[low : high + 1])) + low
    if corr[lag] < _MIN_CORRELATION:
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


def detect_loop(
    signal: Signal,
    sample_rate: int,
    *,
    min_periods: int = _DEFAULT_MIN_PERIODS,
    min_loop_s: float = _DEFAULT_MIN_LOOP_S,
) -> Loop | None:
    """Find a forward loop in the steady region of ``signal``, or ``None`` if it is not periodic enough."""
    total = signal.size
    attack = int(_ATTACK_SKIP_S * sample_rate)
    tail = total - int(_TAIL_SKIP_S * sample_rate)
    if tail - attack < 8:
        return None
    window = signal[attack : min(tail, attack + int(_MAX_ESTIMATION_S * sample_rate))]
    period = _estimate_period(window, sample_rate)
    if period is None:
        return None

    wanted = max(min_periods * period, int(round(min_loop_s * sample_rate)))
    loop_len = max(min_periods, int(round(wanted / period))) * period

    # Place the loop right after the attack and keep it short: we store [0, loop.end) and drop the
    # whole sustain tail past it, so the bytes saved are the tail -- that is the point of looping.
    start = _snap_ascending_zero(signal, attack, radius=period)
    if start + loop_len > tail:  # steady region too short for the wanted loop: take what fits
        loop_len = ((tail - start) // period) * period
        if loop_len < min_periods * period:
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
