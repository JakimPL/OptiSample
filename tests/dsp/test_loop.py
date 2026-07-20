from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.loop import Loop, _autocorrelation, _estimate_period, crossfade_loop, detect_loop

SR = 8_000
FREQ = 200.0
PERIOD = int(round(SR / FREQ))  # 40 frames


def _sine(n: int, freq: float = FREQ, amp: float = 0.8) -> NDArray[np.float64]:
    t = np.arange(n, dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def test_estimate_period_recovers_the_fundamental(loop_config) -> None:
    assert _estimate_period(_sine(SR), SR, loop_config) == pytest.approx(PERIOD, abs=1)


def test_detect_loop_on_a_pure_tone_is_an_integer_number_of_periods(loop_config) -> None:
    loop = detect_loop(_sine(SR), SR, loop_config)
    assert loop is not None
    assert loop.length % PERIOD == 0
    assert loop.length >= 3 * PERIOD  # at least the minimum periods
    assert loop.start >= int(0.05 * SR) - PERIOD  # after the skipped attack (within a period of it)
    assert loop.end <= SR


def test_detect_loop_declines_on_noise(loop_config) -> None:
    rng = np.random.default_rng(0)
    assert detect_loop(rng.standard_normal(SR), SR, loop_config) is None


def test_detect_loop_declines_on_a_decaying_tone(loop_config) -> None:
    # A struck note is pitched but decays; looping it would make it ring forever at the loop's level.
    decay = np.exp(-np.arange(SR, dtype=np.float64) / (0.15 * SR))
    assert detect_loop(decay * _sine(SR), SR, loop_config) is None


def test_detect_loop_declines_on_a_too_short_signal(loop_config) -> None:
    assert detect_loop(_sine(4), SR, loop_config) is None


def test_looping_a_tone_reproduces_its_continuation(loop_config) -> None:
    signal = _sine(SR)
    loop = detect_loop(signal, SR, loop_config)
    assert loop is not None
    extend = 5 * loop.length
    segment = signal[loop.start : loop.end]
    looped = np.concatenate([signal[: loop.end], np.tile(segment, extend // loop.length + 1)])[: loop.end + extend]
    truth = _sine(loop.end + extend)
    assert float(np.max(np.abs(looped - truth))) < 1e-9  # integer periods from a zero crossing -> exact


def test_crossfade_pulls_the_seam_toward_continuity() -> None:
    # An amplitude ramp makes the frame before the loop end differ from the frame before the loop start,
    # so the raw wrap has a step; the crossfade should shrink that step.
    n = SR
    ramp = np.linspace(0.3, 1.0, n)
    signal = ramp * np.sin(2.0 * np.pi * FREQ * np.arange(n) / SR)
    loop = Loop(start=10 * PERIOD, end=18 * PERIOD)
    raw_gap = abs(signal[loop.end - 1] - signal[loop.start - 1])
    faded = crossfade_loop(signal, loop, fade_len=PERIOD)
    faded_gap = abs(faded[loop.end - 1] - signal[loop.start - 1])
    assert faded_gap < raw_gap
    assert np.array_equal(faded[: loop.end - PERIOD], signal[: loop.end - PERIOD])  # only the seam changed


def test_crossfade_is_a_noop_without_room_before_the_loop() -> None:
    signal = _sine(SR)
    loop = Loop(start=0, end=4 * PERIOD)  # no frames precede the start to blend from
    assert np.array_equal(crossfade_loop(signal, loop, fade_len=PERIOD), signal)


# --- degenerate inputs ---------------------------------------------------------------------------


def test_autocorrelation_of_silence_is_zero() -> None:
    assert np.array_equal(_autocorrelation(np.zeros(64)), np.zeros(64))


def test_estimate_period_rejects_short_and_degenerate_bands(loop_config) -> None:
    assert _estimate_period(np.zeros(4), SR, loop_config) is None  # too few frames
    assert _estimate_period(_sine(20), 100_000, loop_config) is None  # pitched band collapses (high <= low)
    assert _estimate_period(np.zeros(SR), SR, loop_config) is None  # silence -> no peak above the correlation floor


def test_detect_loop_shrinks_to_the_periods_that_fit_a_short_steady_region(loop_config) -> None:
    # 400 Hz at 8 kHz -> period 20; a 700-frame tone leaves only ~140 steady frames after the attack,
    # too few for the default loop, so detection takes the whole periods that do fit.
    loop = detect_loop(_sine(700, freq=400.0), SR, loop_config)
    assert loop is not None
    assert loop.length % 20 == 0 and 3 * 20 <= loop.length < int(0.05 * SR)


def test_detect_loop_declines_when_too_few_periods_fit(loop_config) -> None:
    # only ~2 periods fit, below the 3-period minimum
    assert detect_loop(_sine(610, freq=400.0), SR, loop_config) is None
