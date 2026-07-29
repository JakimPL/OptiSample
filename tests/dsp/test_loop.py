from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.codec import LoopConfig
from optisample.dsp.loop import (
    _MIN_SUSTAIN_FRAMES,
    _QUALITY_FFT,
    Loop,
    _autocorrelation,
    _estimate_period,
    _is_sustained,
    crossfade_loop,
    detect_loop,
    loop_candidates,
    loop_quality,
    shortest_loop_frames,
)

SR = 8_000
FREQ = 200.0
PERIOD = int(round(SR / FREQ))  # 40 frames


def _sine(n: int, freq: float = FREQ, amp: float = 0.8) -> NDArray[np.float64]:
    t = np.arange(n, dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def test_estimate_period_recovers_the_fundamental(loop_config: LoopConfig) -> None:
    assert _estimate_period(_sine(SR), SR, loop_config) == pytest.approx(PERIOD, abs=1)


def test_detect_loop_on_a_pure_tone_is_an_integer_number_of_periods(loop_config: LoopConfig) -> None:
    loop = detect_loop(_sine(SR), SR, loop_config)
    assert loop is not None
    assert loop.length % PERIOD == 0
    assert loop.length >= 3 * PERIOD  # at least the minimum periods
    assert loop.start >= int(0.05 * SR) - PERIOD  # after the skipped attack (within a period of it)
    assert loop.end <= SR


def test_detect_loop_declines_on_noise(loop_config: LoopConfig) -> None:
    rng = np.random.default_rng(0)
    assert detect_loop(rng.standard_normal(SR), SR, loop_config) is None


def test_detect_loop_declines_on_a_decaying_tone(loop_config: LoopConfig) -> None:
    # A struck note is pitched but decays; looping it would make it ring forever at the loop's level.
    decay = np.exp(-np.arange(SR, dtype=np.float64) / (0.15 * SR))
    assert detect_loop(decay * _sine(SR), SR, loop_config) is None


def test_detect_loop_declines_on_a_too_short_signal(loop_config: LoopConfig) -> None:
    assert detect_loop(_sine(4), SR, loop_config) is None


def test_looping_a_tone_reproduces_its_continuation(loop_config: LoopConfig) -> None:
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


def test_is_sustained_rejects_a_region_too_short_to_judge() -> None:
    # Below the minimum span there are too few frames to compare early- vs late-energy, so it cannot loop.
    tiny = np.ones(_MIN_SUSTAIN_FRAMES - 1, dtype=np.float64)
    assert _is_sustained(tiny, decay_ratio=0.5) is False


def test_estimate_period_rejects_short_and_degenerate_bands(loop_config: LoopConfig) -> None:
    assert _estimate_period(np.zeros(4), SR, loop_config) is None  # too few frames
    assert _estimate_period(_sine(20), 100_000, loop_config) is None  # pitched band collapses (high <= low)
    assert _estimate_period(np.zeros(SR), SR, loop_config) is None  # silence -> no peak above the correlation floor


def test_detect_loop_declines_where_the_steady_region_holds_less_than_the_floor(loop_config: LoopConfig) -> None:
    # 400 Hz at 8 kHz -> period 20; a 700-frame tone leaves ~140 steady frames after the attack, far under
    # the 0.5 s floor, so the region has no loop long enough to store.
    assert detect_loop(_sine(700, freq=400.0), SR, loop_config) is None


def test_the_floor_is_read_in_whole_periods_covering_both_bounds(loop_config: LoopConfig) -> None:
    floor_s = shortest_loop_frames(PERIOD, SR, loop_config) / SR
    assert floor_s >= loop_config.min_loop_s
    assert shortest_loop_frames(PERIOD, SR, loop_config) % PERIOD == 0
    assert shortest_loop_frames(SR, SR, loop_config) == loop_config.min_periods * SR  # a period past the floor


# --- the candidates a clip chooses among ----------------------------------------------------------


def test_every_candidate_clears_the_floor_and_spans_whole_periods(loop_config: LoopConfig) -> None:
    floor = shortest_loop_frames(PERIOD, SR, loop_config)
    candidates = loop_candidates(_sine(4 * SR), SR, loop_config)

    assert len(candidates) > 1
    assert all(loop.length >= floor and loop.length % PERIOD == 0 for loop in candidates)
    assert all(loop.length >= round(loop_config.min_loop_s * SR) for loop in candidates)


def test_the_first_candidate_is_the_one_a_single_detection_answers_with(loop_config: LoopConfig) -> None:
    signal = _sine(4 * SR)

    assert detect_loop(signal, SR, loop_config) == loop_candidates(signal, SR, loop_config)[0]


def test_candidates_run_placement_major_so_a_prefix_reaches_both_axes(loop_config: LoopConfig) -> None:
    candidates = loop_candidates(_sine(4 * SR), SR, loop_config)
    starts = [loop.start for loop in candidates]

    assert len(set(starts)) > 1  # placements are spread rather than all landing on the attack skip
    assert candidates[1].start == candidates[0].start  # the same start is offered at each length first
    assert candidates[1].length > candidates[0].length
    assert candidates[2].start > candidates[0].start


def test_candidates_stay_inside_the_steady_region_and_stand_apart(loop_config: LoopConfig) -> None:
    signal = _sine(2 * SR)
    attack, tail = int(loop_config.attack_skip_s * SR), signal.size - int(loop_config.tail_skip_s * SR)
    candidates = loop_candidates(signal, SR, loop_config)

    assert len(set(candidates)) == len(candidates)  # placements snapping together are offered once
    assert all(loop.start >= attack - PERIOD for loop in candidates)  # snapping moves a start by a period
    assert all(loop.end <= tail for loop in candidates)


def test_a_length_the_region_lacks_room_for_shrinks_onto_the_whole_periods_that_fit(
    loop_config: LoopConfig,
) -> None:
    # A 1 s tone has room for the 0.5 s floor twice over but not for twice the floor from the attack skip.
    signal = _sine(SR)
    tail = signal.size - int(loop_config.tail_skip_s * SR)
    longest = max(loop_candidates(signal, SR, loop_config), key=lambda loop: loop.length)

    assert longest.length > shortest_loop_frames(PERIOD, SR, loop_config)
    assert longest.length % PERIOD == 0
    assert longest.end <= tail


def test_material_a_loop_has_no_purchase_on_offers_no_candidates(loop_config: LoopConfig) -> None:
    rng = np.random.default_rng(0)

    assert loop_candidates(rng.standard_normal(SR), SR, loop_config) == ()


# --- what a loop is worth --------------------------------------------------------------------------


def test_a_crossfaded_seam_reads_as_a_step_the_waveform_itself_could_have_made(loop_config: LoopConfig) -> None:
    signal = _sine(4 * SR)
    loop = detect_loop(signal, SR, loop_config)
    assert loop is not None

    quality = loop_quality(signal, loop, SR, loop_config)

    assert quality.seam_step < 2.0  # whole periods from a zero crossing: the wrap is the waveform's own motion


def test_a_loop_holding_a_timbre_the_material_moves_away_from_reports_the_distance(
    loop_config: LoopConfig,
) -> None:
    steady = _sine(2 * SR)
    brightened = steady + 0.5 * _sine(2 * SR, freq=5 * FREQ)
    loop = detect_loop(steady, SR, loop_config)
    assert loop is not None

    held = loop_quality(np.concatenate([steady, steady]), loop, SR, loop_config)
    moved = loop_quality(np.concatenate([steady, brightened]), loop, SR, loop_config)

    assert moved.spectral_distance > held.spectral_distance


@pytest.mark.parametrize(
    "left_over",
    [
        pytest.param(0, id="a loop reaching the end of the steady region stands in for nothing"),
        pytest.param(_QUALITY_FFT - 1, id="a sliver under one analysis window carries no spectrum to compare"),
    ],
)
def test_a_loop_the_material_barely_outlasts_reports_no_distance(loop_config: LoopConfig, left_over: int) -> None:
    signal = _sine(2 * SR)
    tail = signal.size - int(loop_config.tail_skip_s * SR)
    end = tail - left_over
    reaching = Loop(start=end - shortest_loop_frames(PERIOD, SR, loop_config), end=end)

    assert loop_quality(signal, reaching, SR, loop_config).spectral_distance == 0.0


def test_a_silent_loop_region_reports_no_seam(loop_config: LoopConfig) -> None:
    assert loop_quality(np.zeros(SR), Loop(start=100, end=500), SR, loop_config).seam_step == 0.0


def test_a_loop_of_one_frame_has_no_step_to_measure_the_seam_against(loop_config: LoopConfig) -> None:
    assert loop_quality(_sine(SR), Loop(start=100, end=101), SR, loop_config).seam_step == 0.0
