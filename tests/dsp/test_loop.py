from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.loop import LoopConfig
from optisample.dsp.loop import (
    _QUALITY_FFT,
    Loop,
    _autocorrelation,
    _estimate_period,
    crossfade_loop,
    loop_candidates,
    loop_quality,
    shortest_loop_frames,
)

SR = 8_000
FREQ = 200.0
PERIOD = int(round(SR / FREQ))  # 40 frames
_FADE = 80  # 10 ms of blend at this rate, which is what the bundled seam asks for


def _sine(n: int, freq: float = FREQ, amp: float = 0.8) -> NDArray[np.float64]:
    t = np.arange(n, dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def _cheapest(signal: NDArray[np.float64], config: GeometryConfig) -> Loop | None:
    """The front of the ladder a settlement climbs: the earliest, shortest loop the geometry allows."""
    candidates = sorted(loop_candidates(signal, SR, config), key=lambda loop: (loop.end, loop.start))
    return candidates[0] if candidates else None


def test_estimate_period_recovers_the_fundamental(geometry_config: GeometryConfig) -> None:
    assert _estimate_period(_sine(SR), SR, geometry_config) == pytest.approx(PERIOD, abs=1)


def test_the_cheapest_candidate_on_a_pure_tone_is_an_integer_number_of_periods(geometry_config: GeometryConfig) -> None:
    loop = _cheapest(_sine(SR), geometry_config)
    assert loop is not None
    assert loop.length % PERIOD == 0
    assert loop.length >= 3 * PERIOD  # at least the minimum periods
    assert loop.start >= int(0.05 * SR) - PERIOD  # after the skipped attack (within a period of it)
    assert loop.end <= SR


def test_the_cheapest_candidate_declines_on_noise(geometry_config: GeometryConfig) -> None:
    rng = np.random.default_rng(0)
    assert _cheapest(rng.standard_normal(SR), geometry_config) is None


def test_a_decaying_tone_loops_where_it_holds_a_period(geometry_config: GeometryConfig) -> None:
    # A struck note is pitched throughout its decay, and the level its loop settles on is brought down
    # outside the PCM (optisample.dsp.decay), so it is stored as attack plus loop like any other tone.
    decay = np.exp(-np.arange(2 * SR, dtype=np.float64) / (0.6 * SR))
    loop = _cheapest(decay * _sine(2 * SR), geometry_config)

    assert loop is not None
    assert loop.length % PERIOD == 0


def test_the_cheapest_candidate_declines_on_a_too_short_signal(geometry_config: GeometryConfig) -> None:
    assert _cheapest(_sine(4), geometry_config) is None


def test_looping_a_tone_reproduces_its_continuation(geometry_config: GeometryConfig) -> None:
    signal = _sine(SR)
    loop = _cheapest(signal, geometry_config)
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


def test_estimate_period_rejects_short_and_degenerate_bands(geometry_config: GeometryConfig) -> None:
    assert _estimate_period(np.zeros(4), SR, geometry_config) is None  # too few frames
    assert _estimate_period(_sine(20), 100_000, geometry_config) is None  # pitched band collapses (high <= low)
    assert _estimate_period(np.zeros(SR), SR, geometry_config) is None  # silence -> no peak above the correlation floor


def test_the_cheapest_candidate_declines_where_the_steady_region_holds_less_than_the_floor(
    geometry_config: GeometryConfig,
) -> None:
    # 400 Hz at 8 kHz -> period 20; a 700-frame tone leaves ~140 steady frames after the attack, far under
    # the 0.5 s floor, so the region has no loop long enough to store.
    assert _cheapest(_sine(700, freq=400.0), geometry_config) is None


def test_the_floor_is_read_in_whole_periods_covering_both_bounds(geometry_config: GeometryConfig) -> None:
    floor_s = shortest_loop_frames(PERIOD, SR, geometry_config) / SR
    assert floor_s >= geometry_config.min_loop_s
    assert shortest_loop_frames(PERIOD, SR, geometry_config) % PERIOD == 0
    assert shortest_loop_frames(SR, SR, geometry_config) == geometry_config.min_periods * SR  # a period past the floor


# --- the candidates a clip chooses among ----------------------------------------------------------


def test_every_candidate_clears_the_floor_and_spans_whole_periods(geometry_config: GeometryConfig) -> None:
    floor = shortest_loop_frames(PERIOD, SR, geometry_config)
    candidates = loop_candidates(_sine(4 * SR), SR, geometry_config)

    assert len(candidates) > 1
    assert all(loop.length >= floor and loop.length % PERIOD == 0 for loop in candidates)
    assert all(loop.length >= round(geometry_config.min_loop_s * SR) for loop in candidates)


def test_the_cheapest_candidate_is_the_one_the_ladder_offers_first(geometry_config: GeometryConfig) -> None:
    signal = _sine(4 * SR)
    ladder = sorted(loop_candidates(signal, SR, geometry_config), key=lambda loop: (loop.end, loop.start))

    assert _cheapest(signal, geometry_config) == ladder[0]


def test_candidates_run_placement_major_so_a_prefix_reaches_both_axes(geometry_config: GeometryConfig) -> None:
    candidates = loop_candidates(_sine(4 * SR), SR, geometry_config)
    starts = [loop.start for loop in candidates]

    assert len(set(starts)) > 1  # placements are spread rather than all landing on the attack skip
    assert candidates[1].start == candidates[0].start  # the same start is offered at each length first
    assert candidates[1].length > candidates[0].length
    assert candidates[2].start > candidates[0].start


def test_candidates_stay_inside_the_steady_region_and_stand_apart(geometry_config: GeometryConfig) -> None:
    signal = _sine(2 * SR)
    attack, tail = int(geometry_config.attack_skip_s * SR), signal.size - int(geometry_config.tail_skip_s * SR)
    candidates = loop_candidates(signal, SR, geometry_config)

    assert len(set(candidates)) == len(candidates)  # placements snapping together are offered once
    assert all(loop.start >= attack - PERIOD for loop in candidates)  # snapping moves a start by a period
    assert all(loop.end <= tail for loop in candidates)


def test_a_length_the_region_lacks_room_for_shrinks_onto_the_whole_periods_that_fit(
    geometry_config: GeometryConfig,
) -> None:
    # A 1 s tone has room for the 0.5 s floor twice over but not for twice the floor from the attack skip.
    signal = _sine(SR)
    tail = signal.size - int(geometry_config.tail_skip_s * SR)
    longest = max(loop_candidates(signal, SR, geometry_config), key=lambda loop: loop.length)

    assert longest.length > shortest_loop_frames(PERIOD, SR, geometry_config)
    assert longest.length % PERIOD == 0
    assert longest.end <= tail


def test_material_a_loop_has_no_purchase_on_offers_no_candidates(geometry_config: GeometryConfig) -> None:
    rng = np.random.default_rng(0)

    assert loop_candidates(rng.standard_normal(SR), SR, geometry_config) == ()


# --- what a loop is worth --------------------------------------------------------------------------


def test_a_crossfaded_seam_reads_as_a_step_the_waveform_itself_could_have_made(geometry_config: GeometryConfig) -> None:
    signal = _sine(4 * SR)
    loop = _cheapest(signal, geometry_config)
    assert loop is not None

    quality = loop_quality(signal, loop, SR, geometry_config, fade_len=_FADE)

    assert quality.seam_step < 2.0  # whole periods from a zero crossing: the wrap is the waveform's own motion


def test_a_loop_holding_a_timbre_the_material_moves_away_from_reports_the_distance(
    geometry_config: GeometryConfig,
) -> None:
    steady = _sine(2 * SR)
    brightened = steady + 0.5 * _sine(2 * SR, freq=5 * FREQ)
    loop = _cheapest(steady, geometry_config)
    assert loop is not None

    held = loop_quality(np.concatenate([steady, steady]), loop, SR, geometry_config, fade_len=_FADE)
    moved = loop_quality(np.concatenate([steady, brightened]), loop, SR, geometry_config, fade_len=_FADE)

    assert moved.spectral_distance > held.spectral_distance


@pytest.mark.parametrize(
    "left_over",
    [
        pytest.param(0, id="a loop reaching the end of the steady region stands in for nothing"),
        pytest.param(_QUALITY_FFT - 1, id="a sliver under one analysis window carries no spectrum to compare"),
    ],
)
def test_a_loop_the_material_barely_outlasts_reports_no_distance(
    geometry_config: GeometryConfig, left_over: int
) -> None:
    signal = _sine(2 * SR)
    tail = signal.size - int(geometry_config.tail_skip_s * SR)
    end = tail - left_over
    reaching = Loop(start=end - shortest_loop_frames(PERIOD, SR, geometry_config), end=end)

    assert loop_quality(signal, reaching, SR, geometry_config, fade_len=_FADE).spectral_distance == 0.0


def test_a_silent_loop_region_reports_no_seam(geometry_config: GeometryConfig) -> None:
    assert loop_quality(np.zeros(SR), Loop(start=100, end=500), SR, geometry_config, fade_len=_FADE).seam_step == 0.0


def test_a_loop_of_one_frame_has_no_step_to_measure_the_seam_against(geometry_config: GeometryConfig) -> None:
    assert loop_quality(_sine(SR), Loop(start=100, end=101), SR, geometry_config, fade_len=_FADE).seam_step == 0.0
