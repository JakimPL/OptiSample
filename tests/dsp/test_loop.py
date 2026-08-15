from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.loop import EnvelopeConfig, GeometryConfig, LoopConfig, SeamConfig
from optisample.dsp.envelope import LevelReading, level_reading, local_level_over
from optisample.dsp.level import gain_to_db
from optisample.dsp.loop import (
    _QUALITY_FFT,
    Loop,
    _estimate_period,
    _searched_lags,
    crossfade_loop,
    level_loop,
    loop_at_rate,
    loop_decline,
    loop_quality,
    loop_search,
    prepare_loop,
    seam_frames,
    shortest_loop_frames,
)
from optisample.dsp.spectral import band_energy
from tests.conftest import recorded

SR = 8_000
FREQ = 200.0
PERIOD = int(round(SR / FREQ))  # 40 frames
UNEVEN_FREQ = 210.0  # a period of 38.1 frames, which no whole number of frames lands on
UNEVEN_PERIOD = SR / UNEVEN_FREQ


def _sine(n: int, freq: float = FREQ, amp: float = 0.8) -> NDArray[np.float64]:
    t = np.arange(n, dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


CROSSOVERS: Final = (250.0, 1_000.0)  # a bank a tone at FREQ and its partials fall on either side of
CROSSOVER_OCTAVES: Final = 0.5
LOW_PARTIAL: Final = 200.0  # 50 whole cycles across the 2000-frame loop below, so it comes round in phase
HIGH_PARTIAL: Final = 1_485.0  # 371.25 cycles across it, so it comes round a quarter turn from where it left
LOW_BAND: Final = (0.0, 250.0)
HIGH_BAND: Final = (1_000.0, SR / 2.0)


def _seam(fade_share: float) -> SeamConfig:
    """A seam asking for a stated share of the loop, so a test reads one blend rather than the shipped one."""
    return SeamConfig(
        fade_share=fade_share,
        min_fade_s=0.0,
        crossovers_hz=CROSSOVERS,
        crossover_octaves=CROSSOVER_OCTAVES,
    )


def _broadband_seam(fade_share: float) -> SeamConfig:
    """A seam weighing the whole spectrum as one band, which is one law serving every partial at once."""
    return SeamConfig(
        fade_share=fade_share,
        min_fade_s=0.0,
        crossovers_hz=(),
        crossover_octaves=CROSSOVER_OCTAVES,
    )


def _band_level_db(signal: NDArray[np.float64], band: tuple[float, float]) -> float:
    """The level ``signal`` holds inside ``band``, in decibels."""
    return gain_to_db(float(np.sqrt(band_energy(signal, SR, band[0], band[1]))))


def _reading(config: EnvelopeConfig, root_hz: float = FREQ) -> LevelReading:
    """How a note played at ``root_hz`` has its level read, which every levelling below is taken under."""
    return level_reading(SR, config, root_hz)


def _cheapest(signal: NDArray[np.float64], config: LoopConfig, root_hz: float = FREQ) -> Loop | None:
    """The front of the ladder a settlement climbs: the earliest, shortest loop the geometry allows."""
    candidates = sorted(loop_search(signal, SR, config, root_hz).candidates, key=lambda loop: (loop.end, loop.start))
    return candidates[0] if candidates else None


def _earliest_start(config: LoopConfig) -> int:
    """The frame a steady tone opens its window at, which is where its candidates may begin.

    Material holding one sound settles from the moment it starts, so what places the window is the blend
    a wrap reaches back for: ``min_fade_s`` of material before the region.
    """
    return round(config.seam.min_fade_s * SR)


def _wrap_mismatch(signal: NDArray[np.float64], loop: Loop) -> float:
    """How far the material after the loop end stands from the material a wrap lands back on."""
    window = round(UNEVEN_PERIOD)
    return float(np.max(np.abs(signal[loop.end : loop.end + window] - signal[loop.start : loop.start + window])))


def _level_of(signal: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(signal**2)))


def test_estimate_period_recovers_the_fundamental(geometry_config: GeometryConfig) -> None:
    assert _estimate_period(_sine(SR), SR, geometry_config, FREQ) == pytest.approx(PERIOD, abs=1)


def test_the_period_is_searched_around_the_pitch_the_note_was_played_at(geometry_config: GeometryConfig) -> None:
    """The lags searched hold the played period and stay well inside the octave above and below it."""
    low, high = _searched_lags(SR, geometry_config, FREQ)

    assert low <= PERIOD <= high
    assert high < 2 * PERIOD  # two rounds of the note sit outside, so the peak read is the note's own
    assert low > PERIOD // 2  # ... and so does half of one


def test_the_pitch_it_is_handed_says_which_repeat_of_the_material_is_read(geometry_config: GeometryConfig) -> None:
    """A tone repeats at every multiple of its period, so the pitch is what states the one a loop spans."""
    signal = _sine(SR)

    assert _estimate_period(signal, SR, geometry_config, FREQ) == pytest.approx(PERIOD, abs=1)
    assert _estimate_period(signal, SR, geometry_config, FREQ / 2) == pytest.approx(2 * PERIOD, abs=1)


def test_a_recording_sounding_below_its_key_is_read_at_the_period_it_holds(
    geometry_config: GeometryConfig,
) -> None:
    """A set filed an octave over what it sounds repeats at twice the period its key names."""
    sounding = _sine(SR, freq=FREQ / 2)

    assert _estimate_period(sounding, SR, geometry_config, FREQ) == pytest.approx(2 * PERIOD, abs=1)


def test_a_tone_is_read_at_its_own_period_rather_than_a_multiple_of_it(
    geometry_config: GeometryConfig,
) -> None:
    """Material repeats at every multiple of its period, so the shallowest octave is the fundamental."""
    assert _estimate_period(_sine(SR), SR, geometry_config, FREQ) == pytest.approx(PERIOD, abs=1)


def test_the_octaves_searched_are_what_reaches_a_recording_sounding_below_its_key(
    geometry_config: GeometryConfig,
) -> None:
    """Asking for no octaves holds the search to the played pitch, which a set sounding lower falls outside."""
    sounding = _sine(SR, freq=FREQ / 2)
    nominal = GeometryConfig.model_validate({**geometry_config.model_dump(), "octaves_below": 0})

    assert _estimate_period(sounding, SR, nominal, FREQ) is None
    assert _estimate_period(sounding, SR, geometry_config, FREQ) is not None


def test_a_period_falling_between_frames_is_read_between_them(geometry_config: GeometryConfig) -> None:
    """The parabola through the peak is what states a lag a whole number of frames lands beside."""
    period = _estimate_period(_sine(SR, freq=UNEVEN_FREQ), SR, geometry_config, UNEVEN_FREQ)

    assert period is not None
    assert period == pytest.approx(UNEVEN_PERIOD, abs=0.1)
    assert abs(period - round(period)) > 0.05  # the frame it would have been read at sits elsewhere


def test_the_cheapest_candidate_on_a_pure_tone_is_an_integer_number_of_periods(loop_config: LoopConfig) -> None:
    loop = _cheapest(_sine(SR), loop_config)
    assert loop is not None
    assert loop.length % PERIOD == 0
    assert loop.length >= 3 * PERIOD  # at least the minimum periods
    assert loop.start >= _earliest_start(loop_config) - PERIOD  # snapping moves a start by a period
    assert loop.end <= SR


def test_the_cheapest_candidate_declines_on_noise(loop_config: LoopConfig) -> None:
    rng = np.random.default_rng(0)
    assert _cheapest(rng.standard_normal(SR), loop_config) is None


def test_a_decaying_tone_loops_where_it_holds_a_period(loop_config: LoopConfig) -> None:
    # A struck note is pitched throughout its decay, and the level its loop settles on is brought down
    # outside the PCM (optisample.dsp.loop.loop_decline), so it is stored as attack plus loop like any tone.
    decay = np.exp(-np.arange(2 * SR, dtype=np.float64) / (0.6 * SR))
    loop = _cheapest(decay * _sine(2 * SR), loop_config)

    assert loop is not None
    assert loop.length % PERIOD == 0


def test_the_cheapest_candidate_declines_on_a_too_short_signal(loop_config: LoopConfig) -> None:
    assert _cheapest(_sine(4), loop_config) is None


def test_looping_a_tone_reproduces_its_continuation(loop_config: LoopConfig) -> None:
    signal = _sine(SR)
    loop = _cheapest(signal, loop_config)
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
    seam = _seam(0.125)
    fade = seam_frames(loop, SR, seam)
    raw_gap = abs(signal[loop.end - 1] - signal[loop.start - 1])

    faded = crossfade_loop(signal, loop, SR, seam)

    assert abs(faded[loop.end - 1] - signal[loop.start - 1]) < raw_gap
    assert np.array_equal(faded[: loop.end - fade], signal[: loop.end - fade])  # only the seam changed


def test_a_loop_the_material_ahead_of_it_holds_no_room_for_wraps_as_it_stands() -> None:
    signal = _sine(SR)
    loop = Loop(start=0, end=4 * PERIOD)  # no frames precede the start to blend from

    assert np.array_equal(crossfade_loop(signal, loop, SR, _seam(0.5)), signal)


def test_a_blend_of_material_that_has_drifted_apart_holds_the_level_it_had() -> None:
    """Reading the weighting off how alike the two sides measure is what keeps the blend from dipping."""
    noise = np.random.default_rng(0).standard_normal(SR)
    loop = Loop(start=SR // 2, end=SR)
    seam = _seam(0.25)
    fade = seam_frames(loop, SR, seam)
    middle = slice(loop.end - 3 * fade // 4, loop.end - fade // 4)  # the middle half of the blend
    progress = np.linspace(0.0, 1.0, fade, endpoint=True)
    summing_to_one = (1.0 - progress) * noise[loop.end - fade : loop.end] + progress * noise[
        loop.start - fade : loop.start
    ]

    blended = crossfade_loop(noise, loop, SR, seam)[middle]

    assert _level_of(blended) == pytest.approx(_level_of(noise), rel=0.1)
    assert _level_of(blended) > 1.2 * _level_of(summing_to_one[fade // 4 : 3 * fade // 4])  # the dip it avoids


def test_each_band_is_weighed_by_how_alike_that_band_came_round() -> None:
    """A string's partials return from a round at their own phases, so one law for all of them notches most.

    The loop spans a whole number of cycles of the loud low partial and a quarter turn more than a whole
    number of the quiet high one, which is the spacing a struck string carries. One law reads the likeness
    of the partial holding the energy and blends the other as though it too had come round in phase, which
    takes 3 dB out of it once per round.
    """
    signal = _sine(4 * SR, freq=LOW_PARTIAL, amp=1.0) + _sine(4 * SR, freq=HIGH_PARTIAL, amp=0.15)
    loop = Loop(start=4_000, end=6_000)
    fade = seam_frames(loop, SR, _seam(0.25))
    middle = slice(loop.end - 3 * fade // 4, loop.end - fade // 4)
    before = signal[middle]

    banded = crossfade_loop(signal, loop, SR, _seam(0.25))[middle]
    broadband = crossfade_loop(signal, loop, SR, _broadband_seam(0.25))[middle]

    assert _band_level_db(banded, HIGH_BAND) == pytest.approx(_band_level_db(before, HIGH_BAND), abs=0.5)
    assert _band_level_db(broadband, HIGH_BAND) < _band_level_db(before, HIGH_BAND) - 2.0
    for blended in (banded, broadband):  # the band holding the energy is served by either reading
        assert _band_level_db(blended, LOW_BAND) == pytest.approx(_band_level_db(before, LOW_BAND), abs=0.5)


def test_a_blend_of_material_that_repeats_exactly_leaves_it_as_it_was() -> None:
    """Two stretches a whole number of periods apart are the same material, so the blend has nothing to move."""
    signal = _sine(SR)
    loop = Loop(start=10 * PERIOD, end=18 * PERIOD)

    blended = crossfade_loop(signal, loop, SR, _seam(0.25))

    assert np.allclose(blended, signal, atol=1e-9)


# --- degenerate inputs ---------------------------------------------------------------------------


def test_estimate_period_rejects_short_and_degenerate_windows(geometry_config: GeometryConfig) -> None:
    assert _estimate_period(np.zeros(4), SR, geometry_config, FREQ) is None  # too few frames
    assert _estimate_period(_sine(20), 100_000, geometry_config, FREQ) is None  # under one period of its own pitch
    assert _estimate_period(np.zeros(SR), SR, geometry_config, FREQ) is None  # silence -> no peak above the floor


def test_the_cheapest_candidate_declines_where_the_steady_region_holds_less_than_the_floor(
    loop_config: LoopConfig,
) -> None:
    # 400 Hz at 8 kHz -> period 20; a 700-frame tone leaves under 500 steady frames, far under the floor,
    # so the region has no loop long enough to store.
    assert _cheapest(_sine(700, freq=400.0), loop_config, 400.0) is None


def test_the_floor_is_read_in_whole_periods_covering_every_bound(geometry_config: GeometryConfig) -> None:
    floor = shortest_loop_frames(PERIOD, SR, geometry_config)

    assert floor / SR >= geometry_config.min_loop_s
    assert floor >= _QUALITY_FFT  # the window a candidate's timbre is read over
    assert floor % PERIOD == 0
    assert shortest_loop_frames(SR, SR, geometry_config) == geometry_config.min_periods * SR  # a period past the floor


def test_a_floor_asked_shorter_than_one_analysis_window_still_offers_a_readable_region(
    loop_config: LoopConfig, loop: Callable[..., LoopConfig]
) -> None:
    """A gate reading a region too short to hold a spectrum would wave it through, so none is offered.

    This is what makes the seconds floor a tuning knob rather than a guard: however short it is set, and
    however high the material is pitched, every candidate offered is one the timbre gate measured.
    """
    high_freq = 800.0
    asked = loop(geometry={**loop_config.geometry.model_dump(), "min_loop_s": 1e-4, "min_periods": 1})
    candidates = loop_search(_sine(4 * SR, freq=high_freq), SR, asked, high_freq).candidates

    assert shortest_loop_frames(SR / high_freq, SR, asked.geometry) >= _QUALITY_FFT
    assert all(candidate.length >= _QUALITY_FFT for candidate in candidates)


# --- the candidates a clip is offered ---------------------------------------------------------------


def test_every_candidate_clears_the_floor_and_spans_whole_periods(loop_config: LoopConfig) -> None:
    floor = shortest_loop_frames(PERIOD, SR, loop_config.geometry)
    candidates = loop_search(_sine(4 * SR), SR, loop_config, FREQ).candidates

    assert len(candidates) > 1
    assert all(loop.length >= floor and loop.length % PERIOD == 0 for loop in candidates)
    assert all(loop.length >= round(loop_config.geometry.min_loop_s * SR) for loop in candidates)


def test_the_cheapest_candidate_is_the_one_the_frontier_offers_first(loop_config: LoopConfig) -> None:
    signal = _sine(4 * SR)
    offers = sorted(loop_search(signal, SR, loop_config, FREQ).candidates, key=lambda loop: (loop.end, loop.start))

    assert _cheapest(signal, loop_config) == offers[0]


def test_candidates_run_from_the_cheapest_stored_span_upward(loop_config: LoopConfig) -> None:
    """Storing a candidate keeps everything up to its end, so the frontier is a rate axis read along it."""
    candidates = loop_search(_sine(4 * SR), SR, loop_config, FREQ).candidates
    ends = [loop.end for loop in candidates]

    assert len(candidates) > 1
    assert ends == sorted(ends)
    assert len(set(ends)) == len(ends)  # each offer costs its own bytes, which is what a budget chooses on


def test_candidates_stay_inside_the_steady_region_and_stand_apart(loop_config: LoopConfig) -> None:
    signal = _sine(2 * SR)
    attack = _earliest_start(loop_config)
    tail = signal.size - round(loop_config.geometry.tail_skip_s * SR)
    candidates = loop_search(signal, SR, loop_config, FREQ).candidates

    assert len(set(candidates)) == len(candidates)  # reaches snapping onto one region are offered once
    assert all(loop.start >= attack - PERIOD for loop in candidates)  # snapping moves a start by a period
    assert all(loop.end <= tail for loop in candidates)


def test_a_note_that_settles_late_has_its_candidates_placed_past_the_stretch_it_settles_over(
    loop_config: LoopConfig,
) -> None:
    """The window opens off the material, so a recording whose sound keeps moving offers only what follows.

    Noise gives way to a steady tone here, which is a spectrum travelling as fast as a shape can and then
    holding still -- so where the candidates begin states where the reading placed the change.
    """
    rng = np.random.default_rng(0)
    moving = 0.8 * rng.standard_normal(SR // 2)
    signal = np.concatenate([moving, _sine(4 * SR)])
    candidates = loop_search(signal, SR, loop_config, FREQ).candidates

    assert candidates != ()
    assert all(loop.start > moving.size // 2 for loop in candidates)


def test_a_length_the_region_lacks_room_for_shrinks_onto_the_whole_periods_that_fit(
    loop_config: LoopConfig,
) -> None:
    # A tone holding the floor and half of it again has room for one candidate at the floor, and for the
    # length above it only once that length is brought back onto the whole periods left before the tail.
    floor = shortest_loop_frames(PERIOD, SR, loop_config.geometry)
    skipped = _earliest_start(loop_config) + round(loop_config.geometry.tail_skip_s * SR)
    signal = _sine(skipped + floor + floor // 2)
    tail = signal.size - round(loop_config.geometry.tail_skip_s * SR)
    longest = max(loop_search(signal, SR, loop_config, FREQ).candidates, key=lambda loop: loop.length)

    assert longest.length > floor
    assert longest.length % PERIOD == 0
    assert longest.end <= tail


def test_material_a_loop_has_no_purchase_on_offers_no_candidates(loop_config: LoopConfig) -> None:
    rng = np.random.default_rng(0)

    assert loop_search(rng.standard_normal(SR), SR, loop_config, FREQ).candidates == ()


def test_a_candidate_on_a_period_that_falls_between_frames_wraps_onto_the_material_it_left(
    loop_config: LoopConfig,
) -> None:
    """A period rounded to a frame drifts over the hundred rounds a loop spans; matching the end closes it."""
    signal = _sine(4 * SR, freq=UNEVEN_FREQ)
    loop = _cheapest(signal, loop_config, UNEVEN_FREQ)
    assert loop is not None

    whole_frames = round(loop.length / UNEVEN_PERIOD) * round(UNEVEN_PERIOD)
    rounded = Loop(start=loop.start, end=loop.start + whole_frames)

    assert _wrap_mismatch(signal, loop) < _wrap_mismatch(signal, rounded)


def test_a_matched_end_stays_inside_the_length_the_geometry_laid_out(loop_config: LoopConfig) -> None:
    """Matching moves an end by up to half a period, which leaves every candidate clearing the floor."""
    floor = shortest_loop_frames(UNEVEN_PERIOD, SR, loop_config.geometry)
    candidates = loop_search(_sine(4 * SR, freq=UNEVEN_FREQ), SR, loop_config, UNEVEN_FREQ).candidates

    assert candidates != ()
    assert all(loop.length >= floor for loop in candidates)
    assert all(loop.length >= round(loop_config.geometry.min_loop_s * SR) for loop in candidates)


# --- the level a stored region holds ----------------------------------------------------------------


_GENTLE_TAU_S = 3.0  # a note falling by about the level drift the shipped gate admits over a 2 s region
_STEEP_TAU_S = 0.6  # a note falling far past it, which is the region the gate is there to turn away
_LOOP = Loop(start=SR, end=3 * SR)


def _declining(frames: int, tau_s: float) -> NDArray[np.float64]:
    """A struck note: one pitch throughout, under an amplitude that falls away over ``tau_s`` as it rings."""
    return np.exp(-np.arange(frames, dtype=np.float64) / (tau_s * SR)) * _sine(frames)


def test_a_region_that_falls_as_it_rings_is_held_at_the_level_it_starts_on(envelope_config: EnvelopeConfig) -> None:
    """A region falling across itself steps the level back up on every wrap, which is the pulse a loop makes."""
    signal = _declining(4 * SR, _GENTLE_TAU_S)
    loop = _LOOP
    window = round(0.1 * SR)

    levelled = level_loop(signal, loop, _reading(envelope_config))

    opening = _level_of(levelled[loop.start : loop.start + window])
    closing = _level_of(levelled[loop.end - window : loop.end])
    assert closing == pytest.approx(opening, rel=0.05)
    assert opening == pytest.approx(_level_of(signal[loop.start : loop.start + window]), rel=0.05)


def test_levelling_leaves_the_attack_the_region_runs_out_of_untouched(envelope_config: EnvelopeConfig) -> None:
    """Pinning the gain at the loop start is what runs the attack into the region without a step."""
    signal = _declining(4 * SR, _GENTLE_TAU_S)
    loop = _LOOP

    levelled = level_loop(signal, loop, _reading(envelope_config))

    assert np.array_equal(levelled[: loop.start], signal[: loop.start])
    assert np.array_equal(levelled[loop.end :], signal[loop.end :])
    assert levelled[loop.start] == pytest.approx(signal[loop.start], rel=0.01)


@pytest.mark.parametrize(
    "loop",
    [
        pytest.param(Loop(start=100, end=140), id="a region of a few frames"),
        pytest.param(Loop(start=0, end=SR), id="a region running up to where the sound starts"),
    ],
)
def test_a_region_carrying_no_sound_is_left_as_it_stands(loop: Loop, envelope_config: EnvelopeConfig) -> None:
    """Levelling asks a gain of the material it is given, so silence comes back out as the silence it was."""
    signal = np.concatenate([np.zeros(SR), _sine(SR)])

    assert np.array_equal(level_loop(signal, loop, _reading(envelope_config)), signal)


# --- the level a held note goes on sounding at ---------------------------------------------------------


_DECLINE_WINDOW_S = 0.05  # the stretch one reading of a decline covers, which is where its last corner lands
_RUNS_ON = 4 * SR  # frames of recording, which is longer than any region a test below stores of it


def _fell_db(signal: NDArray[np.float64], frame: int) -> float:
    """How far under the level at ``_LOOP.start`` the material sits at ``frame``, over one reading window."""
    window = round(_DECLINE_WINDOW_S * SR)
    held = _level_of(signal[_LOOP.start : _LOOP.start + window])
    return gain_to_db(_level_of(signal[frame - window : frame]) / held)


def test_a_struck_note_declines_to_the_level_its_recording_ends_on(envelope_config: EnvelopeConfig) -> None:
    """What the PCM stops carrying is the recording's own fall past ``loop.start``, so the level states it."""
    signal = _declining(_RUNS_ON, _STEEP_TAU_S)

    decline = loop_decline(signal, SR, _LOOP, _reading(envelope_config))

    ends_on = decline.db(np.asarray([_RUNS_ON / SR]))[0]
    assert ends_on == pytest.approx(_fell_db(signal, _RUNS_ON), abs=1.0)


def test_the_decline_holds_unit_gain_over_everything_the_sample_still_stores(
    envelope_config: EnvelopeConfig,
) -> None:
    """The PCM up to ``loop.start`` carries every level it plays at, so nothing is put over it."""
    signal = _declining(_RUNS_ON, _STEEP_TAU_S)

    decline = loop_decline(signal, SR, _LOOP, _reading(envelope_config))

    over_the_attack = decline.db(np.linspace(0.0, _LOOP.start / SR, 32))
    assert float(np.max(np.abs(over_the_attack))) == pytest.approx(0.0, abs=0.5)


def test_a_note_falling_further_is_played_further_down(envelope_config: EnvelopeConfig) -> None:
    reads = np.asarray([_RUNS_ON / SR])

    gentle = loop_decline(_declining(_RUNS_ON, _GENTLE_TAU_S), SR, _LOOP, _reading(envelope_config))
    steep = loop_decline(_declining(_RUNS_ON, _STEEP_TAU_S), SR, _LOOP, _reading(envelope_config))

    assert steep.db(reads)[0] < gentle.db(reads)[0] < 0.0


def test_material_holding_its_level_is_played_as_it_stands(envelope_config: EnvelopeConfig) -> None:
    """A recording that keeps its level states a decline of nothing, so the loop repeats at the level it holds."""
    decline = loop_decline(_sine(_RUNS_ON), SR, _LOOP, _reading(envelope_config))

    assert float(np.max(np.abs(decline.readings.values))) == pytest.approx(0.0, abs=0.5)


def test_material_still_growing_past_the_region_is_played_up_the_way_it_grew(
    envelope_config: EnvelopeConfig,
) -> None:
    """The level is what the recording did, so material louder past the loop than at its start reads above unity."""
    signal = np.linspace(0.2, 1.0, _RUNS_ON) * _sine(_RUNS_ON)

    decline = loop_decline(signal, SR, _LOOP, _reading(envelope_config))

    assert decline.db(np.asarray([_RUNS_ON / SR]))[0] > 0.0


def test_the_decline_follows_a_note_that_rings_down_and_then_holds(envelope_config: EnvelopeConfig) -> None:
    """A curve turning where the material does reaches a knee no single straight run through it would."""
    knee = _RUNS_ON // 4
    envelope = np.concatenate([np.linspace(1.0, 0.2, knee), np.full(_RUNS_ON - knee, 0.2)])
    signal = envelope * _sine(_RUNS_ON)

    decline = loop_decline(signal, SR, _LOOP, _reading(envelope_config))

    at_the_knee = decline.db(np.asarray([knee / SR]))[0]
    assert at_the_knee == pytest.approx(_fell_db(signal, knee), abs=1.5)
    assert decline.db(np.asarray([_RUNS_ON / SR]))[0] == pytest.approx(at_the_knee, abs=1.5)


def test_the_decline_reads_the_level_where_the_region_starts_however_long_the_region_runs(
    envelope_config: EnvelopeConfig,
) -> None:
    """Levelling pins a region at the level it opens on, so where it closes leaves the decline where it was."""
    signal = _declining(_RUNS_ON, _STEEP_TAU_S)
    reads = np.asarray([_RUNS_ON / SR])

    brief = loop_decline(signal, SR, Loop(start=SR, end=SR + round(_DECLINE_WINDOW_S * SR)), _reading(envelope_config))
    whole = loop_decline(signal, SR, Loop(start=SR, end=_RUNS_ON), _reading(envelope_config))

    assert brief.db(reads)[0] == pytest.approx(whole.db(reads)[0], abs=0.5)


def test_a_recording_holding_too_little_past_the_loop_states_no_decline(envelope_config: EnvelopeConfig) -> None:
    """A remainder too short for one reading leaves the sample carrying every level it plays at."""
    signal = _declining(_RUNS_ON, _STEEP_TAU_S)
    reaches_the_end = Loop(start=_RUNS_ON - round(0.5 * _DECLINE_WINDOW_S * SR), end=_RUNS_ON)

    assert loop_decline(signal, SR, reaches_the_end, _reading(envelope_config)).transparent


@pytest.mark.parametrize(
    "root_hz_of",
    [
        pytest.param(lambda config: config.lowest_hz, id="the deepest note, read over the longest weighting"),
        pytest.param(lambda config: config.highest_hz, id="a high one, read over the shortest"),
    ],
)
def test_a_region_shorter_than_a_level_reading_is_held_at_one_level_all_the_same(
    envelope_config: EnvelopeConfig, root_hz_of: Callable[[EnvelopeConfig], float]
) -> None:
    """The level is read with the material around the region, so a brief region is levelled like a long one."""
    signal = _declining(4 * SR, _STEEP_TAU_S)
    brief = Loop(start=SR, end=SR + round(0.05 * SR))
    window = brief.length // 4

    levelled = level_loop(signal, brief, _reading(envelope_config, root_hz_of(envelope_config)))

    fell = _level_of(signal[brief.end - window : brief.end]) / _level_of(signal[brief.start : brief.start + window])
    opening = _level_of(levelled[brief.start : brief.start + window])
    closing = _level_of(levelled[brief.end - window : brief.end])
    assert fell < 0.95  # the material the region is taken from does fall across it
    assert closing == pytest.approx(opening, rel=0.02)


def test_a_prepared_region_is_held_at_one_level_and_blended_at_its_wrap(envelope_config: EnvelopeConfig) -> None:
    """Levelling before the blend is what leaves the blend joining phase over two stretches of one level."""
    signal = _declining(4 * SR, _GENTLE_TAU_S)
    loop = _LOOP

    seam = _seam(0.125)
    fade = seam_frames(loop, SR, seam)

    prepared = prepare_loop(signal, loop, SR, seam, _reading(envelope_config))

    untouched = local_level_over(prepared, _reading(envelope_config), start=loop.start, end=loop.end - fade)
    assert untouched[-1] == pytest.approx(untouched[0], rel=0.1)  # the stretch the blend leaves alone
    assert abs(prepared[loop.end - 1] - prepared[loop.start - 1]) < abs(signal[loop.end - 1] - signal[loop.start - 1])


# --- what a loop is worth --------------------------------------------------------------------------


def test_a_crossfaded_seam_reads_as_a_step_the_waveform_itself_could_have_made(loop_config: LoopConfig) -> None:
    signal = _sine(4 * SR)
    loop = _cheapest(signal, loop_config)
    assert loop is not None

    quality = loop_quality(signal, loop, SR, loop_config, _reading(loop_config.envelope))

    assert quality.seam_step < 2.0  # whole periods from a zero crossing: the wrap is the waveform's own motion


def test_a_loop_holding_a_timbre_the_material_moves_away_from_reports_the_distance(loop_config: LoopConfig) -> None:
    steady = recorded(_sine(2 * SR))
    brightened = steady + 0.5 * _sine(2 * SR, freq=5 * FREQ)
    loop = _cheapest(steady, loop_config)
    assert loop is not None

    held = loop_quality(np.concatenate([steady, steady]), loop, SR, loop_config, _reading(loop_config.envelope))
    moved = loop_quality(np.concatenate([steady, brightened]), loop, SR, loop_config, _reading(loop_config.envelope))

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
    tail = signal.size - int(loop_config.geometry.tail_skip_s * SR)
    end = tail - left_over
    reaching = Loop(start=end - shortest_loop_frames(PERIOD, SR, loop_config.geometry), end=end)

    assert loop_quality(signal, reaching, SR, loop_config, _reading(loop_config.envelope)).spectral_distance == 0.0


def test_a_silent_loop_region_reports_no_seam(loop_config: LoopConfig) -> None:
    assert (
        loop_quality(np.zeros(SR), Loop(start=100, end=500), SR, loop_config, _reading(loop_config.envelope)).seam_step
        == 0.0
    )


def test_a_loop_of_one_frame_has_no_step_to_measure_the_seam_against(loop_config: LoopConfig) -> None:
    assert (
        loop_quality(_sine(SR), Loop(start=100, end=101), SR, loop_config, _reading(loop_config.envelope)).seam_step
        == 0.0
    )


def test_a_region_falling_as_it_rings_states_the_drift_levelling_had_to_flatten(loop_config: LoopConfig) -> None:
    """The drift is read off the recording, so it states the gain flattening asked of the material."""
    steep = loop_quality(_declining(4 * SR, _STEEP_TAU_S), _LOOP, SR, loop_config, _reading(loop_config.envelope))
    gentle = loop_quality(_declining(4 * SR, _GENTLE_TAU_S), _LOOP, SR, loop_config, _reading(loop_config.envelope))
    held = loop_quality(_sine(4 * SR), _LOOP, SR, loop_config, _reading(loop_config.envelope))

    assert steep.level_drift_db > gentle.level_drift_db > held.level_drift_db
    assert held.level_drift_db == pytest.approx(0.0, abs=0.1)


def test_a_region_shorter_than_a_level_reading_states_the_fall_it_makes(loop_config: LoopConfig) -> None:
    """The level is read with the material around a region, so the gate judges every region on a measurement
    of its own and a brief one states exactly the fall its own material made across it.
    """
    brief = Loop(start=SR // 2, end=SR // 2 + PERIOD)

    quality = loop_quality(_declining(SR, _STEEP_TAU_S), brief, SR, loop_config, _reading(loop_config.envelope))

    assert quality.level_drift_db == pytest.approx(-gain_to_db(np.exp(-brief.length / (_STEEP_TAU_S * SR))), abs=0.01)


# --- a loop on a copy stored at another rate -------------------------------------------------------


def test_scaling_a_loop_onto_a_cheaper_copy_covers_the_same_stretch_of_material() -> None:
    """The stage settles a loop once, so a copy stored lower reaches the same region by the rate ratio."""
    loop = Loop(start=4_000, end=12_000)

    scaled = loop_at_rate(loop, SR, SR // 4, frames=SR)

    assert scaled == Loop(start=1_000, end=3_000)


def test_a_scaled_loop_ends_inside_the_frames_the_copy_holds() -> None:
    """The end names a frame a player can wrap from, so it is held inside what the copy actually stores."""
    loop = Loop(start=100, end=SR)

    scaled = loop_at_rate(loop, SR, SR, frames=1_000)

    assert scaled is not None
    assert scaled.end == 1_000


def test_a_rate_too_low_for_the_settled_bounds_to_survive_answers_with_no_loop() -> None:
    """A loop of a few frames scales to fewer than two, which names no wrap, so the trim carries the sample."""
    assert loop_at_rate(Loop(start=0, end=4), SR, SR // 1_000, frames=SR) is None


def test_a_longer_loop_is_blended_over_a_longer_stretch_of_itself() -> None:
    """A share of the loop is what the blend is stated as, so every round is blended the same way."""
    seam = _seam(0.25)
    short, long = Loop(start=SR, end=SR + 400), Loop(start=SR, end=SR + 4_000)

    assert seam_frames(short, SR, seam) == 100
    assert seam_frames(long, SR, seam) == 1_000


@pytest.mark.parametrize(
    ("loop", "expected"),
    [
        pytest.param(Loop(start=SR, end=SR + 2_000), 400, id="a share under the floor blends over the floor"),
        pytest.param(Loop(start=200, end=SR), 200, id="the room before the start is all a blend reaches for"),
        pytest.param(Loop(start=SR, end=SR + 80), 80, id="the region is all a blend has to rewrite"),
    ],
)
def test_the_blend_is_floored_in_seconds_and_bounded_by_the_room_around_it(loop: Loop, expected: int) -> None:
    seam = SeamConfig(
        fade_share=0.05,
        min_fade_s=0.05,
        crossovers_hz=CROSSOVERS,
        crossover_octaves=CROSSOVER_OCTAVES,
    )

    assert seam_frames(loop, SR, seam) == expected


def test_the_wrap_of_a_declining_region_is_read_against_the_motion_it_lands_in(loop_config: LoopConfig) -> None:
    """An average over the whole region would price the wrap in motion it never sits next to."""
    ramp = np.exp(-np.arange(4 * SR, dtype=np.float64) / (0.6 * SR))
    declining = ramp * _sine(4 * SR)
    steady = _sine(4 * SR)
    loop = Loop(start=10 * PERIOD, end=10 * PERIOD + shortest_loop_frames(PERIOD, SR, loop_config.geometry))

    fell = loop_quality(declining, loop, SR, loop_config, _reading(loop_config.envelope)).seam_step
    held = loop_quality(steady, loop, SR, loop_config, _reading(loop_config.envelope)).seam_step

    assert fell == pytest.approx(held, abs=1.0)  # the decline moves the reading by less than one step
