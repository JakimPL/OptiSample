from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.level import (
    Clock,
    Level,
    constant_level,
    curve_level,
    gain_level,
    level_readings,
    product,
    read_level,
    sampled_level,
    unit_level,
)
from optisample.dsp.piecewise import CurveNode, PiecewiseCurve
from optisample.dsp.series import Readings, Series
from optisample.music import semitone_ratio

SR = 44_100

_SPAN_S = 2.0
_FRAMES = round(_SPAN_S * SR)
_OCTAVE = 12  # semitones a pitch is carried by for the played clock to run at twice the speed
_EXACT_DB = 1e-9  # decibels two levels stand apart while stating the same thing
_ROUND_TRIP = 1e-12  # the amplitude a waveform stands from itself after a level is divided out and put back


def _falling(*, span_s: float = _SPAN_S, from_db: float = 0.0, to_db: float = -24.0) -> Level:
    """A level running straight from ``from_db`` down to ``to_db``, turning only at its two ends."""
    return Level(
        readings=Readings(
            values=np.asarray([from_db, to_db], dtype=np.float64),
            seconds=np.asarray([0.0, span_s], dtype=np.float64),
        ),
        clock=Clock.RECORDED,
    )


def _moments(count: int = 9, *, span_s: float = _SPAN_S) -> Series:
    """Moments spread across a stretch, which is where two levels are compared point by point."""
    return np.linspace(0.0, span_s, num=count, dtype=np.float64)


# --- what a level states ------------------------------------------------------------------------------


def test_a_level_holding_one_value_states_it_at_every_moment() -> None:
    held = constant_level(-6.0, Clock.RECORDED)
    assert np.allclose(held.db(_moments()), -6.0)


def test_a_level_holds_its_end_values_past_both_ends() -> None:
    falling = _falling()
    assert falling.db(np.asarray([-1.0])) == pytest.approx(0.0)
    assert falling.db(np.asarray([_SPAN_S + 5.0])) == pytest.approx(-24.0)


def test_a_constant_level_turns_at_one_moment_however_long_it_is_asked_for() -> None:
    """A scalar gain stays one number, which is what keeps carrying a level everywhere affordable."""
    assert constant_level(-6.0, Clock.PLAYED).readings.count == 1


def test_a_level_stated_as_a_gain_reads_back_as_that_gain() -> None:
    assert gain_level(0.5, Clock.PLAYED).gain(_moments()) == pytest.approx(0.5)


def test_only_a_level_holding_unity_throughout_is_transparent() -> None:
    assert unit_level(Clock.PLAYED).transparent
    assert not _falling().transparent


def test_a_level_peaks_at_the_loudest_moment_it_turns_at() -> None:
    assert _falling(from_db=-3.0, to_db=-24.0).peak_db == pytest.approx(-3.0)


# --- the algebra --------------------------------------------------------------------------------------


def test_multiplying_two_levels_adds_their_decibels() -> None:
    composed = constant_level(-6.0, Clock.RECORDED).times(constant_level(-3.0, Clock.RECORDED))
    assert composed.db(_moments()) == pytest.approx(-9.0)


def test_dividing_a_level_by_itself_leaves_a_level_that_changes_nothing() -> None:
    falling = _falling()
    assert falling.over(falling).db(_moments()) == pytest.approx(0.0, abs=_EXACT_DB)


def test_a_composition_turns_wherever_either_level_does() -> None:
    """Both run straight between their own moments, so their product is exact rather than resampled."""
    early = Level(
        readings=Readings(values=np.asarray([0.0, -6.0]), seconds=np.asarray([0.0, 1.0])),
        clock=Clock.RECORDED,
    )
    late = Level(
        readings=Readings(values=np.asarray([0.0, -6.0]), seconds=np.asarray([0.5, 2.0])),
        clock=Clock.RECORDED,
    )
    assert list(early.times(late).readings.seconds) == [0.0, 0.5, 1.0, 2.0]


def test_a_waveform_divided_by_a_level_and_multiplied_back_is_the_waveform_it_started_as() -> None:
    """The whole point of the split: what a level takes out of a carrier, putting it back restores."""
    rng = np.random.default_rng(137)
    waveform = rng.standard_normal(_FRAMES)
    gains = _falling().frame_gains(_FRAMES, SR)
    assert np.allclose(waveform / gains * gains, waveform, atol=_ROUND_TRIP)


def test_scaling_a_level_shifts_it_without_moving_where_it_turns() -> None:
    falling = _falling()
    lifted = falling.scaled_db(6.0)
    assert np.array_equal(lifted.readings.seconds, falling.readings.seconds)
    assert lifted.db(_moments()) == pytest.approx(falling.db(_moments()) + 6.0)


def test_a_product_of_many_levels_is_every_one_of_them_multiplied_on() -> None:
    levels = [constant_level(level_db, Clock.PLAYED) for level_db in (-1.0, -2.0, -3.0)]
    assert product(levels).db(_moments()) == pytest.approx(-6.0)


def test_a_product_states_at_least_one_level() -> None:
    with pytest.raises(ValueError, match="at least one level"):
        product([])


# --- the clock the moments are counted on ---------------------------------------------------------------


def test_levels_counted_on_different_clocks_refuse_to_compose() -> None:
    """A recorded moment and a played one state different things, so multiplying them is a mistake."""
    with pytest.raises(ValueError, match="one clock"):
        constant_level(0.0, Clock.RECORDED).times(constant_level(0.0, Clock.PLAYED))


def test_a_recording_sounded_an_octave_up_runs_its_level_at_twice_the_speed() -> None:
    played = _falling().at_pitch(root_pitch=60, pitch=60 + _OCTAVE)
    assert played.clock is Clock.PLAYED
    assert played.db(np.asarray([_SPAN_S / 2.0])) == pytest.approx(-24.0)


def test_sounding_a_recording_at_its_own_key_leaves_its_moments_where_they_are() -> None:
    falling = _falling()
    played = falling.at_pitch(root_pitch=60, pitch=60)
    assert np.allclose(played.readings.seconds, falling.readings.seconds)


def test_carrying_a_level_to_a_pitch_and_back_returns_the_moments_it_came_from() -> None:
    falling = _falling()
    played = falling.at_pitch(root_pitch=60, pitch=67)
    assert np.allclose(played.readings.seconds / semitone_ratio(60 - 67), falling.readings.seconds)


def test_a_played_level_is_sounded_as_it_stands() -> None:
    """A written envelope walks the tick clock whatever key is struck, so repitching leaves it alone."""
    with pytest.raises(ValueError, match="sounded as it stands"):
        constant_level(0.0, Clock.PLAYED).at_pitch(root_pitch=60, pitch=72)


# --- how far two levels stand apart -----------------------------------------------------------------------


def test_two_levels_stand_apart_by_the_widest_gap_between_them() -> None:
    assert _falling().distance_db(constant_level(0.0, Clock.RECORDED)) == pytest.approx(24.0)


def test_a_level_stands_no_distance_from_itself() -> None:
    falling = _falling()
    assert falling.distance_db(falling) == pytest.approx(0.0, abs=_EXACT_DB)


def test_the_widest_gap_is_read_at_a_moment_one_of_them_turns_at() -> None:
    """Both run straight between their moments, so their difference does too and its peak is a corner."""
    crossing = Level(
        readings=Readings(values=np.asarray([-24.0, 0.0]), seconds=np.asarray([0.0, _SPAN_S])),
        clock=Clock.RECORDED,
    )
    apart = np.abs(_falling().db(_moments(1001)) - crossing.db(_moments(1001)))
    assert _falling().distance_db(crossing) == pytest.approx(float(np.max(apart)))


# --- the shapes a level is built from ---------------------------------------------------------------------


def test_a_recordings_own_level_enters_the_algebra_frame_by_frame() -> None:
    amplitudes = np.linspace(1.0, 0.25, _FRAMES)
    level = sampled_level(amplitudes, SR, Clock.RECORDED)
    assert level.readings.count == _FRAMES
    assert level.gain(np.asarray([0.0])) == pytest.approx(1.0)


def test_readings_already_taken_in_decibels_are_carried_as_the_level_they_state() -> None:
    readings = level_readings(np.full(_FRAMES, 0.5), SR, window_s=0.05)
    assert read_level(readings, Clock.RECORDED).db(np.asarray([1.0])) == pytest.approx(readings.values[0])


def test_a_fitted_curve_is_carried_as_the_level_it_states() -> None:
    curve = PiecewiseCurve(nodes=(CurveNode(seconds=0.0, value=0.0), CurveNode(seconds=1.0, value=-12.0)))
    assert curve_level(curve, Clock.PLAYED).db(np.asarray([0.5])) == pytest.approx(-6.0)


def test_fitting_a_level_states_it_in_the_corners_a_format_has_room_for() -> None:
    windowed = read_level(level_readings(np.linspace(1.0, 0.25, _FRAMES), SR, window_s=0.05), Clock.RECORDED)
    fitted = windowed.fitted(nodes=4)
    assert fitted.readings.count <= 4
    assert fitted.clock is Clock.RECORDED


def test_a_level_turning_at_every_frame_is_read_in_windows_before_it_is_fitted() -> None:
    """A fit tables every stretch against the line through it, so it is bounded in what it may be given."""
    with pytest.raises(ValueError, match="at most"):
        sampled_level(np.linspace(1.0, 0.25, _FRAMES), SR, Clock.RECORDED).fitted(nodes=4)
