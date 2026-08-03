from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.piecewise import CurveNode, PiecewiseCurve, fit_piecewise
from optisample.dsp.series import Readings, Series

_SPAN_S = 4.0
_READINGS = 400  # windows a four-second trajectory is read in, the resolution a tick grid is written on
_SEED = 137
_NOISE_DB = 0.4  # the wobble a level reading carries, wide enough that a fit has something to average out
_TURNS_S = (0.0, 0.6, 2.1, _SPAN_S)  # a struck note: a fast fall, a long ring, then the material running out
_TURNS_DB = (0.0, -3.0, -20.0, -24.0)
_CORNER_TOLERANCE_S = 2 * _SPAN_S / (_READINGS - 1)  # a corner lands on a reading, so it places to the grid


def _grid() -> Series:
    return np.linspace(0.0, _SPAN_S, _READINGS)


def _held(seconds: Series) -> Series:
    """The trajectory the fixtures are read off: straight runs in decibels, turning at ``_TURNS_S``."""
    return np.asarray(np.interp(seconds, _TURNS_S, _TURNS_DB), dtype=np.float64)


@pytest.fixture
def turning() -> Readings:
    """A trajectory made of straight runs, which is a curve a fit can reach exactly."""
    seconds = _grid()
    return Readings(values=_held(seconds), seconds=seconds)


@pytest.fixture
def wobbling() -> Readings:
    """The same trajectory read off material that wobbles, which is what a fit has to average through."""
    seconds = _grid()
    wobble = np.random.default_rng(_SEED).standard_normal(seconds.size)
    return Readings(values=_held(seconds) + _NOISE_DB * wobble, seconds=seconds)


@pytest.fixture
def ringing() -> Readings:
    """A note falling at two rates: a struck string's fast start and the slow ring it settles into.

    One straight ramp is the shape today's decline is written as, and this is the trajectory that shape has
    to stand for, so a fit measured against it says what more corners are worth.
    """
    seconds = _grid()
    fast, slow = np.exp(-9.0 * seconds), 0.06 * np.exp(-0.5 * seconds)
    return Readings(values=20.0 * np.log10(fast + slow), seconds=seconds)


def _gap_db(curve: PiecewiseCurve, readings: Readings) -> float:
    """The widest a curve stands from the trajectory it was fitted to, which is what a listener meets."""
    return float(np.max(np.abs(curve.at(readings.seconds) - readings.values)))


# --- the curve itself -------------------------------------------------------------------------------


def test_a_curve_runs_straight_between_its_corners_and_holds_level_past_both_ends() -> None:
    """A tracker plays its last node for as long as a note runs on, so the curve reads the same way."""
    curve = PiecewiseCurve(
        nodes=(CurveNode(seconds=0.0, value=0.0), CurveNode(seconds=2.0, value=-12.0)),
    )

    read = curve.at(np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0, 5.0]))

    assert read == pytest.approx([0.0, 0.0, -3.0, -6.0, -12.0, -12.0])


# --- where the corners go ---------------------------------------------------------------------------


def test_a_trajectory_of_straight_runs_is_reached_by_a_fit_with_a_corner_for_each(turning: Readings) -> None:
    """The placement is exact over every run of corners, so a trajectory a curve can state is stated."""
    assert _gap_db(fit_piecewise(turning, nodes=len(_TURNS_S)), turning) < 0.05


def test_the_corners_land_where_the_trajectory_turns(turning: Readings) -> None:
    """A corner earns its place at a change of slope, which is what makes the fit follow rather than average."""
    curve = fit_piecewise(turning, nodes=len(_TURNS_S))

    assert curve.seconds == pytest.approx(_TURNS_S, abs=_CORNER_TOLERANCE_S)
    assert curve.values == pytest.approx(_TURNS_DB, abs=0.1)


def test_the_corners_follow_the_trajectory_where_evenly_spread_ones_would_pass_it_by(turning: Readings) -> None:
    """Both turns sit in the first half, so placing corners by the trajectory is what a search buys."""
    curve = fit_piecewise(turning, nodes=len(_TURNS_S))
    evenly = PiecewiseCurve(
        nodes=tuple(
            CurveNode(seconds=float(moment), value=float(value))
            for moment, value in zip(np.linspace(0.0, _SPAN_S, len(_TURNS_S)), _held(np.linspace(0.0, _SPAN_S, 4)))
        )
    )

    assert _gap_db(curve, turning) < 0.1 * _gap_db(evenly, turning)


@pytest.mark.parametrize(("fewer", "more"), [(2, 3), (3, 4), (4, 6), (6, 12), (12, 25)])
def test_each_node_a_fit_is_given_leaves_no_more_error_behind_it(wobbling: Readings, fewer: int, more: int) -> None:
    """More corners can always hold the ones already placed, so the fit's error falls as the node count rises."""
    residual = [
        float(np.sum((fit_piecewise(wobbling, nodes=nodes).at(wobbling.seconds) - wobbling.values) ** 2))
        for nodes in (fewer, more)
    ]

    assert residual[1] <= residual[0]


def test_a_fit_averages_through_the_wobble_a_reading_carries(wobbling: Readings) -> None:
    """The corners answer to every reading at once, so the curve lands near the trajectory under the noise."""
    curve = fit_piecewise(wobbling, nodes=len(_TURNS_S))

    assert _gap_db(curve, Readings(values=_held(wobbling.seconds), seconds=wobbling.seconds)) < _NOISE_DB


# --- what more corners are worth ----------------------------------------------------------------------


def test_a_note_falling_at_two_rates_is_followed_far_closer_than_one_straight_ramp_reaches(
    ringing: Readings,
) -> None:
    """A struck note's fast start and slow ring are two slopes, which is the shape one ramp stands furthest from."""
    ramp = fit_piecewise(ringing, nodes=2)
    fitted = fit_piecewise(ringing, nodes=25)

    assert _gap_db(ramp, ringing) > 10.0
    assert _gap_db(fitted, ringing) < 1.0


# --- what a fit is given ------------------------------------------------------------------------------


def test_a_trajectory_shorter_than_the_nodes_asked_for_is_stated_by_its_own_readings() -> None:
    """Every reading becomes a corner, which runs the curve through the trajectory it was read from."""
    readings = Readings(values=np.asarray([0.0, -4.0, -9.0]), seconds=np.asarray([0.0, 0.1, 0.3]))

    curve = fit_piecewise(readings, nodes=12)

    assert curve.seconds == pytest.approx(readings.seconds)
    assert curve.values == pytest.approx(readings.values)


def test_a_fit_asked_for_fewer_nodes_than_a_straight_run_needs_is_refused(turning: Readings) -> None:
    with pytest.raises(ValueError, match="at least 2 nodes"):
        fit_piecewise(turning, nodes=1)


def test_a_fit_given_nothing_to_follow_is_refused() -> None:
    empty = Readings(values=np.asarray([], dtype=np.float64), seconds=np.asarray([], dtype=np.float64))

    with pytest.raises(ValueError, match="at least one reading"):
        fit_piecewise(empty, nodes=12)


def test_a_trajectory_read_in_more_windows_than_the_search_tables_is_refused() -> None:
    """The table grows with the square of the readings, so a long note is read in longer windows instead."""
    seconds = np.linspace(0.0, 100.0, 4_096)

    with pytest.raises(ValueError, match="read in at most"):
        fit_piecewise(Readings(values=np.zeros_like(seconds), seconds=seconds), nodes=25)
