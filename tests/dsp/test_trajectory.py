from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from optisample.dsp.level import level_readings
from optisample.dsp.piecewise import MOST_READINGS
from optisample.dsp.series import Readings, Series
from optisample.dsp.trajectory import (
    TrajectoryMember,
    fit_shared_trajectory,
    reading_window_s,
)

SR = 44_100

_WINDOW_S = 0.01
_SPAN_S = 2.0
_NODES = 4  # corners the planted shape turns through, so a fit given this many reaches it exactly
_TURNS_S = (0.0, 0.3, 1.0, _SPAN_S)  # a struck note: a fast fall, a long ring, then the material running out
_TURNS_DB = (0.0, -3.0, -18.0, -22.0)
_SAMPLE_OFFSETS_DB = (3.0, -3.0)  # how far two stored samples stand apart, which the format's own gain states
_DYNAMIC_OFFSETS_DB = (2.0, -2.0)  # how far two note volumes stand apart, which the pattern states
_OWN_RATE_DB_PER_S = -6.0  # the rate one member declines at over the shape its instrument plays it through
_GRID_SLIVER_DB = 0.1  # what a turn falling between two readings leaves, since a corner lands on one
_HEAVY_WEIGHT = 9.0
_LIGHT_WEIGHT = 1.0


def _clock(span_s: float) -> Series:
    """The moments a stretch of ``span_s`` seconds is read at, each window centering on its own."""
    count = round(span_s / _WINDOW_S)
    return (np.arange(count, dtype=np.float64) + 0.5) * _WINDOW_S


def _held(seconds: Series) -> Series:
    """The shape every member is planted from: straight runs in decibels, turning at ``_TURNS_S``."""
    return np.asarray(np.interp(seconds, _TURNS_S, _TURNS_DB), dtype=np.float64)


def _member(
    *,
    groups: tuple[int, ...],
    weight: float = _LIGHT_WEIGHT,
    span_s: float = _SPAN_S,
    own_rate_db_per_s: float = 0.0,
    shape: Callable[[Series], Series] = _held,
) -> TrajectoryMember:
    """One member planted from the shared shape, at the level its groups state and a rate of its own."""
    seconds = _clock(span_s)
    offset_db = _SAMPLE_OFFSETS_DB[groups[0]] + _DYNAMIC_OFFSETS_DB[groups[1]]
    values = shape(seconds) + offset_db + own_rate_db_per_s * seconds
    return TrajectoryMember(readings=Readings(values=values, seconds=seconds), groups=groups, weight=weight)


def _crossed() -> tuple[TrajectoryMember, ...]:
    """Every sample against every note volume, which is the design the two ladders of offsets are read from."""
    return tuple(_member(groups=(sample, dynamic)) for sample in range(2) for dynamic in range(2))


def _planted_offsets(offsets: Sequence[float]) -> list[float]:
    """The planted levels stated the way a fit states them: against the mean of the set they belong to."""
    return [level - float(np.mean(offsets)) for level in offsets]


# --- the window a set is read at ----------------------------------------------------------------------


def test_a_short_note_is_read_finer_than_the_grid_its_envelope_is_written_on() -> None:
    assert reading_window_s(SR, SR) == pytest.approx(0.005)


def test_a_note_running_past_what_a_fit_tables_is_read_in_proportionally_longer_windows() -> None:
    frames = 100 * SR

    assert frames / SR / reading_window_s(frames, SR) == pytest.approx(MOST_READINGS, rel=1e-3)


@pytest.mark.parametrize("frames", [MOST_READINGS * 240 + 1, 500_000, 10 * 48_000, 106 * 48_000])
def test_a_stretch_of_any_length_is_read_in_the_windows_a_fit_can_table(frames: int) -> None:
    """The window spans whole frames, so what a reading answers with stays inside what a fit prices."""
    readings = level_readings(np.zeros(frames, dtype=np.float64), SR, window_s=reading_window_s(frames, SR))

    assert readings.count <= MOST_READINGS


# --- what one shape and two ladders of offsets reach ----------------------------------------------------


def test_members_falling_alike_at_levels_of_their_own_are_written_as_one_shape() -> None:
    """The offsets are exactly the levels a format states per sample and per note, so nothing is left over."""
    shared = fit_shared_trajectory(_crossed(), nodes=_NODES)

    assert shared.dispersion_db < _GRID_SLIVER_DB


def test_each_axis_states_how_far_its_own_groups_stand_from_the_instrument() -> None:
    """Centering on the set leaves the shape carrying where the instrument sits and the offsets the spread."""
    shared = fit_shared_trajectory(_crossed(), nodes=_NODES)

    assert shared.offsets_db[0] == pytest.approx(_planted_offsets(_SAMPLE_OFFSETS_DB), abs=1e-6)
    assert shared.offsets_db[1] == pytest.approx(_planted_offsets(_DYNAMIC_OFFSETS_DB), abs=1e-6)


def test_a_member_reaches_its_own_level_through_the_axes_it_belongs_to() -> None:
    """One sample plays at several dynamics and one dynamic across several samples, so the two axes add."""
    shared = fit_shared_trajectory(_crossed(), nodes=_NODES)

    reached = shared.member_offset_db((0, 1))

    assert reached == pytest.approx(shared.offsets_db[0][0] + shared.offsets_db[1][1])


def test_the_shape_a_set_shares_is_the_trajectory_its_members_make() -> None:
    """The offsets carry the levels, so what is left for the curve is the decline every member makes."""
    shared = fit_shared_trajectory(_crossed(), nodes=_NODES)

    read = shared.curve.at(np.asarray(_TURNS_S))

    assert read == pytest.approx(_TURNS_DB, abs=0.2)


# --- what one shape leaves behind -----------------------------------------------------------------------


def test_a_member_declining_at_a_rate_of_its_own_is_what_one_shape_leaves_behind() -> None:
    """A key falling faster than its instrument cannot be written as a level, which is the dispersion."""
    members = (
        _member(groups=(0, 0)),
        _member(groups=(0, 1)),
        _member(groups=(1, 0)),
        _member(groups=(1, 1), own_rate_db_per_s=_OWN_RATE_DB_PER_S),
    )

    shared = fit_shared_trajectory(members, nodes=_NODES)

    assert shared.dispersion_db > 4.0
    assert shared.gaps_db[3] == max(shared.gaps_db)


def test_a_member_worth_more_to_the_instrument_is_followed_closer_than_one_worth_less() -> None:
    """The shape answers to what the material leans on, so weight decides which member gives ground."""
    members = (
        _member(groups=(0, 0), weight=_HEAVY_WEIGHT),
        _member(groups=(1, 1), weight=_LIGHT_WEIGHT, own_rate_db_per_s=_OWN_RATE_DB_PER_S),
    )

    shared = fit_shared_trajectory(members, nodes=_NODES)

    assert shared.gaps_db[0] < shared.gaps_db[1]


def test_a_member_running_shorter_than_its_instrument_is_measured_over_the_stretch_it_reaches() -> None:
    """A note the material holds briefly says nothing about the moments it never reaches."""
    members = (
        _member(groups=(0, 0)),
        _member(groups=(1, 1), span_s=_SPAN_S / 4),
    )

    shared = fit_shared_trajectory(members, nodes=_NODES)

    assert shared.curve.seconds[-1] > _SPAN_S / 2
    assert shared.dispersion_db < _GRID_SLIVER_DB


def test_a_group_no_member_belongs_to_stands_at_the_level_of_the_instrument_itself() -> None:
    """Numbering leaves gaps where a slot stores fewer samples than it reserved, and a gap costs nothing."""
    members = (_member(groups=(0, 0)), _member(groups=(1, 1)))

    shared = fit_shared_trajectory(
        tuple(
            TrajectoryMember(readings=member.readings, groups=(member.groups[0] * 2, member.groups[1]), weight=1.0)
            for member in members
        ),
        nodes=_NODES,
    )

    assert shared.offsets_db[0][1] == pytest.approx(0.0)


# --- what a set is given ---------------------------------------------------------------------------------


def test_a_set_holding_nothing_to_share_a_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        fit_shared_trajectory((), nodes=_NODES)


def test_a_member_worth_nothing_to_its_instrument_is_refused() -> None:
    with pytest.raises(ValueError, match="worth more than nothing"):
        fit_shared_trajectory((_member(groups=(0, 0), weight=0.0),), nodes=_NODES)


def test_members_disagreeing_on_how_many_axes_tell_them_apart_are_refused() -> None:
    stated = _member(groups=(0, 0))
    bare = TrajectoryMember(readings=stated.readings, groups=(0,), weight=1.0)

    with pytest.raises(ValueError, match="one set of axes"):
        fit_shared_trajectory((stated, bare), nodes=_NODES)


def test_members_told_apart_on_no_axis_at_all_are_refused() -> None:
    stated = _member(groups=(0, 0))

    with pytest.raises(ValueError, match="one set of axes"):
        fit_shared_trajectory((TrajectoryMember(readings=stated.readings, groups=(), weight=1.0),), nodes=_NODES)


def test_members_read_at_window_lengths_putting_them_on_different_clocks_are_refused() -> None:
    """One envelope is played on one clock, so a set read two ways states no shape at all."""
    stated = _member(groups=(0, 0))
    shifted = TrajectoryMember(
        readings=Readings(values=stated.readings.values, seconds=stated.readings.seconds + _WINDOW_S / 2),
        groups=(1, 1),
        weight=1.0,
    )

    with pytest.raises(ValueError, match="one window length"):
        fit_shared_trajectory((stated, shifted), nodes=_NODES)
