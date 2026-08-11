from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.piecewise import MOST_READINGS, PiecewiseCurve, fit_piecewise
from optisample.dsp.series import Readings, Series

Groups = NDArray[np.intp]

_FINEST_WINDOW_S: Final = 0.005  # the shortest stretch one reading of a trajectory averages over
_SETTLED_DB: Final = 1e-6  # the widest an offset moves for the sweep that moved it to count as settled
_MOST_SWEEPS: Final = 32  # sweeps the offsets are given to settle, which a handful of them spend
_CLOCK_TOLERANCE_S: Final = 1e-9  # how far two moments stand apart and still centre the same window
_NO_GAP_DB: Final = 0.0  # what a shape leaves behind on a set holding nothing to leave it on
_ONE_AXIS: Final = 1  # offset axes a set states its members apart on, at the fewest


def reading_window_s(frames: int, sample_rate: int) -> float:
    """The window a trajectory of ``frames`` is read in, for a fit to place corners across it.

    A fit prices every stretch of a trajectory against the line through it, so how many readings it can be
    given is bounded (:data:`~optisample.dsp.piecewise.MOST_READINGS`) and a long note is read in
    proportionally longer windows. A short one is read in :data:`_FINEST_WINDOW_S` windows, finer than the
    tick grid an envelope is written on, so the reading holds every corner the format has room to place.

    The window spans whole frames, which is what :func:`~optisample.dsp.levels.level_readings` reads in,
    so the count it answers with stays inside the bound however long the stretch runs.
    """
    return max(_FINEST_WINDOW_S, ceil(frames / MOST_READINGS) / sample_rate)


@dataclass(frozen=True)
class TrajectoryMember:
    """One served note's level trajectory, the offsets it is played through, and what it is worth.

    ``groups`` names the offset this member answers to on each axis -- which stored sample it plays
    through, which note volume the pattern writes for it -- in the order the axes are given, so members
    sharing a sample share that sample's own level and members sharing a dynamic share that dynamic's.
    ``weight`` is what the note is worth to the instrument, which is what makes one shape answer loudest
    to the notes the material leans on.

    ``readings`` stands in decibels, read from the note's own onset at the window length the whole set
    shares (:func:`reading_window_s`), so every member states the opening stretch of one clock.
    """

    readings: Readings
    groups: tuple[int, ...]
    weight: float


@dataclass(frozen=True)
class SharedTrajectory:
    """The one shape an instrument plays every voice down by, and the offsets telling its members apart.

    A tracker keeps one volume envelope per instrument and a level per sample and per pattern note, so
    what a set of notes can be written as is exactly this: one curve on the played clock, plus a level
    each member reaches through the axes it belongs to. ``offsets_db`` holds one level per group per axis,
    centred on the set's own weighted level, so an offset states how far a sample or a dynamic stands from
    its instrument. ``gaps_db`` states, member by member, how far the written pair ends up from the
    trajectory that member actually makes.
    """

    curve: PiecewiseCurve
    offsets_db: tuple[tuple[float, ...], ...]
    gaps_db: tuple[float, ...]

    @property
    def dispersion_db(self) -> float:
        """The widest any member stands from the shape its instrument plays it through.

        This is what one envelope costs the notes sharing it -- large where keys decline at rates of their
        own, near nothing where they decline alike -- so it is the reading that says whether a set is
        better written as several instruments.
        """
        return max(self.gaps_db, default=_NO_GAP_DB)

    def member_offset_db(self, groups: tuple[int, ...]) -> float:
        """The level a member belonging to ``groups`` is shifted by: its own offset on every axis, added."""
        return sum(self.offsets_db[axis][group] for axis, group in enumerate(groups))


@dataclass(frozen=True)
class _Stack:
    """Every member's trajectory laid on one clock, beside the weight each one carries at each moment.

    Members are read from their own onsets at one window length, so a member's readings are the opening
    stretch of the longest member's clock and the rest of its row carries no weight. Laying them out this
    way is what lets a shape be read at each moment off the notes that reach it.
    """

    values: Series
    weights: Series
    seconds: Series

    @property
    def covered(self) -> Series:
        """The weight standing behind each moment, which is what the shape read there answers to."""
        return np.asarray(np.sum(self.weights, axis=0), dtype=np.float64)

    @property
    def held(self) -> Series:
        """The weight each member carries over the whole stretch it was read across."""
        return np.asarray(np.sum(self.weights, axis=1), dtype=np.float64)


def _checked(members: Sequence[TrajectoryMember]) -> None:
    """Confirm the set states one shape's worth of trajectories.

    Raises:
        ValueError: when the set holds no member, when a member is worth nothing or less, or when the
            members disagree on how many axes they are told apart on.
    """
    if not members:
        raise ValueError("a shape is shared by at least one member")

    if any(member.weight <= 0.0 for member in members):
        raise ValueError("a member is worth more than nothing to the instrument sharing the shape")

    axes = {len(member.groups) for member in members}
    if len(axes) > _ONE_AXIS or axes == {0}:
        raise ValueError(f"members are told apart on one set of axes, against the {sorted(axes)} stated")


def _stacked(members: Sequence[TrajectoryMember]) -> _Stack:
    """The members laid on the longest one's clock.

    Raises:
        ValueError: when a member's moments stand apart from the clock the longest member states, which is
            what reading the set at two window lengths leaves behind.
    """
    seconds = max(members, key=lambda member: member.readings.count).readings.seconds
    values = np.zeros((len(members), seconds.size), dtype=np.float64)
    weights = np.zeros_like(values)
    for index, member in enumerate(members):
        reach = member.readings.count
        if not np.allclose(member.readings.seconds, seconds[:reach], atol=_CLOCK_TOLERANCE_S):
            raise ValueError("members are read at one window length, so each states the opening of one clock")

        values[index, :reach] = member.readings.values
        weights[index, :reach] = member.weight

    return _Stack(values=values, weights=weights, seconds=seconds)


def _memberships(members: Sequence[TrajectoryMember]) -> tuple[Groups, ...]:
    """Which offset every member answers to, one array of group positions per axis."""
    return tuple(
        np.asarray([member.groups[axis] for member in members], dtype=np.intp) for axis in range(len(members[0].groups))
    )


def _member_offsets(offsets: Sequence[Series], memberships: Sequence[Groups]) -> Series:
    """The level each member is shifted by: what it answers to on every axis, added."""
    reached = [axis_offsets[membership] for axis_offsets, membership in zip(offsets, memberships)]
    return np.asarray(np.sum(reached, axis=0), dtype=np.float64)


def _shared_values(stack: _Stack, offsets: Series) -> Series:
    """The level the members agree on at each moment, once each is stood at the shape's own level."""
    lifted = stack.weights * (stack.values - offsets[:, np.newaxis])
    return np.asarray(np.sum(lifted, axis=0) / stack.covered, dtype=np.float64)


def _axis_offsets(stack: _Stack, residual: Series, membership: Groups, groups: int) -> Series:
    """Each group's own level: the weighted mean of what its members hold above the shape they share.

    The answer is centred on the set's own weighted level, which leaves the shape carrying where the
    instrument sits and each offset stating how far one group stands from it. A group no member belongs to
    stands at the instrument's own level.
    """
    above = np.bincount(membership, weights=np.sum(stack.weights * residual, axis=1), minlength=groups)
    held = np.bincount(membership, weights=stack.held, minlength=groups)
    offsets = np.divide(above, held, out=np.zeros_like(above), where=held > 0.0)
    return np.asarray(offsets - np.sum(held * offsets) / np.sum(held), dtype=np.float64)


def _settled_offsets(stack: _Stack, memberships: Sequence[Groups]) -> tuple[Series, ...]:
    """The offsets and the shape read off each other in turn, until a sweep leaves the offsets where it found them.

    Holding the shape still leaves each axis's offsets a weighted mean, and holding every offset still
    leaves the shape one as well, so each step of a sweep lowers the same weighted squared error and the
    pair settles. A handful of sweeps reaches it, because the axes carry levels the shape carries none of.
    """
    offsets = [np.zeros(int(membership.max()) + 1, dtype=np.float64) for membership in memberships]
    for _ in range(_MOST_SWEEPS):
        shared = _shared_values(stack, _member_offsets(offsets, memberships))
        moved = 0.0
        for axis, membership in enumerate(memberships):
            others = _member_offsets(offsets, memberships) - offsets[axis][membership]
            residual = stack.values - shared[np.newaxis, :] - others[:, np.newaxis]
            settling = _axis_offsets(stack, residual, membership, offsets[axis].size)
            moved = max(moved, float(np.max(np.abs(settling - offsets[axis]))))
            offsets[axis] = settling

        if moved < _SETTLED_DB:
            break

    return tuple(offsets)


def _member_gaps(stack: _Stack, curve: PiecewiseCurve, offsets: Series) -> tuple[float, ...]:
    """How far the written pair stands from each member, at the moment it stands furthest."""
    written = curve.at(stack.seconds)[np.newaxis, :] + offsets[:, np.newaxis]
    apart = np.where(stack.weights > 0.0, np.abs(stack.values - written), _NO_GAP_DB)
    return tuple(float(gap) for gap in np.max(apart, axis=1))


def fit_shared_trajectory(members: Sequence[TrajectoryMember], *, nodes: int) -> SharedTrajectory:
    """The curve of at most ``nodes`` corners a set of served notes is best played down by, all at once.

    A tracker hands an instrument one volume envelope and a level per sample and per pattern note, so the
    trajectories its voices make are written as one shape plus two ladders of offsets. Which is which is
    settled by reading them off each other in turn (:func:`_settled_offsets`): with the offsets fixed the
    shape is the level the members agree on at each moment, and with the shape fixed each offset is the
    level its own group holds above it. The shape that comes out is then fitted with the corners the
    format has room for (:func:`~optisample.dsp.piecewise.fit_piecewise`), and every member is measured
    against the pair -- which is what :attr:`SharedTrajectory.dispersion_db` states.

    A moment answers to the notes still sounding at it, so a shape follows the many while they last and
    the few that ring on afterwards. Every member is read from its own onset at one window length, which
    is what puts them on the clock a single envelope is played on.

    Raises:
        ValueError: when the set holds nothing to share a shape, when a member is worth nothing to the
            instrument, when the members disagree on how many axes tell them apart, or when they were read
            at window lengths that put them on different clocks.
    """
    _checked(members)
    stack = _stacked(members)
    memberships = _memberships(members)
    offsets = _settled_offsets(stack, memberships)
    reached = _member_offsets(offsets, memberships)
    curve = fit_piecewise(
        Readings(values=_shared_values(stack, reached), seconds=stack.seconds),
        nodes=nodes,
    )
    return SharedTrajectory(
        curve=curve,
        offsets_db=tuple(tuple(float(level) for level in axis_offsets) for axis_offsets in offsets),
        gaps_db=_member_gaps(stack, curve, reached),
    )
