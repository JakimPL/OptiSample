from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.dsp.series import Readings, Series

_LINE_NODES: Final = 2  # nodes a curve holds to state one straight run, which is the shortest fit there is
_FLAT_SPREAD: Final = 1e-12  # the spread in seconds a stretch holds for its own slope to be read off it
_MOST_READINGS: Final = 2048  # readings one table of stretches is filled for, which bounds the search's memory


@dataclass(frozen=True)
class CurveNode:
    """One corner a piecewise-linear curve turns at: the moment it sits at, and the value it holds there."""

    seconds: float
    value: float


@dataclass(frozen=True)
class PiecewiseCurve:
    """A curve stated as the corners it turns at, running straight between each pair and level past both ends.

    This is the shape a tracker's volume envelope plays -- a handful of nodes joined by straight lines, the
    last one held for as long as a note runs on -- so a curve fitted here becomes an envelope by reading its
    nodes onto the format's own grid. Values stand in whatever domain the readings were taken in, and a level
    trajectory is read in decibels: the domain a playback chain's gains add in and a ringing note falls
    straight in, so a handful of corners follow a whole note.
    """

    nodes: tuple[CurveNode, ...]

    @property
    def seconds(self) -> Series:
        """The moment each corner sits at, ascending, which is the timeline the curve is read on."""
        return np.asarray([node.seconds for node in self.nodes], dtype=np.float64)

    @property
    def values(self) -> Series:
        """The value each corner holds, in the order the curve turns through them."""
        return np.asarray([node.value for node in self.nodes], dtype=np.float64)

    def at(self, seconds: Series) -> Series:
        """The value the curve reads at each moment in ``seconds``, held at its end values on either side.

        Holding past both ends is what a tracker envelope does with its last node, so a curve read beyond
        the trajectory it was fitted to answers the level a held note stays at.
        """
        return np.asarray(np.interp(seconds, self.seconds, self.values), dtype=np.float64)


def _curve(seconds: Series, values: Series) -> PiecewiseCurve:
    """The curve turning at each moment in ``seconds`` at the value beside it."""
    return PiecewiseCurve(
        nodes=tuple(CurveNode(seconds=float(moment), value=float(value)) for moment, value in zip(seconds, values))
    )


def _cumulative(values: Series) -> Series:
    """Running totals of ``values`` behind a leading zero, so any stretch's total is one subtraction."""
    return np.asarray(np.concatenate(([0.0], np.cumsum(values))), dtype=np.float64)


@dataclass(frozen=True)
class _RunningTotals:
    """Every total a straight line through a stretch of readings is stated by, each behind a leading zero.

    A least-squares line asks only how many readings a stretch holds and five sums over it, so keeping those
    sums as running totals turns each stretch's fit into a handful of subtractions -- which is what lets a
    search price every stretch there is.
    """

    seconds: Series
    seconds_squared: Series
    values: Series
    values_squared: Series
    joint: Series

    def line_errors(self, opening: int) -> Series:
        """The error one straight line leaves on each stretch opening at ``opening``, closing later each time.

        A stretch whose readings all centre on one moment carries no slope of its own, so the line through it
        is the level they average to.
        """
        closes = slice(opening + _LINE_NODES, None)
        held = np.arange(_LINE_NODES, self.values.size - opening, dtype=np.float64)
        span_seconds = self.seconds[closes] - self.seconds[opening]
        span_values = self.values[closes] - self.values[opening]
        spread = self.seconds_squared[closes] - self.seconds_squared[opening] - span_seconds**2 / held
        swing = self.values_squared[closes] - self.values_squared[opening] - span_values**2 / held
        joint = self.joint[closes] - self.joint[opening] - span_seconds * span_values / held
        sloped = np.divide(joint**2, spread, out=np.zeros_like(joint), where=spread > _FLAT_SPREAD)
        return np.asarray(np.maximum(swing - sloped, 0.0), dtype=np.float64)


def _running_totals(readings: Readings) -> _RunningTotals:
    """The totals every stretch of ``readings`` has its own straight line read off."""
    return _RunningTotals(
        seconds=_cumulative(readings.seconds),
        seconds_squared=_cumulative(readings.seconds**2),
        values=_cumulative(readings.values),
        values_squared=_cumulative(readings.values**2),
        joint=_cumulative(readings.seconds * readings.values),
    )


def _straight_errors(readings: Readings) -> Series:
    """The squared error one straight line leaves on every stretch of ``readings``, read as ``[opening, close]``.

    Stretches holding a single reading and stretches running backwards read as infinity, which keeps every
    corner of a fitted curve a step forward along the timeline.
    """
    count = readings.count
    totals = _running_totals(readings)
    table = np.full((count, count), np.inf, dtype=np.float64)
    for opening in range(count - 1):
        table[opening, opening + 1 :] = totals.line_errors(opening)

    return table


def _corner_indices(errors: Series, segments: int) -> tuple[int, ...]:
    """The readings a curve turns at: the run of ``segments`` straight stretches leaving the least error.

    Every stretch's error is known up front (:func:`_straight_errors`), so the cheapest way of reaching a
    reading with a given number of stretches behind it follows from the cheapest way of reaching each
    earlier one. That makes the whole placement a walk forward over the table -- exact over every run of
    corners there is, where growing a curve one corner at a time reads only the ones that pay off alone.

    Corners come back ascending, opening on the first reading and closing on the last.
    """
    count = errors.shape[0]
    reached = np.full((segments + 1, count), np.inf, dtype=np.float64)
    opened = np.zeros((segments + 1, count), dtype=np.intp)
    reached[0, 0] = 0.0
    for depth in range(1, segments + 1):
        candidates = reached[depth - 1][:, np.newaxis] + errors
        reached[depth] = np.min(candidates, axis=0)
        opened[depth] = np.argmin(candidates, axis=0)

    corners = [count - 1]
    for depth in range(segments, 0, -1):
        corners.append(int(opened[depth, corners[-1]]))

    return tuple(reversed(corners))


def _tent_basis(seconds: Series, corners: Series) -> Series:
    """One shape per corner: full at its own corner, falling straight to nothing at each neighbour.

    A curve through fixed corners is the sum of these scaled by the values those corners hold, so writing it
    this way turns the fit into a linear system the corner values solve.
    """
    unit = np.eye(corners.size, dtype=np.float64)
    return np.asarray(np.stack([np.interp(seconds, corners, shape) for shape in unit], axis=1), dtype=np.float64)


def _corner_values(readings: Readings, corners: Series) -> Series:
    """The values the corners hold to run the curve closest to every reading at once.

    With the corners fixed the curve is linear in the values they hold, so the least-squares choice is the
    solution of a system the size of the corner count -- exact, continuous through every corner, and cheap
    beside the walk that placed them.
    """
    basis = _tent_basis(readings.seconds, corners)
    return np.asarray(np.linalg.solve(basis.T @ basis, basis.T @ readings.values), dtype=np.float64)


def fit_piecewise(readings: Readings, *, nodes: int) -> PiecewiseCurve:
    """The curve of at most ``nodes`` corners running closest to ``readings``.

    Where the corners go is settled first, by the walk that measures every stretch of the trajectory against
    the straight line through it and keeps the cheapest run of them (:func:`_corner_indices`). What each one
    holds follows from the exact solve at those positions (:func:`_corner_values`), so the curve that comes
    back is continuous and least-squares optimal for the corners it turns at. A trajectory holding no more
    readings than the nodes asked for is stated by its own readings, which runs the curve through every one.

    Placement measures every stretch of the trajectory against the line through it, so it fills a table
    quadratic in the reading count. A trajectory is therefore read in at most :data:`_MOST_READINGS`
    windows, which holds that table inside a few tens of megabytes and leaves a curve of a few dozen
    corners every resolution it can express -- a long note is read in proportionally longer windows, which
    is the caller's to choose because only the caller knows how long the note runs.

    Raises:
        ValueError: when fewer than two nodes are asked for, when the readings hold nothing to fit, or when
            they run past :data:`_MOST_READINGS`.
    """
    if nodes < _LINE_NODES:
        raise ValueError(f"a curve turns through at least {_LINE_NODES} nodes, against the {nodes} asked for")

    if readings.count == 0:
        raise ValueError("a curve is fitted to at least one reading")

    if readings.count > _MOST_READINGS:
        raise ValueError(f"a trajectory is read in at most {_MOST_READINGS} windows, against {readings.count}")

    if readings.count <= nodes:
        return _curve(readings.seconds, readings.values)

    corners = readings.seconds[np.asarray(_corner_indices(_straight_errors(readings), nodes - 1), dtype=np.intp)]
    return _curve(corners, _corner_values(readings, corners))
