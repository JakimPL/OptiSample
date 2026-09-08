from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Protocol, TypeVar

import numpy as np

_HULL_EPS: Final = 1e-12


class RDPoint(Protocol):
    """A byte-cost/distortion point -- what the rate-distortion hull reads off whatever it is handed."""

    @property
    def stored_bytes(self) -> int: ...

    @property
    def distortion(self) -> float: ...


_RDPointT = TypeVar("_RDPointT", bound=RDPoint)


def _slope(low: RDPoint, high: RDPoint) -> float:
    """Distortion change per byte between two points (negative: more bytes buy less distortion)."""
    return (high.distortion - low.distortion) / (high.stored_bytes - low.stored_bytes)


def pareto_frontier(points: Sequence[_RDPointT]) -> list[_RDPointT]:
    """Pareto filter: sorted by bytes ascending, keep points strictly better than all cheaper ones.

    A point is dominated (and dropped) if some cheaper point already reaches its distortion or lower;
    the ``stored_bytes`` guard also collapses ties in bytes to the lowest-distortion representative.

    This is the whole of what a byte-budget allocation may choose from. Any solution holding a dominated
    point stays feasible when the point that dominates it is put in its place, at the same distortion or
    less, so the frontier reaches every objective the full set reaches. Filtering ahead of the walk is
    what keeps a table of ``(representative, encoding)`` pairs from being priced one pair at a time
    (:func:`~optisample.optimize.grouping.cost_model.build_zone_options`), and it leaves the hull below
    untouched, since a hull vertex is Pareto-optimal to begin with.
    """
    ordered = sorted(points, key=lambda point: (point.stored_bytes, point.distortion))
    frontier: list[_RDPointT] = []
    best = np.inf
    for point in ordered:
        if point.distortion < best - _HULL_EPS and (not frontier or point.stored_bytes > frontier[-1].stored_bytes):
            frontier.append(point)
            best = point.distortion

    return frontier


def _hull_pop(frontier: Sequence[_RDPointT]) -> list[_RDPointT]:
    """Pass 2 -- convex-hull pop: drop Pareto points that sit above the chord of their neighbors.

    Walking the byte-ordered frontier, a point whose incoming slope is no steeper than the previous
    edge's marks a concave kink; pop the middle point until every successive edge gets strictly
    steeper, leaving only the vertices of the lower convex hull (a Lagrangian sweep's candidates).
    """
    hull: list[_RDPointT] = []
    for point in frontier:
        while len(hull) >= 2 and _slope(hull[-2], hull[-1]) >= _slope(hull[-1], point) - _HULL_EPS:
            hull.pop()

        hull.append(point)

    return hull


def lower_convex_hull(points: Sequence[_RDPointT]) -> list[_RDPointT]:
    """Rate-distortion frontier: Pareto-optimal points on the lower convex hull, ordered by bytes.

    Generic over the point type, so one rule serves every axis a run prices along: the loop lengths one
    recording offers, the encodings one sample is swept over, and the ``(representative, encoding)``
    options of a zone. Built in two passes: :func:`pareto_frontier` drops dominated points, then
    :func:`_hull_pop` drops the concave ones.
    """
    return _hull_pop(pareto_frontier(points))
