from __future__ import annotations

from dataclasses import dataclass

import pytest

from optisample.frontier import lower_convex_hull, pareto_frontier


@dataclass(frozen=True)
class _Point:
    """A byte-cost/distortion pair, which is all the hull reads off whatever it is handed."""

    stored_bytes: int
    distortion: float


@dataclass(frozen=True)
class _HullCase:
    """A lower-convex-hull scenario: the ``(stored_bytes, distortion)`` points fed in and the bytes kept."""

    name: str
    points: tuple[tuple[int, float], ...]
    kept: list[int]


_HULL_CASES = (
    _HullCase("above-chord vertex dropped", ((100, 1.0), (200, 0.5), (300, 0.45), (400, 0.1)), [100, 200, 400]),
    _HullCase("dominated vertex dropped", ((100, 1.0), (200, 0.5), (300, 0.6)), [100, 200]),
    _HullCase("collinear midpoint dropped", ((0, 3.0), (10, 2.0), (20, 1.0)), [0, 20]),
    _HullCase("ties in bytes keep the better point", ((100, 1.0), (100, 0.4), (200, 0.1)), [100, 200]),
)


@pytest.mark.parametrize("case", _HULL_CASES, ids=lambda case: case.name)
def test_lower_convex_hull_keeps_only_frontier_vertices(case: _HullCase) -> None:
    hull = lower_convex_hull([_Point(stored_bytes, distortion) for stored_bytes, distortion in case.points])
    assert [point.stored_bytes for point in hull] == case.kept


_PARETO_CASES = (
    _HullCase("dearer and worse dropped", ((100, 1.0), (200, 0.5), (300, 0.6)), [100, 200]),
    _HullCase("dearer at the same distortion dropped", ((100, 0.5), (200, 0.5)), [100]),
    _HullCase("ties in bytes keep the better point", ((100, 1.0), (100, 0.4), (200, 0.1)), [100, 200]),
)


@pytest.mark.parametrize("case", _PARETO_CASES, ids=lambda case: case.name)
def test_the_frontier_drops_every_point_something_cheaper_already_reaches(case: _HullCase) -> None:
    frontier = pareto_frontier([_Point(stored_bytes, distortion) for stored_bytes, distortion in case.points])
    assert [point.stored_bytes for point in frontier] == case.kept


def test_the_frontier_keeps_the_concave_points_the_hull_drops() -> None:
    """A byte budget may land exactly on a point no slope reaches, so a walk over bytes is handed all of them."""
    points = [
        _Point(stored_bytes, distortion)
        for stored_bytes, distortion in ((100, 1.0), (200, 0.5), (300, 0.45), (400, 0.1))
    ]
    assert [point.stored_bytes for point in pareto_frontier(points)] == [100, 200, 300, 400]
    assert [point.stored_bytes for point in lower_convex_hull(points)] == [100, 200, 400]


def test_lower_convex_hull_edge_cases() -> None:
    assert lower_convex_hull([]) == []
    solo = _Point(50, 0.2)
    assert lower_convex_hull([solo]) == [solo]


def test_every_edge_of_the_hull_buys_more_than_the_one_before_it() -> None:
    """The hull is what a Lagrangian sweep walks, so its slopes have to steepen with every step."""
    points = [_Point(bytes_, distortion) for bytes_, distortion in ((10, 9.0), (20, 5.0), (30, 4.0), (40, 3.8))]
    hull = lower_convex_hull(points)

    slopes = [
        (high.distortion - low.distortion) / (high.stored_bytes - low.stored_bytes) for low, high in zip(hull, hull[1:])
    ]
    assert slopes == sorted(slopes)
