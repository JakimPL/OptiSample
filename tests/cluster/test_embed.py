from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from optisample.cluster.embed import classical_scaling, principal_components
from optisample.cluster.space import Coordinates, pairwise_distances

_SAMPLES = 24  # recordings the layouts are read across
_DIMENSIONS = 6  # coordinates a point stands on before it is laid out
_STRETCH = (9.0, 4.0, 2.0, 1.0, 0.5, 0.2)  # how far the material reaches along each of those coordinates
_SEED = 23  # the entropy the scattered material is drawn from
_SHOWN = 3  # components a picture of the space holds


@pytest.fixture
def coordinates() -> Coordinates:
    """Material reaching much further along some coordinates than others, so the axes come out in an order."""
    return np.asarray(
        np.random.default_rng(_SEED).normal(size=(_SAMPLES, _DIMENSIONS)) * np.asarray(_STRETCH),
        dtype=np.float64,
    )


def test_the_widest_axis_comes_first(coordinates: Coordinates) -> None:
    explained = principal_components(coordinates, _DIMENSIONS).explained

    assert list(explained) == sorted(explained, reverse=True)


def test_keeping_every_axis_carries_the_whole_spread(coordinates: Coordinates) -> None:
    layout = principal_components(coordinates, _DIMENSIONS)

    assert layout.components == _DIMENSIONS
    assert layout.covered == pytest.approx(1.0)


def test_a_picture_of_a_few_axes_says_how_much_of_the_space_it_shows(coordinates: Coordinates) -> None:
    layout = principal_components(coordinates, _SHOWN)

    assert layout.components == _SHOWN
    assert 0.0 < layout.covered < 1.0


def test_the_principal_components_hold_the_distances_the_space_states(coordinates: Coordinates) -> None:
    """Turning the space onto its own axes moves nothing, which is what has a picture stand for a geometry."""
    layout = principal_components(coordinates, _DIMENSIONS)

    assert np.allclose(pdist(layout.coordinates), pdist(coordinates))


def test_classical_scaling_of_euclidean_distances_reproduces_the_principal_components(
    coordinates: Coordinates,
) -> None:
    """One geometry read two ways, the axes free to point either direction along themselves."""
    scaled = classical_scaling(pairwise_distances(coordinates), _SHOWN)
    components = principal_components(coordinates, _SHOWN)

    assert np.allclose(np.abs(scaled.coordinates), np.abs(components.coordinates))
    assert np.allclose(scaled.explained, components.explained)


def test_classical_scaling_places_points_at_the_distances_it_was_given(coordinates: Coordinates) -> None:
    distances = pairwise_distances(coordinates)

    assert np.allclose(pdist(classical_scaling(distances, _DIMENSIONS).coordinates), pdist(coordinates))


def test_asking_for_more_axes_than_the_space_holds_reads_what_is_there(coordinates: Coordinates) -> None:
    assert principal_components(coordinates, _DIMENSIONS * 2).components == _DIMENSIONS
    assert classical_scaling(pairwise_distances(coordinates), _SAMPLES * 2).components == _SAMPLES


def test_a_corpus_standing_at_one_point_carries_no_spread() -> None:
    """Material a layout can tell nothing apart in states zero on every axis, which is what it has to show."""
    coordinates = np.zeros((5, 3))

    assert principal_components(coordinates, _SHOWN).covered == 0.0
    assert classical_scaling(pairwise_distances(coordinates), _SHOWN).covered == 0.0
