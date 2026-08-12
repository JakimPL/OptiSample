from importlib.util import find_spec

import numpy as np
import pytest

from notebooks.utils import layouts
from optisample.cluster.space import pairwise_distances
from tests.notebooks.conftest import Clustered

_SEED = 0
_PLANE = 2
_BOX = 3
_WHOLE = 1.0  # what the shares of every axis a distance-preserving layout keeps come to
_STATES_NONE = 0  # shares a neighbour layout reports, its axes carrying the company a point keeps


def test_the_distance_preserving_pair_is_always_offered() -> None:
    """PCA and MDS come from the project's own numerical stack, so a notebook always has them to draw with."""
    offered = layouts.available_layouts()

    assert offered[:_PLANE] == (layouts.Layout.PCA, layouts.Layout.MDS)
    assert all(layout.preserves_distance for layout in offered[:_PLANE])


def test_a_variant_is_offered_where_the_library_behind_it_is_installed() -> None:
    """Each variant is probed as it is asked for, so what is listed is what this installation can draw."""
    offered = layouts.available_layouts()

    assert (layouts.Layout.TSNE in offered) == (find_spec("sklearn") is not None)
    assert (layouts.Layout.UMAP in offered) == (find_spec("umap") is not None)


@pytest.mark.parametrize("components", [_PLANE, _BOX])
def test_a_layout_places_every_recording_in_the_dimensions_it_was_asked_for(
    clustered: Clustered, components: int
) -> None:
    """One row per recording however the placement was reached, so a picture indexes the corpus.

    A layout placing by neighbourhoods states its shares as none, its axes carrying the company each point
    keeps rather than a share of the space's spread.
    """
    for layout in layouts.available_layouts():
        placed = layouts.draw(clustered.space.coordinates, layout=layout, components=components, seed=_SEED)

        assert placed.coordinates.shape == (clustered.space.samples, components)
        assert placed.components == components
        assert np.isfinite(placed.coordinates).all()
        if not layout.preserves_distance:
            assert placed.explained.size == _STATES_NONE


def test_the_two_distance_preserving_layouts_read_one_geometry(clustered: Clustered) -> None:
    """Classical scaling of Euclidean distances places points where the principal components put them.

    The two differ by a turn of the axes, so what they agree on exactly is the distance between any pair --
    which is what makes either safe to read a group off.
    """
    turned = layouts.draw(clustered.space.coordinates, layout=layouts.Layout.PCA, components=_BOX, seed=_SEED)
    scaled = layouts.draw(clustered.space.coordinates, layout=layouts.Layout.MDS, components=_BOX, seed=_SEED)

    assert pairwise_distances(turned.coordinates) == pytest.approx(pairwise_distances(scaled.coordinates), abs=1e-8)
    assert turned.covered == pytest.approx(scaled.covered)


def test_keeping_every_axis_carries_the_whole_spread(clustered: Clustered) -> None:
    """A layout given as many axes as the space has states that it is showing all of it."""
    placed = layouts.draw(
        clustered.space.coordinates,
        layout=layouts.Layout.PCA,
        components=clustered.space.dimensions,
        seed=_SEED,
    )

    assert placed.covered == pytest.approx(_WHOLE)


def test_one_space_read_twice_draws_one_picture(clustered: Clustered) -> None:
    """Every layout is settled from the seed it was given, so a finding on screen is there again next time."""
    for layout in layouts.available_layouts():
        once = layouts.draw(clustered.space.coordinates, layout=layout, components=_PLANE, seed=_SEED)
        again = layouts.draw(clustered.space.coordinates, layout=layout, components=_PLANE, seed=_SEED)

        assert once.coordinates == pytest.approx(again.coordinates)
