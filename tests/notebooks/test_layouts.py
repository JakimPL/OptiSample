from importlib.util import find_spec

import numpy as np
import pytest

from notebooks.utils import clusters, layouts
from optisample.cluster.space import pairwise_distances
from tests.notebooks.conftest import Clustered

_SEED = 0
_PLANE = 2
_BOX = 3
_WHOLE = 1.0  # what the shares of every axis a distance-preserving layout keeps come to
_STATES_NONE = 0  # shares a neighbor layout reports, its axes carrying the company a point keeps
_ACROSS = 0  # the axis a picture is read across
_UP = 1  # the axis it is read up
_AWAY = 2  # the axis a box is read into


def _space_layouts() -> tuple[layouts.Layout, ...]:
    """Every offered layout placing recordings by the space's own coordinates, which is what ``draw`` takes."""
    return tuple(layout for layout in layouts.available_layouts() if layout.reads_the_space)


def _rows(clustered: Clustered) -> list[dict[str, float | int | str | bool]]:
    """The rows a picture is placed and hovered by, read the way the notebook reads them."""
    return clusters.point_rows(clustered.described, clustered.named, clustered.distances)


def test_the_distance_preserving_pair_is_always_offered() -> None:
    """PCA and MDS come from the project's own numerical stack, so a notebook always has them to draw with."""
    offered = layouts.available_layouts()

    assert offered[:_PLANE] == (layouts.Layout.PCA, layouts.Layout.MDS)
    assert all(layout.preserves_distance for layout in offered[:_PLANE])


def test_the_keys_are_always_offered_beside_them() -> None:
    """A take's own note and velocity are there whatever is installed, so the keyboard is always drawable."""
    offered = layouts.available_layouts()

    assert layouts.Layout.KEYS in offered
    assert not layouts.Layout.KEYS.reads_the_space


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

    A layout placing by neighborhoods states its shares as none, its axes carrying the company each point
    keeps rather than a share of the space's spread.
    """
    for layout in _space_layouts():
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
    for layout in _space_layouts():
        once = layouts.draw(clustered.space.coordinates, layout=layout, components=_PLANE, seed=_SEED)
        again = layouts.draw(clustered.space.coordinates, layout=layout, components=_PLANE, seed=_SEED)

        assert once.coordinates == pytest.approx(again.coordinates)


def test_the_keys_are_placed_off_the_rows_the_recordings_carry(clustered: Clustered) -> None:
    """A take's key is what it plays rather than where the space put it, so the rows are what places it."""
    with pytest.raises(ValueError, match="key it plays"):
        layouts.draw(clustered.space.coordinates, layout=layouts.Layout.KEYS, components=_PLANE, seed=_SEED)


@pytest.mark.parametrize("components", [_PLANE, _BOX])
def test_every_layout_places_one_point_per_recording_on_the_axes_it_names(
    clustered: Clustered, components: int
) -> None:
    """One picture shape however a placement was reached, so a field is drawn and clicked the same way."""
    rows = _rows(clustered)
    for layout in layouts.available_layouts():
        placement = layouts.place(clustered.space.coordinates, rows, layout=layout, components=components, seed=_SEED)

        assert placement.components == components
        assert len(placement.titles) == components
        assert all(len(axis) == len(rows) for axis in placement.axes)
        assert placement.is_plane == (components == _PLANE)


def test_the_keys_lay_the_corpus_out_at_the_note_and_velocity_each_take_plays(clustered: Clustered) -> None:
    """The corpus as the keyboard holds it, which is what says where a group sits across the keys."""
    rows = _rows(clustered)
    placement = layouts.place(
        clustered.space.coordinates, rows, layout=layouts.Layout.KEYS, components=_PLANE, seed=_SEED
    )

    assert placement.titles == ("pitch", "velocity")
    assert placement.axes[_ACROSS] == tuple(float(row["pitch"]) for row in rows)
    assert placement.axes[_UP] == tuple(float(row["velocity"]) for row in rows)


def test_the_keys_read_in_a_box_carry_how_long_each_take_rings(clustered: Clustered) -> None:
    """A third axis puts the length beside the key, which is where a long take among short ones shows."""
    rows = _rows(clustered)
    placement = layouts.place(
        clustered.space.coordinates, rows, layout=layouts.Layout.KEYS, components=_BOX, seed=_SEED
    )

    assert placement.titles[_AWAY] == "duration (s)"
    assert placement.axes[_AWAY] == tuple(float(row["dur_s"]) for row in rows)


def test_a_layout_placing_by_neighborhoods_names_its_axes_by_their_number(clustered: Clustered) -> None:
    """A neighbor layout's axes carry the company a point keeps, so each is named by its number alone."""
    for layout in (found for found in _space_layouts() if not found.preserves_distance):
        placement = layouts.place(
            clustered.space.coordinates, _rows(clustered), layout=layout, components=_PLANE, seed=_SEED
        )

        assert placement.titles == ("component 1", "component 2")


def test_a_layout_reading_the_space_names_its_axes_for_the_spread_they_carry(clustered: Clustered) -> None:
    """A reader knows how much of the geometry a picture is showing, which is what makes it safe to read."""
    drawn = layouts.draw(clustered.space.coordinates, layout=layouts.Layout.PCA, components=_PLANE, seed=_SEED)
    placement = layouts.place(
        clustered.space.coordinates, _rows(clustered), layout=layouts.Layout.PCA, components=_PLANE, seed=_SEED
    )

    assert f"{drawn.explained[_ACROSS]:.1%}" in placement.titles[_ACROSS]
    assert placement.axes[_ACROSS] == tuple(float(value) for value in drawn.coordinates[:, _ACROSS])
