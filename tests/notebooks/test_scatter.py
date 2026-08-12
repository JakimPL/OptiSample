import numpy as np
import pytest

from notebooks.utils import clusters, layouts, scatter
from optisample.cluster.partition import cut, hierarchy, sweep
from optisample.config import OptiConfig
from tests.notebooks.conftest import Clustered

_SEED = 0
_PLANE = 2
_BOX = 3
_REPRESENTATIVES = 1  # the one trace the takes standing for their groups are drawn over the field in
_LEAVES = 5
_INDEX = 0  # where a point's place in the space sits among the values it carries


def _placed(clustered: Clustered, components: int) -> layouts.Embedding:
    """The corpus laid out on the axes carrying most of its spread, which is what the panels draw."""
    return layouts.draw(clustered.space.coordinates, layout=layouts.Layout.PCA, components=components, seed=_SEED)


def _rows(clustered: Clustered) -> list[dict[str, float | int | str | bool]]:
    """The rows the scatter hovers, read the way the notebook reads them."""
    return clusters.point_rows(clustered.described, clustered.groups, clustered.distances, rule=clustered.rule)


@pytest.mark.parametrize("components", [_PLANE, _BOX])
def test_a_named_column_draws_one_trace_per_name_beside_the_representatives(
    clustered: Clustered, components: int
) -> None:
    """A column holding names draws one colour and one legend entry apiece, so a click isolates a group."""
    rows = _rows(clustered)
    representatives = [group.representative(clustered.rule) for group in clustered.groups]
    figure = scatter.space_scatter(
        _placed(clustered, components),
        rows,
        colour_by="group",
        representatives=representatives,
        title="by group",
    )

    assert len(figure.data) == len(clustered.groups) + _REPRESENTATIVES
    assert sum(len(trace.x) for trace in figure.data[:-1]) == len(rows)
    assert len(figure.data[-1].x) == len(representatives)


def test_a_measured_column_draws_one_trace_shaded_along_a_scale(clustered: Clustered) -> None:
    """A column holding measurements is shaded beside the picture rather than split into named sets."""
    rows = _rows(clustered)
    figure = scatter.space_scatter(
        _placed(clustered, _PLANE), rows, colour_by="pitch", representatives=[0], title="by pitch"
    )

    assert len(figure.data) == 1 + _REPRESENTATIVES
    assert list(figure.data[0].marker.color) == [row["pitch"] for row in rows]


def test_every_point_carries_its_place_in_the_space(clustered: Clustered) -> None:
    """A click hands back a place in the corpus, so what is examined is the take that was drawn."""
    rows = _rows(clustered)
    figure = scatter.space_scatter(
        _placed(clustered, _PLANE), rows, colour_by="group", representatives=[1], title="places"
    )
    carried = sorted(int(point[_INDEX]) for trace in figure.data[:-1] for point in trace.customdata)

    assert carried == list(range(len(rows)))


def test_the_axes_state_the_share_of_the_spread_they_carry(clustered: Clustered) -> None:
    """A reader knows how much of the geometry a picture is showing, which is what makes it safe to read."""
    placed = _placed(clustered, _PLANE)
    figure = scatter.space_scatter(placed, _rows(clustered), colour_by="group", representatives=[0], title="axes")

    assert f"{placed.explained[0]:.1%}" in figure.layout.xaxis.title.text
    assert "component 2" in figure.layout.yaxis.title.text


def test_a_layout_stating_no_shares_names_its_axes_plainly(clustered: Clustered) -> None:
    """A neighbour layout's axes carry the company a point keeps, so they are named without a share."""
    placed = layouts.Embedding(
        coordinates=_placed(clustered, _PLANE).coordinates, explained=np.zeros(0, dtype=np.float64)
    )
    figure = scatter.space_scatter(placed, _rows(clustered), colour_by="group", representatives=[0], title="plain")

    assert figure.layout.xaxis.title.text == "component 1"


def test_a_picture_in_three_dimensions_names_all_three(clustered: Clustered) -> None:
    """A box is drawn on the scene's own axes, so each of the three says what it carries."""
    figure = scatter.space_scatter(
        _placed(clustered, _BOX), _rows(clustered), colour_by="group", representatives=[0], title="box"
    )

    assert figure.layout.scene.zaxis.title.text.startswith("component 3")


def test_a_click_reads_back_the_recordings_it_landed_on() -> None:
    """The place travels with the point, so a selection comes back as indices into the very corpus drawn."""
    clicked = [{"customdata": [7, "0007_p060"], "pointNumber": 3}, {"customdata": [2, "0002_p048"]}]

    assert scatter.selected(clicked) == (7, 2)


def test_a_click_landing_on_a_point_carrying_nothing_names_no_recording() -> None:
    """A payload without the values a point was drawn with picks nothing, leaving the panel as it stood."""
    assert scatter.selected([{"pointNumber": 3}]) == ()


def test_the_sweep_curve_marks_the_count_on_screen(clustered: Clustered, config: OptiConfig) -> None:
    """The curve is what a count is picked off, and the one being shown is ringed on it."""
    climbed = sweep(clustered.space.coordinates, config.cluster.partition)
    chosen = len(clustered.groups)
    figure = scatter.sweep_curve(clusters.sweep_rows(climbed, chosen=chosen), title="separation")

    assert list(figure.data[0].x) == [found.groups for found in climbed]
    assert list(figure.data[1].x) == [chosen]


def test_the_height_a_cut_reads_a_tree_at_leaves_the_count_it_asked_for(
    clustered: Clustered, config: OptiConfig
) -> None:
    """Cutting the tree at that height is what the group count means, so reading it back states the count."""
    tree = hierarchy(clustered.space.coordinates, config.cluster.partition.linkage)
    for groups in range(2, clustered.space.samples):
        height = scatter.cut_height(tree, groups)

        assert len(set(cut(tree, groups).tolist())) == groups
        assert int((tree[:, 2] < height).sum()) == clustered.space.samples - groups


def test_the_dendrogram_draws_the_top_of_the_tree_and_the_height_the_cut_reads_it_at(
    clustered: Clustered, config: OptiConfig
) -> None:
    """Showing the top is what makes a tree of a few hundred takes readable, with the cut drawn across it."""
    tree = hierarchy(clustered.space.coordinates, config.cluster.partition.linkage)
    groups = len(clustered.groups)
    figure = scatter.dendrogram_figure(tree, groups=groups, leaves=_LEAVES, title="tree")

    assert len(figure.data) == 1
    assert np.isnan(figure.data[0].x).sum() == _LEAVES - 1
    assert figure.layout.shapes[0].y0 == pytest.approx(scatter.cut_height(tree, groups))
