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
_FIELD = slice(0, -1)  # the traces drawing the corpus itself, the rings over them going last
_RINGS = -1  # where that ring trace sits among the traces of a picture
_LEAVES = 5
_FIRST = 0  # the trace a picture opens with, and the point a trace opens with
_SECOND = 1  # the one after it
_STOOD = 2  # the recording a panel was already reading when a picture named none
_HAIRLINE = 0  # the width the run joining a flat trace's markers is stroked at


def _rows(clustered: Clustered) -> list[dict[str, float | int | str | bool]]:
    """The rows the scatter hovers, read the way the notebook reads them."""
    return clusters.point_rows(clustered.described, clustered.named, clustered.distances)


def _placed(clustered: Clustered, components: int, layout: layouts.Layout = layouts.Layout.PCA) -> layouts.Placement:
    """The corpus where a layout puts it, which is what the panels draw."""
    return layouts.place(
        clustered.space.coordinates, _rows(clustered), layout=layout, components=components, seed=_SEED
    )


def _coloring(clustered: Clustered, column: str) -> scatter.Coloring:
    """What a field is drawn by, carrying the color each group's own key turned, as the notebook builds it."""
    return scatter.Coloring(column=column, groups=clusters.group_colors(clustered.named))


def _picture(
    clustered: Clustered,
    column: str,
    *,
    components: int = _PLANE,
    layout: layouts.Layout = layouts.Layout.PCA,
) -> scatter.Picture:
    """One whole field as the notebook draws it: every recording, ringed and colored, ready to be clicked."""
    return scatter.field(
        _placed(clustered, components, layout),
        _rows(clustered),
        coloring=_coloring(clustered, column),
        representatives=[group.representative for group in clustered.groups],
        title=column,
    )


@pytest.mark.parametrize("components", [_PLANE, _BOX])
def test_a_named_column_draws_one_trace_per_name_beside_the_representatives(
    clustered: Clustered, components: int
) -> None:
    """A column holding names draws one color and one legend entry apiece, so a click isolates a group."""
    picture = _picture(clustered, "group", components=components)
    traces = picture.figure.data

    assert len(traces) == len(clustered.groups) + _REPRESENTATIVES
    assert sum(len(trace.x) for trace in traces[_FIELD]) == len(_rows(clustered))
    assert len(traces[_RINGS].x) == len(clustered.groups)


def test_a_measured_column_draws_one_trace_shaded_along_a_scale(clustered: Clustered) -> None:
    """A column holding measurements is shaded beside the picture rather than split into named sets."""
    rows = _rows(clustered)
    picture = _picture(clustered, "pitch")

    assert len(picture.figure.data) == 1 + _REPRESENTATIVES
    assert list(picture.figure.data[_FIRST].marker.color) == [row["pitch"] for row in rows]


def test_a_field_colored_by_group_draws_each_one_in_its_own_keys_color(clustered: Clustered) -> None:
    """The keyboard is what the field carries, so a legend entry is the color its group's key turned."""
    picture = _picture(clustered, "group")
    drawn = {trace.name: trace.marker.color for trace in picture.figure.data[_FIELD]}

    assert drawn == clusters.group_colors(clustered.named)


def test_a_field_colored_by_another_name_is_handed_the_palette(clustered: Clustered) -> None:
    """A column saying nothing about the keys is told apart by the palette rather than by a key's color."""
    picture = _picture(clustered, "role")
    drawn = {str(trace.marker.color) for trace in picture.figure.data[_FIELD]}

    assert drawn.isdisjoint(set(clusters.group_colors(clustered.named).values()))


def test_a_flat_trace_is_drawn_as_the_run_of_markers_a_click_comes_back_from(clustered: Clustered) -> None:
    """marimo forwards a click from a scatter drawn with lines, so a flat field joins its markers at no width."""
    picture = _picture(clustered, "group")

    assert all("lines" in trace.mode for trace in picture.figure.data)
    assert all(trace.line.width == _HAIRLINE for trace in picture.figure.data)


def test_a_field_drawn_in_a_box_is_read_by_eye(clustered: Clustered) -> None:
    """A scene draws its points alone, so a box shows the markers and the panels are told which by name."""
    picture = _picture(clustered, "group", components=_BOX)

    assert all(trace.mode == "markers" for trace in picture.figure.data)


def test_every_recording_is_drawn_once_under_the_rings_over_them(clustered: Clustered) -> None:
    """The picture holds which recording each trace drew, which is what a click is turned into a take by."""
    picture = _picture(clustered, "group")
    field = sorted(place for places in picture.places[_FIELD] for place in places)

    assert field == list(range(len(_rows(clustered))))
    assert picture.places[_RINGS] == tuple(group.representative for group in clustered.groups)


def test_the_axes_carry_the_names_the_placement_gave_them(clustered: Clustered) -> None:
    """A reader knows how much of the geometry a picture is showing, which is what makes it safe to read."""
    placement = _placed(clustered, _PLANE)
    picture = _picture(clustered, "group")

    assert picture.figure.layout.xaxis.title.text == placement.titles[_FIRST]
    assert picture.figure.layout.yaxis.title.text == placement.titles[_SECOND]


def test_a_picture_in_three_dimensions_names_all_three(clustered: Clustered) -> None:
    """A box is drawn on the scene's own axes, so each of the three says what it carries."""
    picture = _picture(clustered, "group", components=_BOX)

    assert picture.figure.layout.scene.zaxis.title.text.startswith("component 3")


def test_the_keys_are_drawn_on_the_axes_the_keyboard_names(clustered: Clustered) -> None:
    """The keys layout is drawn as any other field is, so the corpus reads against the keyboard it came from."""
    picture = _picture(clustered, "group", layout=layouts.Layout.KEYS)

    assert picture.figure.layout.xaxis.title.text == "pitch"
    assert picture.figure.layout.yaxis.title.text == "velocity"


@pytest.mark.parametrize("column", ["group", "pitch"])
def test_the_take_standing_for_a_group_is_ringed_in_that_groups_color(clustered: Clustered, column: str) -> None:
    """A ring says which take was chosen and which group chose it, whatever the field is colored by."""
    rows = _rows(clustered)
    picture = _picture(clustered, column)
    colors = clusters.group_colors(clustered.named)
    ringed = picture.figure.data[_RINGS]

    assert list(ringed.marker.color) == [colors[str(rows[group.representative]["group"])] for group in clustered.groups]
    assert ringed.marker.size > picture.figure.data[_FIRST].marker.size
    assert "open" in ringed.marker.symbol


def test_a_click_reads_back_the_recording_the_trace_drew_where_it_landed(clustered: Clustered) -> None:
    """A point comes back naming its trace and its place inside it, which the picture turns into a take."""
    picture = _picture(clustered, "group")
    clicked = [
        {"curveNumber": _FIRST, "pointNumber": _SECOND},
        {"curveNumber": _SECOND, "pointIndex": _FIRST},
    ]

    assert scatter.selected(picture, clicked) == (picture.places[_FIRST][_SECOND], picture.places[_SECOND][_FIRST])


def test_a_click_on_the_keys_reads_back_the_corpus_the_space_reads_back(clustered: Clustered) -> None:
    """Both layouts draw the same recordings in the same traces, so either picks a take out of one corpus."""
    keyed = _picture(clustered, "group", layout=layouts.Layout.KEYS)
    placed = _picture(clustered, "group")

    assert keyed.places == placed.places


def test_a_click_landing_where_the_picture_drew_nothing_names_no_recording(clustered: Clustered) -> None:
    """A payload naming no drawn point picks nothing, which leaves the panels reading what they stood on."""
    picture = _picture(clustered, "group")

    assert scatter.selected(picture, [{"pointNumber": _FIRST}]) == ()
    assert scatter.selected(picture, [{"curveNumber": len(picture.places), "pointNumber": _FIRST}]) == ()
    assert scatter.selected(picture, [{"curveNumber": _FIRST, "pointNumber": len(picture.places[_FIRST])}]) == ()


def test_a_picture_naming_no_recording_leaves_the_one_that_stood(clustered: Clustered) -> None:
    """A redrawn picture hands back nothing, so the take a reader picked last is what the panels keep on."""
    picture = _picture(clustered, "group")

    assert scatter.picked(picture, [{"curveNumber": _FIRST, "pointNumber": _FIRST}], _STOOD) == picture.places[0][0]
    assert scatter.picked(picture, [], _STOOD) == _STOOD


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
