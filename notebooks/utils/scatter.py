from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from scipy.cluster.hierarchy import dendrogram

from notebooks.utils.views import Row
from optisample.cluster.embed import Embedding
from optisample.cluster.partition import Tree

SelectionPoint = Mapping[str, Any]  # one clicked point as plotly serialized it and marimo handed it back

_PALETTE: Final = tuple(qualitative.Dark24)
_CONTINUOUS: Final = "Viridis"  # the scale a reading with an order of its own is drawn along
_INDEX_COLUMN: Final = 0  # where a point's place in the space sits among the values it carries
_FIRST_VALUE: Final = 1  # where the readings a hover lists start, the place having gone first
_CUSTOM_DATA: Final = "customdata"  # what plotly calls the values a point carries beside its position
_PLANE: Final = 2  # components a picture drawn on a plane holds
_MARKER_SIZE: Final = 7  # how big a recording is drawn on a plane
_SOLID_MARKER_SIZE: Final = 4  # how big it is drawn in a box, where depth already thins the field
_REPRESENTATIVE_SIZE: Final = 13  # how big the take standing for a group is drawn
_REPRESENTATIVE_SYMBOL: Final = "diamond-open"  # the one outline both the plane and the box draw it with
_LINE_WIDTH: Final = 1.6
_HEIGHT: Final = 620  # pixels a scatter is given, ample for a group to be picked out by eye
_CURVE_HEIGHT: Final = 320  # pixels the two reading curves are given
_FLOAT_FORMAT: Final = ":.3f"  # how a measured reading is spelled in a hover
_PLAIN: Final = ""  # how a name or a count is spelled there
_OUTLINE: Final = "#111111"
_CUT_COLOUR: Final = "#d62728"
_LINK_COLOUR: Final = "#4c78a8"
_LEAF_SCALE: Final = 10.0  # the width scipy spaces dendrogram leaves by, which its own coordinates count in
_MERGES_TO_LEAVES: Final = 1  # the one leaf a tree holds beyond the merges that joined them
_LAST_KEPT: Final = 1  # the step back from the first merge a cut drops to the last one it keeps


def _formats(rows: Sequence[Row], keys: Sequence[str]) -> tuple[str, ...]:
    """How each reading is spelled in a hover: measured values to three places, names and counts as they are."""
    return tuple(_FLOAT_FORMAT if isinstance(rows[0][key], float) else _PLAIN for key in keys)


def _hover(rows: Sequence[Row], keys: Sequence[str]) -> str:
    """The tooltip one point shows, listing every reading its row carries."""
    formats = _formats(rows, keys)
    lines = [f"{key}: %{{customdata[{index + _FIRST_VALUE}]{formats[index]}}}" for index, key in enumerate(keys)]
    return "<br>".join(lines) + "<extra></extra>"


def _carried(rows: Sequence[Row], keys: Sequence[str]) -> list[list[object]]:
    """What each point carries beside its position: its place in the space, then every reading it was read for."""
    return [[index, *(row[key] for key in keys)] for index, row in enumerate(rows)]


def _positions(embedding: Embedding, places: Sequence[int]) -> dict[str, Sequence[float]]:
    """Where the named recordings sit, as the axis arguments a plotly trace is built from."""
    axes = ("x", "y", "z")
    return {
        axes[axis]: [float(embedding.coordinates[place, axis]) for place in places]
        for axis in range(embedding.components)
    }


def _trace(embedding: Embedding, *, places: Sequence[int], **options: object) -> go.Scatter | go.Scatter3d:
    """One set of recordings drawn where the layout put them, on a plane or in a box as it has axes for."""
    positions = _positions(embedding, places)
    if embedding.components <= _PLANE:
        return go.Scatter(mode="markers", **positions, **options)

    return go.Scatter3d(mode="markers", **positions, **options)


def _named_traces(
    embedding: Embedding,
    rows: Sequence[Row],
    carried: Sequence[Sequence[object]],
    hover: str,
    colour_by: str,
) -> list[go.Scatter | go.Scatter3d]:
    """One trace per value the colouring column names, so each reads as its own colour and legend entry."""
    size = _MARKER_SIZE if embedding.components <= _PLANE else _SOLID_MARKER_SIZE
    names = sorted({str(row[colour_by]) for row in rows})
    return [
        _trace(
            embedding,
            places=[index for index, row in enumerate(rows) if str(row[colour_by]) == name],
            name=name,
            marker={"size": size, "color": _PALETTE[position % len(_PALETTE)]},
            customdata=[carried[index] for index, row in enumerate(rows) if str(row[colour_by]) == name],
            hovertemplate=hover,
        )
        for position, name in enumerate(names)
    ]


def _scaled_trace(
    embedding: Embedding,
    rows: Sequence[Row],
    carried: Sequence[Sequence[object]],
    hover: str,
    colour_by: str,
) -> go.Scatter | go.Scatter3d:
    """Every recording in one trace, shaded along the reading the colouring column holds."""
    size = _MARKER_SIZE if embedding.components <= _PLANE else _SOLID_MARKER_SIZE
    return _trace(
        embedding,
        places=range(len(rows)),
        name=colour_by,
        marker={
            "size": size,
            "color": [float(row[colour_by]) for row in rows],
            "colorscale": _CONTINUOUS,
            "showscale": True,
            "colorbar": {"title": colour_by},
        },
        customdata=list(carried),
        hovertemplate=hover,
    )


def _representative_trace(
    embedding: Embedding,
    carried: Sequence[Sequence[object]],
    hover: str,
    representatives: Sequence[int],
) -> go.Scatter | go.Scatter3d:
    """The takes standing for their groups, drawn over the field so a selection reads at a glance."""
    return _trace(
        embedding,
        places=representatives,
        name="stands for its group",
        marker={
            "size": _REPRESENTATIVE_SIZE,
            "symbol": _REPRESENTATIVE_SYMBOL,
            "color": _OUTLINE,
            "line": {"width": _LINE_WIDTH, "color": _OUTLINE},
        },
        customdata=[carried[place] for place in representatives],
        hovertemplate=hover,
    )


def space_scatter(
    embedding: Embedding,
    rows: Sequence[Row],
    *,
    colour_by: str,
    representatives: Sequence[int],
    title: str,
) -> go.Figure:
    """Every recording where the layout placed it, shaded by one of its readings and hovering all of them.

    A column holding names -- the group a take fell into, the note it plays -- draws one colour and one
    legend entry per name, so a click on the legend isolates that set; a column holding measurements is
    shaded along a scale beside the picture. The takes standing for their groups are drawn over the top,
    and every point carries its place in the space, which is what a click hands back to Python.
    """
    keys = list(rows[0]) if rows else []
    carried = _carried(rows, keys)
    hover = _hover(rows, keys) if rows else _PLAIN
    named = isinstance(rows[0][colour_by], str) if rows else False
    traces = (
        _named_traces(embedding, rows, carried, hover, colour_by)
        if named
        else [_scaled_trace(embedding, rows, carried, hover, colour_by)]
    )
    figure = go.Figure(data=[*traces, _representative_trace(embedding, carried, hover, representatives)])
    figure.update_layout(
        title=title,
        height=_HEIGHT,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"itemsizing": "constant"},
        clickmode="event+select",
    )
    _label_axes(figure, embedding)
    return figure


def _axis_title(embedding: Embedding, axis: int) -> str:
    """What one drawn axis is called, carrying the share of the spread it holds where the layout states one."""
    if axis < embedding.explained.size:
        return f"component {axis + 1} — {embedding.explained[axis]:.1%}"

    return f"component {axis + 1}"


def _label_axes(figure: go.Figure, embedding: Embedding) -> None:
    """Name every drawn axis on the figure, on the plane or on the box as the layout has axes for."""
    titles = [_axis_title(embedding, axis) for axis in range(embedding.components)]
    if embedding.components <= _PLANE:
        figure.update_layout(xaxis_title=titles[0], yaxis_title=titles[1])
        return

    figure.update_layout(scene={"xaxis_title": titles[0], "yaxis_title": titles[1], "zaxis_title": titles[2]})


def selected(points: Sequence[SelectionPoint]) -> tuple[int, ...]:
    """Which recordings a click on the scatter picked out, as places in the space.

    Each point carries its place among the values it was drawn with, so a selection comes back as indices
    into the very corpus the picture was built from whichever trace the click landed on.
    """
    return tuple(int(point[_CUSTOM_DATA][_INDEX_COLUMN]) for point in points if _CUSTOM_DATA in point)


def sweep_curve(rows: Sequence[Row], *, title: str) -> go.Figure:
    """What every group count the sweep climbed is worth, with the one on screen marked.

    The curve is what a reader picks a count off: its top is the number of groups this corpus tells apart
    most cleanly, and a flat stretch says several counts read the material about as well as each other.
    """
    counts = [int(row["groups"]) for row in rows]
    scores = [float(row["silhouette"]) for row in rows]
    on_screen = [index for index, row in enumerate(rows) if bool(row["on_screen"])]
    figure = go.Figure(
        data=[
            go.Scatter(x=counts, y=scores, mode="lines+markers", name="silhouette", line={"color": _LINK_COLOUR}),
            go.Scatter(
                x=[counts[index] for index in on_screen],
                y=[scores[index] for index in on_screen],
                mode="markers",
                name="on screen",
                marker={"size": _REPRESENTATIVE_SIZE, "symbol": "circle-open", "color": _CUT_COLOUR},
            ),
        ]
    )
    figure.update_layout(
        title=title,
        height=_CURVE_HEIGHT,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        xaxis_title="groups",
        yaxis_title="silhouette",
    )
    return figure


def cut_height(tree: Tree, groups: int) -> float:
    """The height ``tree`` is read at to leave ``groups`` standing, midway between the merges it separates.

    A hierarchy is cut by keeping every merge under one height, so the height leaving a count standing sits
    between the last merge kept and the first one dropped -- which is where a dendrogram draws the line.
    """
    merges = int(tree.shape[0])
    dropped = min(max(merges + _MERGES_TO_LEAVES - groups, _LAST_KEPT), merges - _LAST_KEPT)
    return float((tree[dropped - _LAST_KEPT, 2] + tree[dropped, 2]) / 2.0)


def _link_lines(icoord: Sequence[Sequence[float]], dcoord: Sequence[Sequence[float]]) -> tuple[list[float], ...]:
    """Every merge of a drawn tree as one run of points, the links parted by a gap so they draw as brackets."""
    across: list[float] = []
    upward: list[float] = []
    for horizontal, vertical in zip(icoord, dcoord):
        across.extend([value / _LEAF_SCALE for value in horizontal])
        across.append(np.nan)
        upward.extend(list(vertical))
        upward.append(np.nan)

    return across, upward


def dendrogram_figure(tree: Tree, *, groups: int, leaves: int, title: str) -> go.Figure:
    """The last ``leaves`` merges of ``tree``, with the height leaving ``groups`` standing drawn across them.

    Showing the top of the tree is what makes it readable for a corpus of a few hundred takes: each drawn
    leaf is a whole branch, its width says how many recordings it gathered, and the line says where the
    count on screen cut through.
    """
    drawn = dendrogram(tree, no_plot=True, truncate_mode="lastp", p=leaves)
    across, upward = _link_lines(drawn["icoord"], drawn["dcoord"])
    height = cut_height(tree, groups)
    figure = go.Figure(data=[go.Scatter(x=across, y=upward, mode="lines", name="merges", line={"color": _LINK_COLOUR})])
    figure.add_hline(
        y=height,
        line={"color": _CUT_COLOUR, "dash": "dash"},
        annotation_text=f"{groups} groups at {height:.2f}",
    )
    figure.update_layout(
        title=title,
        height=_CURVE_HEIGHT,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        xaxis_title="branches, in the order the tree joined them",
        yaxis_title="height",
        showlegend=False,
    )
    return figure
