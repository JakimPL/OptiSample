from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from scipy.cluster.hierarchy import dendrogram

from notebooks.utils.layouts import Placement
from notebooks.utils.views import Row
from optisample.cluster.partition import Tree

SelectionPoint = Mapping[str, Any]  # one clicked point as plotly serialized it and marimo handed it back

_PALETTE: Final = tuple(qualitative.Dark24)
_CONTINUOUS: Final = "Viridis"  # the scale a reading with an order of its own is drawn along
_CURVE: Final = "curveNumber"  # what plotly calls the trace a clicked point was drawn in
_POINT_INDEX: Final = "pointIndex"  # what it calls the place of that point inside its own trace
_POINT_NUMBER: Final = "pointNumber"  # the same place under the name a click payload states it by
_CLICKABLE_MODE: Final = "lines+markers"  # what a flat trace is drawn as, which is what marimo forwards clicks from
_SOLID_MODE: Final = "markers"  # what a trace in a box is drawn as, a scene reading points alone
_HAIRLINE: Final = 0  # the width the run joining a flat trace's markers is stroked at, leaving the points alone
_MARKER_SIZE: Final = 7  # how big a recording is drawn on a plane
_SOLID_MARKER_SIZE: Final = 4  # how big it is drawn in a box, where depth already thins the field
_REPRESENTATIVE_SIZE: Final = 19  # how big the ring around the take standing for a group is drawn on a plane
_REPRESENTATIVE_SOLID_SIZE: Final = 12  # how big that ring is drawn in a box, over the smaller points there
_REPRESENTATIVE_SYMBOL: Final = "diamond-open"  # the one outline both the plane and the box draw it with
_RING_WIDTH: Final = 3.0  # how thick that outline is stroked, well above the field it stands over
_ON_SCREEN_SIZE: Final = 13  # how big the count a sweep curve is being read at is ringed
_GROUP_COLUMN: Final = "group"  # the column naming the group a take fell into, which rings it in that colour
_HEIGHT: Final = 620  # pixels a scatter is given, ample for a group to be picked out by eye
_CURVE_HEIGHT: Final = 320  # pixels the two reading curves are given
_FLOAT_FORMAT: Final = ":.3f"  # how a measured reading is spelled in a hover
_PLAIN: Final = ""  # how a name or a count is spelled there
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
    lines = [f"{key}: %{{customdata[{index}]{formats[index]}}}" for index, key in enumerate(keys)]
    return "<br>".join(lines) + "<extra></extra>"


def _carried(rows: Sequence[Row], keys: Sequence[str]) -> list[list[object]]:
    """What each point carries beside its position: every reading its recording was read for, in one list."""
    return [[row[key] for key in keys] for row in rows]


@dataclass(frozen=True)
class _Reading:
    """The rows of one picture as every trace of it draws them, read once for the whole figure.

    ``rows`` is what each recording was read for, ``carried`` the same readings as the values a point is
    drawn with -- its place in the space first -- and ``hover`` the tooltip listing them. Every trace of a
    field draws from these, so a point picked out of any of them says what a point of any other says.
    """

    rows: Sequence[Row]
    carried: Sequence[Sequence[object]]
    hover: str


def _reading(rows: Sequence[Row]) -> _Reading:
    """One picture's rows read for what its points carry and the tooltip they are hovered by."""
    keys = list(rows[0]) if rows else []
    return _Reading(rows=rows, carried=_carried(rows, keys), hover=_hover(rows, keys) if rows else _PLAIN)


@dataclass(frozen=True)
class _Drawn:
    """One trace of a field beside the recordings it drew, in the order it drew them.

    A click comes back naming the trace it landed on and where inside that trace the point sits, so the
    places a trace was built from are what turn either number into a recording of the corpus.
    """

    trace: go.Scatter | go.Scatter3d
    places: tuple[int, ...]


def _positions(placement: Placement, places: Sequence[int]) -> dict[str, Sequence[float]]:
    """Where the named recordings sit, as the axis arguments a plotly trace is built from."""
    axes = ("x", "y", "z")
    return {axes[axis]: [placement.axes[axis][place] for place in places] for axis in range(placement.components)}


def _trace(placement: Placement, *, places: Sequence[int], **options: object) -> _Drawn:
    """One set of recordings drawn where the picture puts them, on a plane or in a box as it has axes for.

    A flat trace is drawn as a run of markers joined by a hairline, which is the shape marimo forwards a
    click back from; the run is stroked at no width, so the field on screen is the markers alone.
    """
    positions = _positions(placement, places)
    drawn: go.Scatter | go.Scatter3d
    if placement.is_plane:
        drawn = go.Scatter(mode=_CLICKABLE_MODE, line={"width": _HAIRLINE}, **positions, **options)
    else:
        drawn = go.Scatter3d(mode=_SOLID_MODE, **positions, **options)

    return _Drawn(trace=drawn, places=tuple(places))


def _marker_size(placement: Placement) -> int:
    """How big a recording is drawn, a box thinning the field by depth where a plane holds it all at once."""
    return _MARKER_SIZE if placement.is_plane else _SOLID_MARKER_SIZE


@dataclass(frozen=True)
class Colouring:
    """Which reading a field is drawn by, and the colour each group of the cut holds.

    ``column`` names the reading every point is coloured by: a column of names draws one colour and one
    legend entry apiece, a column of measurements shades along a scale. ``groups`` is the colour each group
    goes by, read off the key the take standing for it plays -- it strokes the rings over the field
    whatever the field is coloured by, and colours the field itself where a reader is colouring by group.
    """

    column: str
    groups: Mapping[str, str]


def _colours(names: Sequence[str]) -> dict[str, str]:
    """The colour each name is drawn in, which is the one spelling every trace of a picture reads it by."""
    return {name: _PALETTE[position % len(_PALETTE)] for position, name in enumerate(names)}


def _set_colours(rows: Sequence[Row], colouring: Colouring) -> dict[str, str]:
    """The colour each set of the colouring column is drawn in, the sets in the order a legend lists them.

    Colouring by group hands each set the colour its own key turned, so a field, its legend and the rings
    over it all carry the keyboard; any other column of names is handed the palette, which tells its sets
    apart without claiming to say anything about the keys.
    """
    names = sorted({str(row[colouring.column]) for row in rows})
    if colouring.column == _GROUP_COLUMN:
        return {name: colouring.groups[name] for name in names}

    return _colours(names)


def _named_traces(placement: Placement, read: _Reading, colouring: Colouring) -> list[_Drawn]:
    """One trace per value the colouring column names, so each reads as its own colour and legend entry."""
    rows = read.rows
    column = colouring.column
    return [
        _trace(
            placement,
            places=[index for index, row in enumerate(rows) if str(row[column]) == name],
            name=name,
            marker={"size": _marker_size(placement), "color": colour},
            customdata=[read.carried[index] for index, row in enumerate(rows) if str(row[column]) == name],
            hovertemplate=read.hover,
        )
        for name, colour in _set_colours(rows, colouring).items()
    ]


def _scaled_trace(placement: Placement, read: _Reading, colouring: Colouring) -> _Drawn:
    """Every recording in one trace, shaded along the reading the colouring column holds."""
    column = colouring.column
    return _trace(
        placement,
        places=list(range(len(read.rows))),
        name=column,
        marker={
            "size": _marker_size(placement),
            "color": [float(row[column]) for row in read.rows],
            "colorscale": _CONTINUOUS,
            "showscale": True,
            "colorbar": {"title": column},
        },
        customdata=list(read.carried),
        hovertemplate=read.hover,
    )


def _representative_trace(
    placement: Placement,
    read: _Reading,
    colouring: Colouring,
    representatives: Sequence[int],
) -> _Drawn:
    """The takes standing for their groups, ringed over the field so a selection reads at a glance.

    The ring stands well clear of the point it surrounds and is stroked in the colour of the group that
    take was chosen for, so a representative is picked out of the field by eye and read as a member of its
    own group at once -- whichever reading the field itself is coloured by.
    """
    return _trace(
        placement,
        places=representatives,
        name="stands for its group",
        marker={
            "size": _REPRESENTATIVE_SIZE if placement.is_plane else _REPRESENTATIVE_SOLID_SIZE,
            "symbol": _REPRESENTATIVE_SYMBOL,
            "color": [colouring.groups[str(read.rows[place][_GROUP_COLUMN])] for place in representatives],
            "line": {"width": _RING_WIDTH},
        },
        customdata=[read.carried[place] for place in representatives],
        hovertemplate=read.hover,
    )


@dataclass(frozen=True)
class Picture:
    """A drawn field beside the recordings each of its traces drew, which is what a click is read through.

    ``figure`` is what marimo shows; ``places`` says, trace by trace and point by point, which recording of
    the corpus stands there. A click hands back the trace it landed on and the place of the point inside
    it, so the two together name a recording however the field was split into traces.
    """

    figure: go.Figure
    places: tuple[tuple[int, ...], ...]


def field(
    placement: Placement,
    rows: Sequence[Row],
    *,
    colouring: Colouring,
    representatives: Sequence[int],
    title: str,
) -> Picture:
    """Every recording where ``placement`` puts it, shaded by one of its readings and hovering all of them.

    A column holding names -- the group a take fell into, the note it plays -- draws one colour and one
    legend entry per name, so a click on the legend isolates that set; a column holding measurements is
    shaded along a scale beside the picture. The takes standing for their groups are ringed over the top,
    and the picture keeps which recording each trace drew, which is what turns a click into a recording.
    """
    read = _reading(rows)
    named = isinstance(rows[0][colouring.column], str) if rows else False
    drawn = [
        *(_named_traces(placement, read, colouring) if named else [_scaled_trace(placement, read, colouring)]),
        _representative_trace(placement, read, colouring, representatives),
    ]
    figure = go.Figure(data=[one.trace for one in drawn])
    figure.update_layout(
        title=title,
        height=_HEIGHT,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"itemsizing": "constant"},
        clickmode="event+select",
    )
    _label_axes(figure, placement)
    return Picture(figure=figure, places=tuple(one.places for one in drawn))


def _label_axes(figure: go.Figure, placement: Placement) -> None:
    """Name every drawn axis on the figure, on the plane or on the box as the picture has axes for."""
    titles = placement.titles
    if placement.is_plane:
        figure.update_layout(xaxis_title=titles[0], yaxis_title=titles[1])
        return

    figure.update_layout(scene={"xaxis_title": titles[0], "yaxis_title": titles[1], "zaxis_title": titles[2]})


def _landed(point: SelectionPoint) -> tuple[int, int] | None:
    """Which trace a clicked point was drawn in and where inside it, as the payload states the pair.

    A click arriving from the browser is read for the two numbers plotly always sends with a point; a
    payload stating either of them some other way names nothing, which leaves the panels as they stood.
    """
    curve = point.get(_CURVE)
    inside = point.get(_POINT_INDEX, point.get(_POINT_NUMBER))
    if isinstance(curve, int) and isinstance(inside, int):
        return curve, inside

    return None


def _place(picture: Picture, point: SelectionPoint) -> int | None:
    """Which recording of the corpus one clicked point stands for, where ``picture`` drew one there."""
    landed = _landed(point)
    if landed is None:
        return None

    curve, inside = landed
    if curve in range(len(picture.places)) and inside in range(len(picture.places[curve])):
        return picture.places[curve][inside]

    return None


def selected(picture: Picture, points: Sequence[SelectionPoint]) -> tuple[int, ...]:
    """Which recordings a click or a drag over ``picture`` picked out, as places in the corpus.

    Every point comes back naming the trace it was drawn in and where it sits inside that trace, and the
    picture holds which recording each trace drew there, so a selection reads as indices into the very
    corpus the picture was built from whichever trace the click landed on.
    """
    found = (_place(picture, point) for point in points)
    return tuple(place for place in found if place is not None)


def picked(picture: Picture, points: Sequence[SelectionPoint], standing: int) -> int:
    """Which recording a click picked out, holding ``standing`` where the click named none.

    A picture redrawn hands back nothing at all, so holding what stood leaves the panels reading the take a
    reader picked last.
    """
    found = selected(picture, points)
    return found[0] if found else standing


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
                marker={"size": _ON_SCREEN_SIZE, "symbol": "circle-open", "color": _CUT_COLOUR},
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
