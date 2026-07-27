from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate, combinations
from typing import Final

from optisample.model import NoteEvent
from optisample.music import MIDI_MAX_VELOCITY

_SOFTEST_VELOCITY: Final = 0  # the quietest cell reaches down to silence, so every dynamic resolves


@dataclass(frozen=True, order=True)
class VelocityBand:
    """A run of velocities one stored recording answers for, ``lowest`` and ``highest`` both inside it."""

    lowest: int
    highest: int

    def covers(self, velocity: int) -> bool:
        """Whether a note struck at ``velocity`` is served by this band."""
        return self.lowest <= velocity <= self.highest

    @property
    def label(self) -> str:
        """Stable display name: the velocity span the band answers for (``v000-v063``)."""
        return f"v{self.lowest:03d}-v{self.highest:03d}"


@dataclass(frozen=True)
class VelocityCells:
    """The elementary velocity ranges every candidate band is assembled from.

    Cells ascend and tile the whole velocity axis, each holding roughly the same share of the material's
    playing time, so a band built from a run of them stands for a share of what is played rather than a
    share of the axis. How many there are is the cost dial: the bands, and so the splits the allocation
    prices and the reconstructions each one asks for, are counted from it.
    """

    cells: tuple[VelocityBand, ...]

    @property
    def count(self) -> int:
        """How many cells the material cut into, which bounds how many layers can be told apart."""
        return len(self.cells)

    def band(self, start: int, stop: int) -> VelocityBand:
        """The band spanning cells ``[start, stop)``: the first cell's floor up to the last cell's ceiling."""
        return VelocityBand(self.cells[start].lowest, self.cells[stop - 1].highest)


@dataclass(frozen=True)
class VelocityLayers:
    """One split of the velocity axis into the bands an instrument stores a recording for.

    The bands ascend and tile the whole axis, so every velocity resolves to exactly one of them, and the
    index it resolves to names the instrument the written pattern plays that note through.
    """

    bands: tuple[VelocityBand, ...]

    @property
    def count(self) -> int:
        """How many layers this split stores, which is how many instruments the module carries."""
        return len(self.bands)

    def band_index(self, velocity: int) -> int:
        """Which layer serves a note struck at ``velocity``.

        Raises:
            ValueError: when ``velocity`` lies above the loudest the topmost band answers for.
        """
        for index, band in enumerate(self.bands):
            if velocity <= band.highest:
                return index

        raise ValueError(f"velocity {velocity} lies above the topmost layer {self.bands[-1].label}")


UNSPLIT: Final = VelocityLayers((VelocityBand(_SOFTEST_VELOCITY, MIDI_MAX_VELOCITY),))  # one layer, every dynamic


def _weight_by_velocity(material: Sequence[NoteEvent]) -> dict[int, float]:
    """Playing time the material spends at each velocity it strikes."""
    weights: dict[int, float] = {}
    for event in material:
        weights[event.velocity] = weights.get(event.velocity, 0.0) + event.weight

    return weights


def _cut_positions(cumulative: Sequence[float], total: float, nodes: int) -> list[int]:
    """Indices into the played velocities that each cell below the top ends at.

    Cutting on cumulative playing time puts an equal share of the material in every cell, so the axis is
    resolved finely where the instrument spends its time. A quantile landing on a velocity already cut at,
    or on the loudest one played, keeps the cut already in place, which is what lets a material spending
    its time at a handful of dynamics answer with fewer, wider cells than were asked for.
    """
    positions: list[int] = []
    for step in range(1, nodes):
        index = bisect_left(cumulative, total * step / nodes)
        if index < len(cumulative) - 1 and (not positions or index > positions[-1]):
            positions.append(index)

    return positions


def velocity_cells(material: Sequence[NoteEvent], nodes: int) -> VelocityCells:
    """Cut the velocity axis into at most ``nodes`` cells holding equal shares of the material's time.

    Every cut lands on a velocity the material plays, so each boundary sits between two dynamics that were
    actually struck and each cell holds at least one of them. The quietest cell reaches down to silence
    and the loudest up to the top of the MIDI range, so a dynamic the material leaves unplayed still
    resolves to a layer.
    """
    weights = _weight_by_velocity(material)
    played = sorted(weights)
    cumulative = list(accumulate(weights[velocity] for velocity in played))
    edges = [played[index] for index in _cut_positions(cumulative, sum(weights.values()), nodes)]
    lowest = (_SOFTEST_VELOCITY, *(edge + 1 for edge in edges))
    highest = (*edges, MIDI_MAX_VELOCITY)
    return VelocityCells(tuple(VelocityBand(bottom, top) for bottom, top in zip(lowest, highest)))


def partitions(cells: VelocityCells, max_layers: int) -> Iterator[VelocityLayers]:
    """Every split of ``cells`` into at most ``max_layers`` bands, fewest layers first.

    A layer is a run of neighbouring cells, so a split is a choice of which cell boundaries to keep as
    layer boundaries: over ``V`` cells there are ``C(V-1, L-1)`` splits into ``L`` layers. Ordering them by
    layer count states the single-layer split first, which is the plan every richer one has to beat by the
    configured margin.
    """
    for count in range(1, min(max_layers, cells.count) + 1):
        for cuts in combinations(range(1, cells.count), count - 1):
            edges = (0, *cuts, cells.count)
            yield VelocityLayers(tuple(cells.band(start, stop) for start, stop in zip(edges, edges[1:])))
