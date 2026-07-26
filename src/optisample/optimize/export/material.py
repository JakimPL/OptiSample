import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.config.render import PlaybackConfig
from optisample.dsp.timebase import row_seconds
from optisample.io.tracker.target import ExportTarget
from optisample.model import NoteEvent
from optisample.music import tracker_key
from optisample.optimize.velocity_map import VelocityVolumeMap
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.patterns.grid import Pattern
from trackmod.core.songs.order import OrderList

CHANNELS: Final = 1  # the material plays one voice at a time, so one channel carries the whole song.

_CHANNEL: Final = 0
_INSTRUMENT: Final = 0
_RELEASE_ROWS: Final = 1  # the row a note's release occupies, right after the rows it is held for.
_RELEASE_MARGIN: Final = 2  # rows a note leaves free so it and its release fit inside one pattern.


@dataclass(frozen=True)
class _Placement:
    """One event laid onto a pattern: where it starts, how many rows it holds, and how it sounds."""

    row: int
    rows: int
    note: Note
    volume: int

    @property
    def release_row(self) -> int:
        """The row the note is released on, one past the rows it is held for."""
        return self.row + self.rows

    @property
    def extent(self) -> int:
        """How many rows this placement occupies, its release included."""
        return self.rows + _RELEASE_ROWS


def event_rows(duration_s: float, seconds_per_row: float, max_rows: int) -> int:
    """A note's length in pattern rows: at least one row, capped so the note plus its release fit."""
    return min(max(1, math.ceil(duration_s / seconds_per_row)), max_rows - _RELEASE_MARGIN)


def _pattern_rows(placements: Sequence[_Placement], min_rows: int) -> int:
    """How tall a pattern holding ``placements`` must be, grown to the shortest height the format takes."""
    occupied = max(
        (placement.release_row + _RELEASE_ROWS for placement in placements),
        default=0,
    )
    return max(occupied, min_rows)


def _split_into_patterns(
    material: Sequence[NoteEvent],
    velocity_map: VelocityVolumeMap,
    *,
    seconds_per_row: float,
    max_rows: int,
) -> list[list[_Placement]]:
    """Place every event end to end, opening a fresh pattern when the next note would overflow one."""
    patterns: list[list[_Placement]] = []
    current: list[_Placement] = []
    cursor = 0
    for event in material:
        rows = event_rows(event.duration_s, seconds_per_row, max_rows)
        if current and cursor + rows + _RELEASE_ROWS > max_rows:
            patterns.append(current)
            current, cursor = [], 0

        placement = _Placement(
            row=cursor,
            rows=rows,
            note=tracker_key(event.pitch),
            volume=velocity_map.volume(event.velocity),
        )
        current.append(placement)
        cursor += placement.extent

    patterns.append(current)
    return patterns


def _build_pattern(
    placements: Sequence[_Placement],
    *,
    rows: int,
    target: ExportTarget,
) -> Pattern:
    """Write one pattern's note-on and release cells into a grid of ``rows`` rows."""
    builder = PatternBuilder(rows=rows, channels=CHANNELS)
    for placement in placements:
        builder.place(
            placement.row,
            _CHANNEL,
            Cell(note=placement.note, instrument=_INSTRUMENT, volume=placement.volume),
        )
        builder.place(placement.release_row, _CHANNEL, target.release_cell())

    return builder.build()


def material_patterns(
    material: Sequence[NoteEvent],
    velocity_map: VelocityVolumeMap,
    playback: PlaybackConfig,
    target: ExportTarget,
) -> tuple[tuple[Pattern, ...], OrderList]:
    """Lay the material events into patterns, applying the velocity->volume map, and order them.

    Each note writes a note-on cell (its key, the instrument, and its velocity mapped to a volume) at
    the current row and a release cell one row past its length, so it occupies its duration in rows
    plus one.
    """
    groups = _split_into_patterns(
        material,
        velocity_map,
        seconds_per_row=row_seconds(playback.speed, playback.tempo),
        max_rows=target.max_rows,
    )
    patterns = tuple(
        _build_pattern(
            group,
            rows=_pattern_rows(group, target.min_rows),
            target=target,
        )
        for group in groups
    )
    return patterns, OrderList.sequential(len(patterns))
