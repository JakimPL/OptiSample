"""Pattern records (cells, rows and the packed row stream) and the global playback settings.

A pattern is a grid of sparse cells; :func:`_pack_pattern` serializes it into IT's channel-marker byte
stream. Playback (speed/tempo and the volume ceilings) rides alongside because ``speed``/``tempo`` set
each row's duration and so belong with the row model.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from optisample.config.render import PlaybackConfig
from optisample.io.it_writer.constants import (
    _MASK_EFFECT,
    _MASK_INSTRUMENT,
    _MASK_NOTE,
    _MASK_VOLUME,
    CHANNEL_MARKER,
    END_OF_ROW,
    MAX_ROWS,
)


@dataclass(frozen=True)
class ITCell:
    """One pattern cell; ``None`` fields are left empty (not written) in the packed stream."""

    note: int | None = None
    instrument: int | None = None
    volume: int | None = None
    effect: tuple[int, int] | None = None


@dataclass(frozen=True)
class ITPattern:
    """A pattern: a row count and the sparse cells placed on it as ``(row, channel, cell)``."""

    rows: int
    cells: tuple[tuple[int, int, ITCell], ...] = ()


@dataclass(frozen=True)
class ITPlayback:
    """Global playback settings (bundled to keep :class:`~optisample.io.it_writer.module.ITModule` small).

    Built from :class:`~optisample.config.render.PlaybackConfig` at the export boundary via
    :func:`it_playback`; ``speed``/``tempo`` set the row duration, so they affect rendered timing.
    """

    speed: int
    tempo: int
    global_volume: int
    mix_volume: int


def it_playback(config: PlaybackConfig) -> ITPlayback:
    """Build the writer's :class:`ITPlayback` value-object from a :class:`PlaybackConfig`."""
    return ITPlayback(
        speed=config.speed,
        tempo=config.tempo,
        global_volume=config.global_volume,
        mix_volume=config.mix_volume,
    )


def _pack_cell(stream: bytearray, channel: int, cell: ITCell) -> None:
    """Append one channel marker + its present fields to a row's packed stream."""
    mask = 0
    if cell.note is not None:
        mask |= _MASK_NOTE
    if cell.instrument is not None:
        mask |= _MASK_INSTRUMENT
    if cell.volume is not None:
        mask |= _MASK_VOLUME
    if cell.effect is not None:
        mask |= _MASK_EFFECT
    if mask == 0:
        return
    stream.append(((channel + 1) | CHANNEL_MARKER) & 0xFF)
    stream.append(mask)
    if cell.note is not None:
        stream.append(cell.note & 0xFF)
    if cell.instrument is not None:
        stream.append(cell.instrument & 0xFF)
    if cell.volume is not None:
        stream.append(cell.volume & 0xFF)
    if cell.effect is not None:
        command, value = cell.effect
        stream.append(command & 0xFF)
        stream.append(value & 0xFF)


def _pack_pattern(pattern: ITPattern) -> bytes:
    """Serialize a pattern: 8-byte header (packed length, rows, reserved) + the packed row stream."""
    if not 1 <= pattern.rows <= MAX_ROWS:
        raise ValueError(f"pattern rows {pattern.rows} out of range 1..{MAX_ROWS}")
    by_row: dict[int, list[tuple[int, ITCell]]] = {}
    for row, channel, cell in pattern.cells:
        if not 0 <= row < pattern.rows:
            raise ValueError(f"cell row {row} out of range 0..{pattern.rows - 1}")
        by_row.setdefault(row, []).append((channel, cell))
    stream = bytearray()
    for row in range(pattern.rows):
        for channel, cell in sorted(by_row.get(row, []), key=lambda item: item[0]):
            _pack_cell(stream, channel, cell)
        stream.append(END_OF_ROW)
    return struct.pack("<HHI", len(stream), pattern.rows, 0) + bytes(stream)
