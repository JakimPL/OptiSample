from __future__ import annotations

import struct
from collections.abc import Callable

import numpy as np
import pytest

from optisample.io.it_writer import (
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    identity_note_map,
    write_it_module,
)

NOTE_OFF = 255  # IT pattern note value that releases the playing note; verified here as an arbitrary note byte.

Offsets = Callable[[bytes], tuple[list[int], list[int], list[int]]]


def unpack_pattern(blob: bytes, offset: int) -> dict[tuple[int, int], dict[str, object]]:
    packed_len = int(struct.unpack_from("<H", blob, offset)[0])
    rows = int(struct.unpack_from("<H", blob, offset + 2)[0])
    data = blob[offset + 8 : offset + 8 + packed_len]
    cells: dict[tuple[int, int], dict[str, object]] = {}
    index, row = 0, 0
    while row < rows:
        marker = data[index]
        index += 1
        if marker == 0:
            row += 1
            continue
        channel = (marker - 1) & 0x3F
        mask = data[index]
        index += 1
        cell: dict[str, object] = {}
        if mask & 0x01:
            cell["note"] = data[index]
            index += 1
        if mask & 0x02:
            cell["instrument"] = data[index]
            index += 1
        if mask & 0x04:
            cell["volume"] = data[index]
            index += 1
        if mask & 0x08:
            cell["effect"] = (data[index], data[index + 1])
            index += 2
        cells[(row, channel)] = cell
    return cells


def test_pattern_packs_note_instrument_volume_and_note_off(
    offset_tables: Offsets, ramp: Callable[..., np.ndarray], playback: ITPlayback
) -> None:
    sample = ITSample(name="s", pcm=ramp(64), depth_bits=16)
    instrument = ITInstrument(name="i", note_map=identity_note_map({60: 1}))
    pattern = ITPattern(
        rows=16,
        cells=(
            (0, 0, ITCell(note=60, instrument=1, volume=40)),
            (4, 0, ITCell(note=NOTE_OFF)),
            (8, 0, ITCell(effect=(19, 0x80))),  # Sxx-style command+value
        ),
    )
    module = ITModule(
        name="m", samples=(sample,), instruments=(instrument,), patterns=(pattern,), orders=(0,), playback=playback
    )
    blob = write_it_module(module)
    _, _, (pat,) = offset_tables(blob)
    cells = unpack_pattern(blob, pat)
    assert cells[(0, 0)] == {"note": 60, "instrument": 1, "volume": 40}
    assert cells[(4, 0)] == {"note": NOTE_OFF}
    assert cells[(8, 0)] == {"effect": (19, 0x80)}
    assert set(cells) == {(0, 0), (4, 0), (8, 0)}  # empty rows carry nothing


def test_empty_cells_emit_nothing(
    offset_tables: Offsets, ramp: Callable[..., np.ndarray], playback: ITPlayback
) -> None:
    sample = (ITSample("s", ramp(16), 16),)
    instrument = (ITInstrument("i", identity_note_map({60: 1})),)
    pattern = ITPattern(rows=4, cells=((0, 0, ITCell()), (1, 0, ITCell(note=60, instrument=1))))
    blob = write_it_module(ITModule("m", sample, instrument, (pattern,), (0,), playback))
    _, _, (pat,) = offset_tables(blob)
    assert set(unpack_pattern(blob, pat)) == {(1, 0)}  # the all-None cell is skipped


def test_pattern_row_bounds_are_validated(ramp: Callable[..., np.ndarray], playback: ITPlayback) -> None:
    good_sample = (ITSample("s", ramp(16), 16),)
    instrument = (ITInstrument("i", identity_note_map({60: 1})),)
    with pytest.raises(ValueError, match="rows .* out of range"):
        write_it_module(ITModule("m", good_sample, instrument, (ITPattern(rows=0),), (0,), playback))
    with pytest.raises(ValueError, match="cell row .* out of range"):
        bad = ITPattern(rows=4, cells=((9, 0, ITCell(note=60)),))
        write_it_module(ITModule("m", good_sample, instrument, (bad,), (0,), playback))
