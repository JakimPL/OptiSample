from __future__ import annotations

from collections.abc import Callable

import pytest

from optisample.io.it_writer import ITInstrument, ITModule, identity_note_map, write_it_module
from optisample.io.it_writer.instruments import _instrument_header

Reader = Callable[[bytes, int], int]
Blob = Callable[..., ITModule]
Offsets = Callable[[bytes], tuple[list[int], list[int], list[int]]]


def test_instrument_note_map_is_identity_with_assigned_sample(one_sample_module: Blob, offset_tables: Offsets) -> None:
    blob = write_it_module(one_sample_module())
    (ins,), _, _ = offset_tables(blob)
    table = ins + 64  # keyboard table starts at 0x40 within the instrument header
    assert blob[table + 2 * 60] == 60 and blob[table + 2 * 60 + 1] == 1  # key 60 -> note 60, sample 1
    assert blob[table + 2 * 61] == 61 and blob[table + 2 * 61 + 1] == 0  # unused key -> no sample
    assert blob[ins + 24] == 128  # global volume
    assert blob[ins + 23] == 60  # pitch-pan centre C-5


def test_identity_note_map_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside the IT key range"):
        identity_note_map({120: 1})
    with pytest.raises(ValueError, match="non-negative"):
        identity_note_map({60: -1})


def test_instrument_header_serializer_requires_a_full_note_map() -> None:
    with pytest.raises(ValueError, match="note map must have"):
        _instrument_header(ITInstrument("i", note_map=((60, 1),)))  # far short of 120 entries
