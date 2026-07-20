from __future__ import annotations

import struct

import pytest

from optisample.io.it_format import (
    CHANNELS_STORED,
    FILE_HEADER,
    INSTRUMENT_HEADER,
    KEYBOARD_NOTES,
    MAX_IT_NOTE,
    SAMPLE_HEADER,
    ArrayField,
    Field,
    ITRecord,
)
from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, SAMPLE_HEADER_BYTES


def test_max_it_note_is_the_last_keyboard_key() -> None:
    assert KEYBOARD_NOTES == 120
    assert MAX_IT_NOTE == 119


def test_pack_places_each_field_at_its_offset() -> None:
    record = ITRecord(size=8, fields=(Field("magic", 0, "4s"), Field("value", 4, "<I")))
    blob = record.pack({"magic": b"ABCD", "value": 0x01020304})
    assert blob == b"ABCD" + struct.pack("<I", 0x01020304)


def test_pack_zero_fills_unwritten_offsets() -> None:
    record = ITRecord(size=6, fields=(Field("byte", 2, "B"),))
    blob = record.pack({"byte": 0x7F})
    assert blob == bytes([0, 0, 0x7F, 0, 0, 0])  # only offset 2 is written; the rest stay zero


def test_string_field_is_null_padded_by_struct() -> None:
    record = ITRecord(size=4, fields=(Field("name", 0, "4s"),))
    assert record.pack({"name": b"hi"}) == b"hi\x00\x00"


def test_array_field_walks_the_stride() -> None:
    record = ITRecord(size=8, fields=(), arrays=(ArrayField("pairs", 0, 3, "BB"),))
    blob = record.pack({"pairs": ((1, 2), (3, 4), (5, 6))})
    assert blob == bytes([1, 2, 3, 4, 5, 6, 0, 0])  # three 2-byte pairs, then the reserved tail


def test_array_field_starts_at_its_offset() -> None:
    record = ITRecord(size=6, fields=(), arrays=(ArrayField("pairs", 2, 2, "BB"),))
    blob = record.pack({"pairs": ((9, 9), (8, 8))})
    assert blob == bytes([0, 0, 9, 9, 8, 8])


def test_record_sizes_match_the_byte_model() -> None:
    assert FILE_HEADER.size == FILE_HEADER_BYTES
    assert SAMPLE_HEADER.size == SAMPLE_HEADER_BYTES
    assert INSTRUMENT_HEADER.size == INSTRUMENT_HEADER_BYTES


@pytest.mark.parametrize("record", [FILE_HEADER, SAMPLE_HEADER, INSTRUMENT_HEADER])
def test_every_field_fits_inside_the_record(record: ITRecord) -> None:
    for field in record.fields:
        assert field.offset + struct.calcsize(field.code) <= record.size
    for array in record.arrays:
        assert array.offset + array.count * struct.calcsize(array.code) <= record.size


def test_instrument_note_map_covers_the_full_keyboard() -> None:
    (note_map,) = INSTRUMENT_HEADER.arrays
    assert note_map.count == KEYBOARD_NOTES


def test_file_header_channel_tables_span_all_channels() -> None:
    pans = {field.name: field for field in FILE_HEADER.fields}
    assert struct.calcsize(pans["channel_pan"].code) == CHANNELS_STORED
    assert struct.calcsize(pans["channel_volume"].code) == CHANNELS_STORED
