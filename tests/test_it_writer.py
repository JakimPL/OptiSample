from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from optisample.io.it_writer import (
    NOTE_OFF,
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    _instrument_header,
    _sample_header,
    identity_note_map,
    write_it,
    write_it_module,
)
from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, SAMPLE_HEADER_BYTES


def _u16(blob: bytes, offset: int) -> int:
    return int(struct.unpack_from("<H", blob, offset)[0])


def _u32(blob: bytes, offset: int) -> int:
    return int(struct.unpack_from("<I", blob, offset)[0])


def ramp(frames: int = 256) -> np.ndarray:
    return np.linspace(-0.9, 0.9, frames, dtype=np.float64)


def one_sample_module(depth: int = 16, c5: int = 22_050, frames: int = 256, volume: int = 48) -> ITModule:
    sample = ITSample(name="ramp", pcm=ramp(frames), depth_bits=depth, c5speed=c5)
    instrument = ITInstrument(name="inst", note_map=identity_note_map({60: 1}))
    pattern = ITPattern(rows=8, cells=((0, 0, ITCell(note=60, instrument=1, volume=volume)),))
    return ITModule(name="probe", samples=(sample,), instruments=(instrument,), patterns=(pattern,), orders=(0,))


def offset_tables(blob: bytes) -> tuple[list[int], list[int], list[int]]:
    ins_num, smp_num, pat_num = _u16(blob, 34), _u16(blob, 36), _u16(blob, 38)
    base = FILE_HEADER_BYTES + _u16(blob, 32)  # after the file header + order list
    ins = [_u32(blob, base + 4 * i) for i in range(ins_num)]
    smp = [_u32(blob, base + 4 * ins_num + 4 * i) for i in range(smp_num)]
    pat = [_u32(blob, base + 4 * ins_num + 4 * smp_num + 4 * i) for i in range(pat_num)]
    return ins, smp, pat


def test_file_header_magic_flags_and_counts() -> None:
    blob = write_it_module(one_sample_module())
    assert blob[:4] == b"IMPM"
    assert (_u16(blob, 32), _u16(blob, 34), _u16(blob, 36), _u16(blob, 38)) == (
        2,
        1,
        1,
        1,
    )  # order(+terminator)/ins/smp/pat
    assert _u16(blob, 40) == 0x0214  # created-with -> selects the 554-byte instrument format
    assert _u16(blob, 44) == 0x0C  # use-instruments | linear-slides
    assert blob[FILE_HEADER_BYTES + 1] == 0xFF  # order list ends with the 0xFF terminator


def test_offset_tables_point_at_the_right_magic_and_chain_exactly() -> None:
    blob = write_it_module(one_sample_module())
    (ins,), (smp,), (pat,) = offset_tables(blob)
    assert blob[ins : ins + 4] == b"IMPI"
    assert blob[smp : smp + 4] == b"IMPS"
    assert ins + INSTRUMENT_HEADER_BYTES == smp  # instrument header is exactly 554 B
    assert smp + SAMPLE_HEADER_BYTES == pat  # sample header is exactly 80 B
    packed_len = _u16(blob, pat)
    data_pointer = _u32(blob, smp + 72)
    assert pat + 8 + packed_len == data_pointer  # pattern block precedes the PCM
    assert data_pointer + 256 * 2 == len(blob)  # 16-bit PCM runs to EOF


def test_sample_header_fields_16bit() -> None:
    blob = write_it_module(one_sample_module(depth=16, c5=22_050, frames=256))
    _, (smp,), _ = offset_tables(blob)
    assert blob[smp + 18] == 0x03  # Flg: data present | 16-bit
    assert blob[smp + 46] == 0x01  # Cvt: signed PCM
    assert _u32(blob, smp + 48) == 256  # Length is in frames
    assert _u32(blob, smp + 60) == 22_050  # C5Speed
    assert blob[smp + 20 : smp + 24] == b"ramp"


def test_sample_header_flag_is_8bit_when_depth_is_8() -> None:
    blob = write_it_module(one_sample_module(depth=8))
    _, (smp,), _ = offset_tables(blob)
    assert blob[smp + 18] == 0x01  # data present, 16-bit bit clear
    data_pointer = _u32(blob, smp + 72)
    assert data_pointer + 256 == len(blob)  # one byte per frame


@pytest.mark.parametrize("depth,scale,dtype", [(16, 32768.0, "<i2"), (8, 128.0, "<i1")])
def test_pcm_round_trips_in_file(depth: int, scale: float, dtype: str) -> None:
    signal = ramp(256)
    blob = write_it_module(one_sample_module(depth=depth, frames=256))
    _, (smp,), _ = offset_tables(blob)
    pointer = _u32(blob, smp + 72)
    stored = np.frombuffer(blob, dtype=dtype, count=256, offset=pointer)
    expected = np.clip(np.round(signal * scale), -scale, scale - 1).astype(dtype)
    assert np.array_equal(stored, expected)


def test_instrument_note_map_is_identity_with_assigned_sample() -> None:
    blob = write_it_module(one_sample_module())
    (ins,), _, _ = offset_tables(blob)
    table = ins + 64  # keyboard table starts at 0x40 within the instrument header
    assert blob[table + 2 * 60] == 60 and blob[table + 2 * 60 + 1] == 1  # key 60 -> note 60, sample 1
    assert blob[table + 2 * 61] == 61 and blob[table + 2 * 61 + 1] == 0  # unused key -> no sample
    assert blob[ins + 24] == 128  # global volume
    assert blob[ins + 23] == 60  # pitch-pan centre C-5


def unpack_pattern(blob: bytes, offset: int) -> dict[tuple[int, int], dict[str, object]]:
    packed_len, rows = _u16(blob, offset), _u16(blob, offset + 2)
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


def test_pattern_packs_note_instrument_volume_and_note_off() -> None:
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
    module = ITModule(name="m", samples=(sample,), instruments=(instrument,), patterns=(pattern,), orders=(0,))
    blob = write_it_module(module)
    _, _, (pat,) = offset_tables(blob)
    cells = unpack_pattern(blob, pat)
    assert cells[(0, 0)] == {"note": 60, "instrument": 1, "volume": 40}
    assert cells[(4, 0)] == {"note": NOTE_OFF}
    assert cells[(8, 0)] == {"effect": (19, 0x80)}
    assert set(cells) == {(0, 0), (4, 0), (8, 0)}  # empty rows carry nothing


def test_multiple_samples_orders_and_playback() -> None:
    samples = (ITSample("a", ramp(32), 16), ITSample("b", ramp(48), 8))
    instrument = ITInstrument("i", identity_note_map({60: 1, 62: 2}))
    patterns = (ITPattern(rows=4), ITPattern(rows=8, cells=((0, 0, ITCell(note=62, instrument=1, volume=64)),)))
    module = ITModule(
        name="multi",
        samples=samples,
        instruments=(instrument,),
        patterns=patterns,
        orders=(0, 1, 0),
        playback=ITPlayback(speed=3, tempo=140, global_volume=100, mix_volume=64),
    )
    blob = write_it_module(module)
    assert (_u16(blob, 32), _u16(blob, 34), _u16(blob, 36), _u16(blob, 38)) == (4, 1, 2, 2)  # 3 orders + terminator
    assert blob[FILE_HEADER_BYTES : FILE_HEADER_BYTES + 4] == bytes((0, 1, 0, 0xFF))
    assert blob[50] == 3 and blob[51] == 140  # speed / tempo
    _, (smp_a, smp_b), _ = offset_tables(blob)
    assert blob[smp_a + 18] == 0x03 and blob[smp_b + 18] == 0x01  # 16-bit then 8-bit


def test_identity_note_map_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of IT range"):
        identity_note_map({120: 1})
    with pytest.raises(ValueError, match="non-negative"):
        identity_note_map({60: -1})


def test_unsupported_depth_raises() -> None:
    with pytest.raises(ValueError, match="unsupported depth"):
        write_it_module(one_sample_module(depth=24))


def test_pattern_row_bounds_are_validated() -> None:
    good_sample = (ITSample("s", ramp(16), 16),)
    instrument = (ITInstrument("i", identity_note_map({60: 1})),)
    with pytest.raises(ValueError, match="rows .* out of range"):
        write_it_module(ITModule("m", good_sample, instrument, (ITPattern(rows=0),), (0,)))
    with pytest.raises(ValueError, match="cell row .* out of range"):
        bad = ITPattern(rows=4, cells=((9, 0, ITCell(note=60)),))
        write_it_module(ITModule("m", good_sample, instrument, (bad,), (0,)))


def test_private_serializers_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported depth"):
        _sample_header(ITSample("s", ramp(8), depth_bits=24), data_offset=0)
    with pytest.raises(ValueError, match="note map must have"):
        _instrument_header(ITInstrument("i", note_map=((60, 1),)))  # far short of 120 entries


def test_empty_cells_emit_nothing() -> None:
    sample = (ITSample("s", ramp(16), 16),)
    instrument = (ITInstrument("i", identity_note_map({60: 1})),)
    pattern = ITPattern(rows=4, cells=((0, 0, ITCell()), (1, 0, ITCell(note=60, instrument=1))))
    blob = write_it_module(ITModule("m", sample, instrument, (pattern,), (0,)))
    _, _, (pat,) = offset_tables(blob)
    assert set(unpack_pattern(blob, pat)) == {(1, 0)}  # the all-None cell is skipped


@pytest.mark.parametrize("depth", [16, 8])
def test_xmodits_round_trip_extracts_matching_pcm(tmp_path: Path, depth: int) -> None:
    xmodits = pytest.importorskip("xmodits")
    signal = ramp(300)
    sample = ITSample(name="rt", pcm=signal, depth_bits=depth, c5speed=16_000)
    module = ITModule(
        name="rt",
        samples=(sample,),
        instruments=(ITInstrument("i", identity_note_map({60: 1})),),
        patterns=(ITPattern(rows=4, cells=((0, 0, ITCell(note=60, instrument=1)),)),),
        orders=(0,),
    )
    it_path = tmp_path / "rt.it"
    write_it(it_path, module)
    dest = tmp_path / "out"
    dest.mkdir()
    xmodits.dump(str(it_path), str(dest), format="wav")
    extracted = sorted(dest.glob("*.wav"))
    assert len(extracted) == 1
    data, sample_rate = sf.read(str(extracted[0]), dtype="float64", always_2d=False)
    ripped = np.asarray(data, dtype=np.float64).ravel()
    assert sample_rate == 16_000
    assert ripped.size == 300
    # An independent ripper decodes the same waveform we stored (tolerance covers 8/16-bit quantization).
    assert np.max(np.abs(ripped - signal)) < (0.02 if depth == 16 else 0.05)
