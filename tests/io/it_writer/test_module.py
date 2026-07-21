from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from optisample.io.it_writer import (
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    identity_note_map,
    write_it,
    write_it_module,
)
from optisample.metrics.size import FILE_HEADER_BYTES, INSTRUMENT_HEADER_BYTES, SAMPLE_HEADER_BYTES

Reader = Callable[[bytes, int], int]
Blob = Callable[..., ITModule]
Offsets = Callable[[bytes], tuple[list[int], list[int], list[int]]]


def test_file_header_magic_flags_and_counts(one_sample_module: Blob, u16: Reader) -> None:
    blob = write_it_module(one_sample_module())
    assert blob[:4] == b"IMPM"
    assert (u16(blob, 32), u16(blob, 34), u16(blob, 36), u16(blob, 38)) == (
        2,
        1,
        1,
        1,
    )  # order(+terminator)/ins/smp/pat
    assert u16(blob, 40) == 0x0214  # created-with -> selects the 554-byte instrument format
    assert u16(blob, 44) == 0x0C  # use-instruments | linear-slides
    assert blob[FILE_HEADER_BYTES + 1] == 0xFF  # order list ends with the 0xFF terminator


def test_offset_tables_point_at_the_right_magic_and_chain_exactly(
    one_sample_module: Blob, offset_tables: Offsets, u16: Reader, u32: Reader
) -> None:
    blob = write_it_module(one_sample_module())
    (ins,), (smp,), (pat,) = offset_tables(blob)
    assert blob[ins : ins + 4] == b"IMPI"
    assert blob[smp : smp + 4] == b"IMPS"
    assert ins + INSTRUMENT_HEADER_BYTES == smp  # instrument header is exactly 554 B
    assert smp + SAMPLE_HEADER_BYTES == pat  # sample header is exactly 80 B
    packed_len = u16(blob, pat)
    data_pointer = u32(blob, smp + 72)
    assert pat + 8 + packed_len == data_pointer  # pattern block precedes the PCM
    assert data_pointer + 256 * 2 == len(blob)  # 16-bit PCM runs to EOF


def test_multiple_samples_orders_and_playback(
    offset_tables: Offsets, u16: Reader, ramp: Callable[..., np.ndarray]
) -> None:
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
    assert (u16(blob, 32), u16(blob, 34), u16(blob, 36), u16(blob, 38)) == (4, 1, 2, 2)  # 3 orders + terminator
    assert blob[FILE_HEADER_BYTES : FILE_HEADER_BYTES + 4] == bytes((0, 1, 0, 0xFF))
    assert blob[50] == 3 and blob[51] == 140  # speed / tempo
    _, (smp_a, smp_b), _ = offset_tables(blob)
    assert blob[smp_a + 18] == 0x03 and blob[smp_b + 18] == 0x01  # 16-bit then 8-bit


@pytest.mark.parametrize("depth", [16, 8])
def test_xmodits_round_trip_extracts_matching_pcm(
    tmp_path: Path, depth: int, ramp: Callable[..., np.ndarray], playback: ITPlayback
) -> None:
    xmodits = pytest.importorskip("xmodits")
    signal = ramp(300)
    sample = ITSample(name="rt", pcm=signal, depth_bits=depth, c5speed=16_000)
    module = ITModule(
        name="rt",
        samples=(sample,),
        instruments=(ITInstrument("i", identity_note_map({60: 1})),),
        patterns=(ITPattern(rows=4, cells=((0, 0, ITCell(note=60, instrument=1)),)),),
        orders=(0,),
        playback=playback,
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
