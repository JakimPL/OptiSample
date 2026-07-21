from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.io.it_writer import ITInstrument, ITModule, ITSample, write_it_module
from optisample.io.it_writer.samples import _sample_header

Reader = Callable[[bytes, int], int]
Blob = Callable[..., ITModule]
Offsets = Callable[[bytes], tuple[list[int], list[int], list[int]]]


def test_sample_header_fields_16bit(one_sample_module: Blob, offset_tables: Offsets, u32: Reader) -> None:
    blob = write_it_module(one_sample_module(depth=16, c5=22_050, frames=256))
    _, (smp,), _ = offset_tables(blob)
    assert blob[smp + 18] == 0x03  # Flg: data present | 16-bit
    assert blob[smp + 46] == 0x01  # Cvt: signed PCM
    assert u32(blob, smp + 48) == 256  # Length is in frames
    assert u32(blob, smp + 60) == 22_050  # C5Speed
    assert blob[smp + 20 : smp + 24] == b"ramp"


def test_sample_header_flag_is_8bit_when_depth_is_8(
    one_sample_module: Blob, offset_tables: Offsets, u32: Reader
) -> None:
    blob = write_it_module(one_sample_module(depth=8))
    _, (smp,), _ = offset_tables(blob)
    assert blob[smp + 18] == 0x01  # data present, 16-bit bit clear
    data_pointer = u32(blob, smp + 72)
    assert data_pointer + 256 == len(blob)  # one byte per frame


def test_sample_header_writes_loop_flag_and_points(
    one_sample_module: Blob, offset_tables: Offsets, u32: Reader
) -> None:
    module = one_sample_module(frames=256)
    looped = ITSample(name="ramp", pcm=module.samples[0].pcm, depth_bits=16, c5speed=22_050, loop=(40, 200))
    blob = write_it_module(
        ITModule(
            name="probe",
            samples=(looped,),
            instruments=module.instruments,
            patterns=module.patterns,
            orders=(0,),
            playback=module.playback,
        )
    )
    _, (smp,), _ = offset_tables(blob)
    assert blob[smp + 18] == 0x03 | 0x10  # data | 16-bit | use-loop
    assert u32(blob, smp + 52) == 40  # Loop Begin
    assert u32(blob, smp + 56) == 200  # Loop End


def test_sample_header_has_no_loop_flag_by_default(
    one_sample_module: Blob, offset_tables: Offsets, u32: Reader
) -> None:
    blob = write_it_module(one_sample_module())
    _, (smp,), _ = offset_tables(blob)
    assert blob[smp + 18] & 0x10 == 0  # use-loop bit clear
    assert u32(blob, smp + 52) == 0 and u32(blob, smp + 56) == 0


@pytest.mark.parametrize("depth,scale,dtype", [(16, 32768.0, "<i2"), (8, 128.0, "<i1")])
def test_pcm_round_trips_in_file(
    depth: int, scale: float, dtype: str, one_sample_module: Blob, offset_tables: Offsets, u32: Reader
) -> None:
    signal = np.linspace(-0.9, 0.9, 256, dtype=np.float64)
    blob = write_it_module(one_sample_module(depth=depth, frames=256))
    _, (smp,), _ = offset_tables(blob)
    pointer = u32(blob, smp + 72)
    stored = np.frombuffer(blob, dtype=dtype, count=256, offset=pointer)
    expected = np.clip(np.round(signal * scale), -scale, scale - 1).astype(dtype)
    assert np.array_equal(stored, expected)


def test_unsupported_depth_raises(one_sample_module: Blob) -> None:
    with pytest.raises(ValueError, match="unsupported depth"):
        write_it_module(one_sample_module(depth=24))


def test_sample_header_serializer_rejects_unsupported_depth(ramp: Callable[..., np.ndarray]) -> None:
    with pytest.raises(ValueError, match="unsupported depth"):
        _sample_header(ITSample("s", ramp(8), depth_bits=24), data_offset=0)
