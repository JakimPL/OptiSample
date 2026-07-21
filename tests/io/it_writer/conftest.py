"""Shared builders and byte-readers for the IT-writer format tests.

The ``playback`` value-object comes from the root conftest (built from the bundled config); these
format tests do not assert on it (except the dedicated playback test, which builds its own), so they
share it. ``one_sample_module`` is the minimal one-note module most header tests write and parse.
"""

from __future__ import annotations

import struct
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.io.it_writer import ITCell, ITInstrument, ITModule, ITPattern, ITPlayback, ITSample, identity_note_map
from optisample.metrics.size import FILE_HEADER_BYTES

Reader = Callable[[bytes, int], int]


@pytest.fixture
def u16() -> Reader:
    """Read a little-endian ``uint16`` at ``offset``."""
    return lambda blob, offset: int(struct.unpack_from("<H", blob, offset)[0])


@pytest.fixture
def u32() -> Reader:
    """Read a little-endian ``uint32`` at ``offset``."""
    return lambda blob, offset: int(struct.unpack_from("<I", blob, offset)[0])


@pytest.fixture
def ramp() -> Callable[..., NDArray[np.float64]]:
    """Factory: a linear ramp from -0.9 to 0.9 over ``frames`` (a simple, exactly-representable PCM)."""
    return lambda frames=256: np.linspace(-0.9, 0.9, frames, dtype=np.float64)


@pytest.fixture
def one_sample_module(ramp: Callable[..., NDArray[np.float64]], playback: ITPlayback) -> Callable[..., ITModule]:
    """Factory: the minimal one-note module (one sample, one instrument, one pattern)."""

    def _build(depth: int = 16, c5: int = 22_050, frames: int = 256, volume: int = 48) -> ITModule:
        sample = ITSample(name="ramp", pcm=ramp(frames), depth_bits=depth, c5speed=c5)
        instrument = ITInstrument(name="inst", note_map=identity_note_map({60: 1}))
        pattern = ITPattern(rows=8, cells=((0, 0, ITCell(note=60, instrument=1, volume=volume)),))
        return ITModule(
            name="probe",
            samples=(sample,),
            instruments=(instrument,),
            patterns=(pattern,),
            orders=(0,),
            playback=playback,
        )

    return _build


@pytest.fixture
def offset_tables(u16: Reader, u32: Reader) -> Callable[[bytes], tuple[list[int], list[int], list[int]]]:
    """Factory: parse the instrument/sample/pattern offset tables out of a written module."""

    def _offset_tables(blob: bytes) -> tuple[list[int], list[int], list[int]]:
        ins_num, smp_num, pat_num = u16(blob, 34), u16(blob, 36), u16(blob, 38)
        base = FILE_HEADER_BYTES + u16(blob, 32)  # after the file header + order list
        ins = [u32(blob, base + 4 * i) for i in range(ins_num)]
        smp = [u32(blob, base + 4 * ins_num + 4 * i) for i in range(smp_num)]
        pat = [u32(blob, base + 4 * ins_num + 4 * smp_num + 4 * i) for i in range(pat_num)]
        return ins, smp, pat

    return _offset_tables
