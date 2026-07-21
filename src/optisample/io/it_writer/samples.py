"""The stored-sample record: its data shape, PCM quantization, and 80-byte IMPS header."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.surrogate import MAX_VOLUME
from optisample.io.it_format import SAMPLE_HEADER
from optisample.io.it_writer.constants import (
    _CVT_SIGNED,
    _IMPS,
    _INT8_SCALE,
    _INT16_SCALE,
    _SMP_FLAG_16BIT,
    _SMP_FLAG_DATA,
    _SMP_FLAG_LOOP,
    NAME_BYTES,
    _ascii,
)

_DEFAULT_DEPTH_BITS: Final = 16
_DEFAULT_C5SPEED_HZ: Final = 44_100


@dataclass(frozen=True)
class ITSample:
    """One stored sample: its PCM (float in ``[-1, 1]``), storage depth and playback rate."""

    name: str
    pcm: NDArray[np.floating]
    depth_bits: int = _DEFAULT_DEPTH_BITS
    c5speed: int = _DEFAULT_C5SPEED_HZ
    global_volume: int = MAX_VOLUME
    default_volume: int = MAX_VOLUME
    loop: tuple[int, int] | None = None  # forward loop over half-open frame range [begin, end)

    @property
    def frames(self) -> int:
        return int(np.asarray(self.pcm).size)


def _depth_dtype_and_scale(depth_bits: int) -> tuple[str, float]:
    """Return the little-endian signed dtype and full-scale factor for an 8- or 16-bit depth.

    This is the single home of the supported-depth guard: both the PCM conversion and the sample-header
    flag derive their 8-vs-16-bit choice from it, so an unsupported depth raises in exactly one place.
    """
    if depth_bits == 16:
        return "<i2", _INT16_SCALE
    if depth_bits == 8:
        return "<i1", _INT8_SCALE
    raise ValueError(f"unsupported depth {depth_bits} (expected 8 or 16)")


def _pcm_bytes(sample: ITSample) -> bytes:
    """Convert float PCM in ``[-1, 1]`` to signed little-endian bytes at the sample's depth."""
    pcm = np.asarray(sample.pcm, dtype=np.float64)
    dtype, scale = _depth_dtype_and_scale(sample.depth_bits)
    quantized = np.clip(np.round(pcm * scale), -scale, scale - 1).astype(dtype)
    return quantized.tobytes()


def _sample_header(sample: ITSample, data_offset: int) -> bytes:
    """Serialize an 80-byte IMPS sample header pointing at ``data_offset``."""
    dtype, _ = _depth_dtype_and_scale(sample.depth_bits)  # validates the depth; raises if unsupported
    flags = _SMP_FLAG_DATA | (_SMP_FLAG_16BIT if dtype == "<i2" else 0)
    loop_begin, loop_end = 0, 0
    if sample.loop is not None:
        flags |= _SMP_FLAG_LOOP
        loop_begin, loop_end = sample.loop  # Loop End is the frame after the loop; playback wraps here.
    return SAMPLE_HEADER.pack(
        {
            "magic": _IMPS,
            "global_volume": min(sample.global_volume, MAX_VOLUME),
            "flags": flags,
            "default_volume": min(sample.default_volume, MAX_VOLUME),
            "name": _ascii(sample.name, NAME_BYTES),
            "convert": _CVT_SIGNED,
            "length": sample.frames,
            "loop_begin": loop_begin,
            "loop_end": loop_end,
            "c5speed": int(sample.c5speed),
            "sample_pointer": data_offset,
        }
    )
