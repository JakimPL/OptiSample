"""Exact uncompressed IT size model.

Impulse Tracker stores signed little-endian PCM; ``Length`` counts frames. Overhead is fixed:
80 B per sample header, 554 B per instrument header (incl. the 240-B note map), ~192 B file
header. IT214/215 compression is content-dependent (no fixed ratio); the optimizer stays correct
on this uncompressed model regardless.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.dsp.quantize import VALID_DEPTHS

SAMPLE_HEADER_BYTES: Final = 80
INSTRUMENT_HEADER_BYTES: Final = 554
FILE_HEADER_BYTES: Final = 192
_BYTES_PER_KIB: Final = 1024
_MONO_CHANNELS: Final = 1


def _bytes_per_frame(depth_bits: int, channels: int) -> int:
    if depth_bits not in VALID_DEPTHS:
        raise ValueError(f"IT samples are 8- or 16-bit, got {depth_bits}")
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    return (depth_bits // 8) * channels


@dataclass(frozen=True)
class SampleSize:
    """Storage footprint of one stored sample."""

    frames: int
    depth_bits: int
    channels: int = _MONO_CHANNELS

    @property
    def pcm_bytes(self) -> int:
        return self.frames * _bytes_per_frame(self.depth_bits, self.channels)

    @property
    def total_bytes(self) -> int:
        return self.pcm_bytes + SAMPLE_HEADER_BYTES


def module_bytes(samples: Sequence[SampleSize], n_instruments: int) -> int:
    """Total uncompressed module size: file header + instrument headers + sample headers + PCM."""
    return FILE_HEADER_BYTES + n_instruments * INSTRUMENT_HEADER_BYTES + sum(sample.total_bytes for sample in samples)


def max_frames_for_budget(budget_bytes: int, depth_bits: int, channels: int = _MONO_CHANNELS) -> int:
    """Largest frame count whose stored sample (incl. header) fits in ``budget_bytes`` (0 if none)."""
    usable = budget_bytes - SAMPLE_HEADER_BYTES
    if usable <= 0:
        return 0
    return usable // _bytes_per_frame(depth_bits, channels)


def bytes_to_kib(num_bytes: float) -> float:
    return num_bytes / _BYTES_PER_KIB


def kib_to_bytes(kib: float) -> int:
    return int(round(kib * _BYTES_PER_KIB))
