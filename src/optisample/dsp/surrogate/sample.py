"""The encoded, IT-ready stored sample and the IT note-volume ceiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.loop import Loop
from optisample.metrics.size import SampleSize

Signal = NDArray[np.float64]

MAX_VOLUME: Final = 64  # IT note volume spans 0x00..0x40 (linear in amplitude).


@dataclass(frozen=True)
class StoredSample:
    """An encoded, IT-ready sample: its PCM, stored rate/depth, natural pitch, and applied gain."""

    pcm: Signal
    sample_rate: int
    depth_bits: int
    root_pitch: int
    gain: float = 1.0  # normalization gain applied at encode time (stored = source * gain).
    loop: Loop | None = None  # forward loop over stored frames, sustaining notes held past the sample

    @property
    def frames(self) -> int:
        return int(self.pcm.size)

    @property
    def size(self) -> SampleSize:
        return SampleSize(frames=self.frames, depth_bits=self.depth_bits)

    @property
    def stored_bytes(self) -> int:
        return self.size.total_bytes

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0
