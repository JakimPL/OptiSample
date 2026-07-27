from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.loop import Loop
from trackmod.core.samples.depth import BitDepth

Signal = NDArray[np.float64]


@dataclass(frozen=True)
class StoredSample:
    """An encoded, tracker-ready sample: its PCM, stored rate/depth, natural pitch, and applied gain."""

    pcm: Signal
    sample_rate: int
    depth_bits: int
    root_pitch: int
    gain: float = 1.0
    loop: Loop | None = None

    @property
    def frames(self) -> int:
        return int(self.pcm.size)

    @property
    def depth(self) -> BitDepth:
        """The stored depth as the tracker vocabulary names it, which is what prices the frames.

        Raises:
            ValueError: when the sample was encoded at a depth no tracker format stores.
        """
        return BitDepth(self.depth_bits)

    @property
    def playback_gain(self) -> float:
        """What playback multiplies the stored PCM by to sound at the level it was recorded at.

        Storing hot spends the whole depth on one recording, which is what :attr:`gain` records; playing
        the sample back through this undoes that scaling, so a quiet recording sounds quiet again and an
        instrument keeps the balance between its samples. A sample stored at unit gain plays as it is.
        """
        return 1.0 / self.gain if self.gain > 0.0 else 1.0

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0
