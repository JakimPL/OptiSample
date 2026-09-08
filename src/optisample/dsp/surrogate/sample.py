from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from trackmod import BitDepth

from optisample.dsp.level import Clock, Level, unit_level
from optisample.dsp.loop import Loop

Signal = NDArray[np.float64]

NO_RELEASE_RAMP: Final = 0  # frames; a span reaching its end at the level it holds there
CARRIES_ITS_LEVEL: Final = unit_level(Clock.PLAYED)  # what a sample whose own PCM holds every level it plays at


@dataclass(frozen=True)
class StoredSample:
    """An encoded, tracker-ready sample: its PCM, stored rate/depth, natural pitch, gain, and how it ends.

    ``release_frames`` is how many of its last frames the ramp closing it covers, which the encoder
    applies and a score against its source puts over the source as well. ``level`` is the curve the sample
    is played down by, and it says what the PCM leaves to something else to supply: for a looped sample,
    the decline the recording went on making past the one level its loop repeats at; for a carrier, the
    whole level its instrument's volume envelope carries. It belongs to the slot rather than the waveform,
    so the PCM stays the material as it was stored, and it is counted on the played clock -- the timeline
    a tracker walks whatever key is struck.
    """

    pcm: Signal
    sample_rate: int
    depth: BitDepth
    root_pitch: int
    gain: float = 1.0
    loop: Loop | None = None
    release_frames: int = NO_RELEASE_RAMP
    level: Level = CARRIES_ITS_LEVEL

    @property
    def frames(self) -> int:
        return int(self.pcm.size)

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
