from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.envelope import LevelReading, local_level_over
from optisample.dsp.level import decay_trend
from optisample.dsp.loop import Loop

Signal = NDArray[np.float64]

NO_DECAY: Final = None  # a stored sample whose own material carries every level it plays at
_STEADY_GAIN: Final = 0.99  # a fall of under a percent is a level held, so the ramp is left off


@dataclass(frozen=True)
class LinearDecay:
    """The straight amplitude ramp a looped sample declines on, in seconds from the note's onset.

    A loop repeats one region held at a single level for as long as a note is held, so the sample carries
    one amplitude from ``start_s`` on. This is the decline the recording makes over that stretch: unit gain
    up to ``start_s``, a straight line down to ``final_gain`` at ``end_s``, and ``final_gain`` held for as
    long as the note runs on. ``start_s`` is where the stored material stops following the recording's own
    envelope, which is where the loop region begins.

    Seconds run on the played timeline, which is the clock a tracker's volume envelope runs on, so every
    key sounding the sample declines over the same stretch of time as its root does.
    """

    start_s: float
    end_s: float
    final_gain: float

    @property
    def span_s(self) -> float:
        """How long the ramp takes to reach ``final_gain``, which is at least one level reading long."""
        return self.end_s - self.start_s

    def envelope(self, frames: int, sample_rate: int) -> Signal:
        """The gain of each of the first ``frames`` frames of a note played at ``sample_rate``."""
        seconds = np.arange(frames, dtype=np.float64) / sample_rate
        progress = np.clip((seconds - self.start_s) / self.span_s, 0.0, 1.0)
        return np.asarray(1.0 + progress * (self.final_gain - 1.0), dtype=np.float64)


def fit_linear_decay(signal: Signal, sample_rate: int, loop: Loop, reading: LevelReading) -> LinearDecay | None:
    """The decline ``signal`` makes from the loop a sample stores of it.

    The stored region is held at the level it starts on (:func:`~optisample.dsp.loop.level_loop`), so the
    ramp holds unit gain up to ``loop.start`` and states the fall the recording makes from there. The level
    the region holds is the one the material holds at its first frame
    (:func:`~optisample.dsp.envelope.local_level_over`), which is the very reading levelling pinned it to.
    Where a held note ends up is read off everything from ``loop.start`` on, as the line those readings make
    in decibels (:func:`~optisample.dsp.level.readings.decay_trend`) -- the domain a ringing note falls straight in,
    so the reading holds at the far end of a remainder however long the loop left it. The ratio of the two
    is how far the note is played down by the time the material runs out.

    Returns ``None`` where the recording states no decline worth playing a note down by: a remainder too
    short for a line to be drawn through, or a level still within ``_STEADY_GAIN`` of the region's own by
    the time the recording ends.
    """
    remaining = np.asarray(signal[loop.start :], dtype=np.float64)
    onward = decay_trend(remaining, sample_rate)
    if onward is None:
        return NO_DECAY

    held = float(local_level_over(signal, reading, start=loop.start, end=loop.end)[0])
    final_gain = onward.at(remaining.size / sample_rate) / held
    if final_gain > _STEADY_GAIN:
        return NO_DECAY

    return LinearDecay(
        start_s=loop.start / sample_rate,
        end_s=signal.size / sample_rate,
        final_gain=final_gain,
    )
