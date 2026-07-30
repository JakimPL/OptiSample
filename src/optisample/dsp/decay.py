from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.levels import decay_trend, level_trend
from optisample.dsp.loop import Loop

Signal = NDArray[np.float64]

NO_DECAY: Final = None  # a stored sample whose own material carries every level it plays at
_LEVEL_FLOOR: Final = 1e-12  # the level a silent stretch reads as, which leaves a ratio against it finite
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


def fit_linear_decay(signal: Signal, sample_rate: int, loop: Loop) -> LinearDecay | None:
    """The decline ``signal`` makes from the loop a sample stores of it.

    The stored region is held at the level it starts on (:func:`~optisample.dsp.loop.level_loop`), so the
    ramp holds unit gain up to ``loop.start`` and states the fall the recording makes from there. The level
    the region holds is the line through its own readings read at its first frame, which is the same reading
    levelling pinned it to (:func:`~optisample.dsp.levels.level_trend`). Where a held note ends up is read
    off everything from ``loop.start`` on, as the line those readings make in decibels
    (:func:`~optisample.dsp.levels.decay_trend`) -- the domain a ringing note falls straight in, so the
    reading holds at the far end of a remainder however long the loop left it. The ratio of the two is how
    far the note is played down by the time the material runs out.

    Returns ``None`` where the recording states no decline worth playing a note down by: a region or a
    remainder too short for a line to be drawn through, a region starting from silence, or a level still
    within ``_STEADY_GAIN`` of the region's own by the time the recording ends.
    """
    remaining = np.asarray(signal[loop.start :], dtype=np.float64)
    region = np.asarray(signal[loop.start : loop.end], dtype=np.float64)
    held, onward = level_trend(region, sample_rate), decay_trend(remaining, sample_rate)
    if held is None or onward is None or held.at(0.0) <= _LEVEL_FLOOR:
        return NO_DECAY

    final_gain = onward.at(remaining.size / sample_rate) / held.at(0.0)
    if final_gain > _STEADY_GAIN:
        return NO_DECAY

    return LinearDecay(
        start_s=loop.start / sample_rate,
        end_s=signal.size / sample_rate,
        final_gain=final_gain,
    )
