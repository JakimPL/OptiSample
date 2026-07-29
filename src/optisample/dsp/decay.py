from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.dsp.levels import mean_energy
from optisample.dsp.loop import Loop

Signal = NDArray[np.float64]

NO_DECAY: Final = None  # a stored sample whose own material carries every level it plays at
_LEVEL_WINDOW_S: Final = 0.05  # the stretch one reading of the material's level averages over
_MIN_READINGS: Final = 2  # readings a line can be drawn through, which is what states a decline
_LEVEL_FLOOR: Final = 1e-12  # the level a silent stretch reads as, which leaves a ratio against it finite
_STEADY_GAIN: Final = 0.99  # a fall of under a percent is a level held, so the ramp is left off


@dataclass(frozen=True)
class LinearDecay:
    """The straight amplitude ramp a looped sample declines on, in seconds from the note's onset.

    A loop repeats one region for as long as a note is held, so the sample holds the level that region
    was recorded at. This is the decline the recording goes on making past it: unit gain up to
    ``start_s``, a straight line down to ``final_gain`` at ``end_s``, and ``final_gain`` held for as long
    as the note runs on. ``start_s`` is where the stored material ends, so the ramp begins exactly where
    the sample stops carrying the recording's own envelope.

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


def _level(signal: Signal) -> float:
    """RMS amplitude of ``signal``: the level a stretch holds, read past the phase of its waveform."""
    return float(np.sqrt(mean_energy(signal)))


def _block_levels(signal: Signal, block: int) -> tuple[Signal, Signal]:
    """The level of each whole ``block``-frame window of ``signal``, and the frame each one centres on."""
    count = signal.size // block
    windows = np.asarray(signal[: count * block], dtype=np.float64).reshape(count, block)
    return np.sqrt(np.mean(windows**2, axis=1)), (np.arange(count, dtype=np.float64) + 0.5) * block


def _level_at(levels: Signal, seconds: Signal, moment_s: float) -> float:
    """The least-squares line through the level readings, read at ``moment_s``.

    Taking the line through every reading lets the body of the material set the decline: a note holding
    its level until a short release at the very end reports the shallow fall it spent its length making,
    and a struck note reports the steep one. A line also states the level at a moment past the readings,
    which is where the ramp has to land.
    """
    centered = seconds - float(np.mean(seconds))
    slope = float(np.sum(centered * levels) / np.sum(centered**2))
    return float(np.mean(levels)) + slope * (moment_s - float(np.mean(seconds)))


def fit_linear_decay(signal: Signal, sample_rate: int, loop: Loop) -> LinearDecay | None:
    """The decline ``signal`` goes on making past the loop a sample stores of it.

    Storage ends at ``loop.end``, so the ramp holds unit gain up to there and what follows is read off
    the material the loop stands in for: its level is sampled in short windows and a line drawn through
    them (:func:`_level_at`) says where the recording ends up. The loop repeats at the level of the
    region it holds, so the ratio of the two is how far a held note is played down by the time the
    material runs out.

    Returns ``None`` where the recording states no decline worth playing a note down by: material past
    the loop holding fewer than ``_MIN_READINGS`` windows, a silent loop region, or a level still within
    ``_STEADY_GAIN`` of the loop's own by the time the recording ends.
    """
    block = max(1, round(_LEVEL_WINDOW_S * sample_rate))
    remaining = np.asarray(signal[loop.end :], dtype=np.float64)
    held = _level(signal[loop.start : loop.end])
    if remaining.size < _MIN_READINGS * block or held <= _LEVEL_FLOOR:
        return NO_DECAY

    levels, centres = _block_levels(remaining, block)
    span_s = remaining.size / sample_rate
    final_gain = max(_level_at(levels, centres / sample_rate, span_s), 0.0) / held
    if final_gain > _STEADY_GAIN:
        return NO_DECAY

    return LinearDecay(
        start_s=loop.end / sample_rate,
        end_s=signal.size / sample_rate,
        final_gain=final_gain,
    )
