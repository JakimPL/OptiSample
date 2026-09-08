from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from optisample.config.loop import EnvelopeConfig
from optisample.dsp.level import db_to_gain, gain_to_db, peak_amplitude
from optisample.dsp.series import hann_kernel, weighted_mean

Signal = NDArray[np.float64]

_PERIODS_PER_KERNEL: Final = 2  # periods of the lowest frequency the weighting spans, which puts it on a null
_QUIET_LEVEL: Final = 1e-12  # the level material carrying no sound reads as, which leaves the split defined


@dataclass(frozen=True)
class Decomposition:
    """A recording as the level it holds times the carrier that level scales: ``signal == level * carrier``.

    The split is exact for any strictly positive level, so the pair between them carries everything the
    recording did. ``level`` is the smooth amplitude the material keeps as it sounds, which is where its
    dynamics live and what a fitted ramp or a tracker envelope stands in for. ``carrier`` is the same sound
    at one loudness throughout, which is where its timbre lives: a spectrum read off it states the shape of
    the sound at that moment, and a stored copy of it spends the whole depth of its grid on the waveform.
    """

    level: Signal
    carrier: Signal

    @property
    def level_db(self) -> Signal:
        """The level in decibels, which is the scale a ringing note falls straight on and is stored on."""
        return gain_to_db(self.level)

    def recombined(self) -> Signal:
        """The recording the split was taken from, put back together as level times carrier."""
        return np.asarray(self.level * self.carrier, dtype=np.float64)


@dataclass(frozen=True)
class LevelReading:
    """How one recording has its level read: the weighting a mean square is taken under, and the floor it clears.

    The weighting follows from the pitch the recording was played at (:func:`level_reading`), so a note is
    read over a stretch of its own material rather than over a span fixed for every note alike. One reading
    serves every level taken of one recording, which is what has a candidate loop and the region finally
    stored measured under the same weighting.
    """

    kernel: Signal
    floor_db: float

    @property
    def reach(self) -> int:
        """How far the weighting reaches either side of a frame, which is the material one reading spans."""
        return self.kernel.size // 2


def power_kernel(sample_rate: int, frequency: float) -> Signal:
    """The unit-sum weighting a local mean square is read under, spanning two periods of ``frequency``.

    A tone carries its power at twice its own frequency, and a Hann weighting lasting ``span`` seconds
    answers zero at every multiple of ``1 / span`` from ``2 / span`` up. Spanning two periods of
    ``frequency`` puts that first zero on ``frequency`` itself, so the ripple of every tone from half of it
    upward averages away and what the weighting leaves is the level the material holds. Between the zeros
    the shape holds the sidelobes some 31 dB down, which keeps a tone landing between two of them as flat
    as one landing on one.

    The tap count is odd, so the weighting reads each frame from the material centered on it.
    """
    return hann_kernel(round(_PERIODS_PER_KERNEL * sample_rate / frequency))


def reading_frequency(config: EnvelopeConfig, root_hz: float) -> float:
    """The frequency a recording's weighting spans two periods of: its own pitch, held inside the config's band.

    A note's own period is the stretch its power ripple repeats in, so spanning two of them is what leaves
    the reading holding the level while a note two octaves higher is read over a proportionally shorter
    stretch and follows its own movement just as closely.

    The band bounds how far that goes either way. ``lowest_hz`` caps the span, so the deepest notes are read
    over a stretch a loop region still has room for; ``highest_hz`` floors it, so the beating of two partials
    a few hertz apart stays in the carrier where it is heard as timbre.
    """
    return min(max(root_hz, config.lowest_hz), config.highest_hz)


def level_reading(sample_rate: int, config: EnvelopeConfig, root_hz: float) -> LevelReading:
    """The reading a recording played at ``root_hz`` and held at ``sample_rate`` has every level taken under."""
    return LevelReading(
        kernel=power_kernel(sample_rate, reading_frequency(config, root_hz)),
        floor_db=config.floor_db,
    )


def local_level_over(signal: Signal, reading: LevelReading, *, start: int, end: int) -> Signal:
    """The level ``signal`` holds over ``[start, end)``, read with the material around it the weighting spans.

    The weighting reaches half its taps either side of a frame, so reading a stretch together with that much
    of the recording on each side answers what reading the whole recording and taking that stretch of it
    answers. Leveling a loop region therefore costs the region's own length, whatever the recording it came
    from runs to.

    The mean is taken of the material's power, with the level read off that afterwards, which is what puts
    the curve on the level the material holds: power is the quantity that averages, so the reading stays a
    true local level straight through the zero crossings the waveform makes.
    """
    low, high = max(0, start - reading.reach), min(signal.size, end + reading.reach)
    read = np.asarray(signal[low:high], dtype=np.float64)
    floor = max(db_to_gain(-reading.floor_db) * peak_amplitude(signal), _QUIET_LEVEL)
    level = np.sqrt(np.maximum(weighted_mean(read**2, reading.kernel), 0.0) + floor**2)
    return np.asarray(level[start - low : end - low], dtype=np.float64)


def local_level(signal: Signal, reading: LevelReading) -> Signal:
    """The level ``signal`` holds at each of its frames: one smooth, strictly positive amplitude curve.

    The curve occupies the band under the frequency the weighting was formed at, so it is worth a few
    hundred points a note however long the note runs, and it stays clear of zero by the reading's floor,
    which is what makes dividing the recording by it a split that can be put back together.
    """
    return local_level_over(signal, reading, start=0, end=signal.size)


def decompose(signal: Signal, reading: LevelReading) -> Decomposition:
    """Split ``signal`` into the level it holds and the carrier that level scales.

    Scaling a recording as a whole scales its level by the same amount and leaves its carrier as it stands,
    so what the split reads of a sound is a property of the sound at whatever level it was captured at.
    """
    level = local_level(signal, reading)
    return Decomposition(level=level, carrier=np.asarray(signal / level, dtype=np.float64))
