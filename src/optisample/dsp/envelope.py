from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from optisample.config.loop import EnvelopeConfig
from optisample.dsp.levels import db_to_gain, gain_to_db, peak_amplitude

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


def power_kernel(sample_rate: int, lowest_hz: float) -> Signal:
    """The unit-sum weighting a local mean square is read under, spanning two periods of ``lowest_hz``.

    A tone carries its power at twice its own frequency, and a Hann weighting lasting ``span`` seconds
    answers zero at every multiple of ``1 / span`` from ``2 / span`` up. Spanning two periods of
    ``lowest_hz`` puts that first zero on ``lowest_hz`` itself, so the ripple of every tone from half of it
    upward averages away and what the weighting leaves is the level the material holds. Between the zeros
    the shape holds the sidelobes some 31 dB down, which keeps a tone landing between two of them as flat
    as one landing on one.

    The tap count is odd, so the weighting reads each frame from the material centred on it.
    """
    span = round(_PERIODS_PER_KERNEL * sample_rate / lowest_hz)
    weighting = np.hanning(span + span % 2 + 1)
    return np.asarray(weighting / np.sum(weighting), dtype=np.float64)


def _weighted_mean(values: Signal, kernel: Signal) -> Signal:
    """The mean of ``values`` around each of its frames under ``kernel``, over the weight it covers there.

    Dividing by the weight landing inside the stretch keeps the first and last frames a mean of the material
    that is there, so the curve holds its level at both ends of a recording.
    """
    covered = fftconvolve(np.ones_like(values), kernel, mode="same")
    return np.asarray(fftconvolve(values, kernel, mode="same") / covered, dtype=np.float64)


def local_level_over(
    signal: Signal,
    sample_rate: int,
    config: EnvelopeConfig,
    *,
    start: int,
    end: int,
) -> Signal:
    """The level ``signal`` holds over ``[start, end)``, read with the material around it the weighting spans.

    The weighting reaches half its taps either side of a frame, so reading a stretch together with that much
    of the recording on each side answers what reading the whole recording and taking that stretch of it
    answers. Levelling a loop region therefore costs the region's own length, whatever the recording it came
    from runs to.

    The mean is taken of the material's power, with the level read off that afterwards, which is what puts
    the curve on the level the material holds: power is the quantity that averages, so the reading stays a
    true local level straight through the zero crossings the waveform makes.
    """
    kernel = power_kernel(sample_rate, config.lowest_hz)
    reach = kernel.size // 2
    low, high = max(0, start - reach), min(signal.size, end + reach)
    read = np.asarray(signal[low:high], dtype=np.float64)
    floor = max(db_to_gain(-config.floor_db) * peak_amplitude(signal), _QUIET_LEVEL)
    level = np.sqrt(np.maximum(_weighted_mean(read**2, kernel), 0.0) + floor**2)
    return np.asarray(level[start - low : end - low], dtype=np.float64)


def local_level(signal: Signal, sample_rate: int, config: EnvelopeConfig) -> Signal:
    """The level ``signal`` holds at each of its frames: one smooth, strictly positive amplitude curve.

    The curve occupies the band under ``config.lowest_hz``, so it is worth a few hundred points a note
    however long the note runs, and it stays clear of zero by the floor
    (:class:`~optisample.config.loop.EnvelopeConfig`), which is what makes dividing the recording by it
    a split that can be put back together.
    """
    return local_level_over(signal, sample_rate, config, start=0, end=signal.size)


def decompose(signal: Signal, sample_rate: int, config: EnvelopeConfig) -> Decomposition:
    """Split ``signal`` into the level it holds and the carrier that level scales.

    Scaling a recording as a whole scales its level by the same amount and leaves its carrier as it stands,
    so what the split reads of a sound is a property of the sound at whatever level it was captured at.
    """
    level = local_level(signal, sample_rate, config)
    return Decomposition(level=level, carrier=np.asarray(signal / level, dtype=np.float64))
