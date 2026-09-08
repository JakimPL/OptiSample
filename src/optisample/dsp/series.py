from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

Series = NDArray[np.float64]

_PEAK_CURVATURE: Final = 1e-12  # concavity a peak holds for a parabola to read a lag between points


@dataclass(frozen=True, eq=False)
class Readings:
    """A stretch read in windows: one value per window, beside the moment that window centers on.

    Reading a stretch this way leaves a curve short enough to fit whole and smooth enough to follow, and
    carrying each value's own moment beside it puts every fit on the played timeline -- so readings taken
    at one window length and a curve fitted through them state their results on the same clock.

    Moments ascend, which is what lets a fit place its corners in the order they are played.
    """

    values: Series
    seconds: Series

    @property
    def count(self) -> int:
        """How many windows the stretch was read in, which is what a fit's own cost is counted in."""
        return int(self.values.size)

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` is the same reading of the same stretch, value for value and moment for moment.

        Readings travel inside the values a stage settles and a document reads back
        (:class:`~optisample.dsp.level.Level`), so two of them compare as the series they are -- which is
        what lets a settlement, a container and a run in another process be checked against each other.
        """
        if not isinstance(other, Readings):
            return NotImplemented

        return bool(np.array_equal(self.values, other.values) and np.array_equal(self.seconds, other.seconds))


def hann_kernel(span: int) -> Series:
    """A unit-sum Hann weighting spanning ``span`` points, with an odd tap count.

    An odd count reads each point from the material centered on it, and a unit sum leaves a mean taken
    under the weighting on the scale of what it averages. A span of zero is one tap, which is the point
    itself.
    """
    taps = np.hanning(span + span % 2 + 1)
    return np.asarray(taps / np.sum(taps), dtype=np.float64)


def weighted_mean(values: Series, kernel: Series) -> Series:
    """The mean of ``values`` around each of its points under ``kernel``, over the weight it covers there.

    Dividing by the weight landing inside the series keeps the first and last points a mean of the
    material that is there, so the curve holds its level at both ends.
    """
    covered = fftconvolve(np.ones_like(values), kernel, mode="same")
    return np.asarray(fftconvolve(values, kernel, mode="same") / covered, dtype=np.float64)


def autocorrelation(series: Series) -> Series:
    """How alike ``series`` is to itself at each lag from ``0`` to its length, normalized so lag 0 reads 1.0.

    Centering first leaves the reading a statement about the movement the series makes rather than the
    level it sits at, and the transform runs over a power of two past twice the length so the wrap a
    circular correlation makes lands outside the lags read.

    A series holding one value throughout reads 0.0 at every lag, which is material a period has no
    purchase on.
    """
    centered = series - float(np.mean(series))
    length = centered.size
    size = 1 << int(np.ceil(np.log2(2 * length)))
    spectrum = np.fft.rfft(centered, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[:length]
    if correlation[0] <= 0.0:
        return np.zeros(length, dtype=np.float64)

    return np.asarray(correlation / correlation[0], dtype=np.float64)


def refined_lag(correlation: Series, lag: int) -> float:
    """The lag ``correlation`` peaks at, read between points by the parabola through its top three.

    A peak read to the nearest point sits up to half of one from the material's own cycle, and a loop
    spans several periods, so that error accumulates into a wrap landing part-way through a cycle.
    Fitting a parabola to the peak and its two neighbors states the lag to a fraction of a point, which
    is what lets a whole number of periods mean what it says at every rate a recording is analyzed at.

    A peak on the first or last lag searched, or one the neighbors make no concave top with, is read at
    its own point -- there is no parabola through it to place the lag between points.
    """
    if lag <= 0 or lag >= correlation.size - 1:
        return float(lag)

    before, peak, after = float(correlation[lag - 1]), float(correlation[lag]), float(correlation[lag + 1])
    curvature = before - 2.0 * peak + after
    if curvature > -_PEAK_CURVATURE:
        return float(lag)

    return float(lag) + float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))
