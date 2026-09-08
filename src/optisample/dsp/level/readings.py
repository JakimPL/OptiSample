from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.dsp.level.decibels import Signal, db_to_gain, gain_to_db
from optisample.dsp.series import Readings

_LEVEL_WINDOW_S: Final = 0.05  # the stretch one reading of the material's level averages over
_MIN_READINGS: Final = 2  # readings a line can be drawn through, which is what states a trend


@dataclass(frozen=True)
class DecayTrend:
    """The straight line through a stretch's level readings in decibels, in seconds from its own first frame.

    A ringing note loses a steady number of decibels a second, so its readings lie on a line once they are
    read as levels rather than as amplitudes, and the line states where the note has reached at any moment
    it spans. Fitting the same readings in amplitude follows a curve instead, which reads the far end of a
    long stretch well under the level the material holds there.
    """

    mean_db: float
    mean_s: float
    slope_db: float

    def at(self, moment_s: float) -> float:
        """The amplitude the line reads at ``moment_s`` seconds into the stretch it was drawn through."""
        return db_to_gain(self.mean_db + self.slope_db * (moment_s - self.mean_s))


def level_readings(signal: Signal, sample_rate: int, *, window_s: float) -> Readings:
    """``signal``'s level in decibels, read in whole ``window_s`` windows, beside the moment each centers on.

    Reading the level this way leaves the material's envelope in the series and the phase of its waveform
    out of it, which is what a line or a curve is drawn through. The window length is the caller's because
    it settles how many readings come back, and what a reading is worth to the caller -- a line asks for
    little and a fit placing corners asks for as many as it can table
    (:func:`~optisample.dsp.piecewise.fit_piecewise`). A stretch shorter than one whole window is read in
    no windows at all, which comes back holding nothing.
    """
    block = max(1, round(window_s * sample_rate))
    data = np.asarray(signal, dtype=np.float64)
    count = data.size // block
    windows = data[: count * block].reshape(count, block)
    return Readings(
        values=gain_to_db(np.sqrt(np.mean(windows**2, axis=1))),
        seconds=(np.arange(count, dtype=np.float64) + 0.5) * block / sample_rate,
    )


def _line(values: Signal, seconds: Signal) -> tuple[float, float, float]:
    """Mean value, mean moment, and slope of the least-squares line through ``values`` against ``seconds``."""
    mean_s = float(np.mean(seconds))
    centered = seconds - mean_s
    return float(np.mean(values)), mean_s, float(np.sum(centered * values) / np.sum(centered**2))


def decay_trend(signal: Signal, sample_rate: int) -> DecayTrend | None:
    """The line ``signal``'s own level readings make in decibels, in seconds from its first frame.

    This is the reading to take of material that rings: a note losing a steady number of decibels a second
    puts its readings on a straight line here, so the fit states the level the note reaches at the far end
    of a stretch as well as at its middle. Returns ``None`` for a stretch holding fewer than
    ``_MIN_READINGS`` whole windows, which is too little for a line to be drawn through.
    """
    readings = level_readings(signal, sample_rate, window_s=_LEVEL_WINDOW_S)
    if readings.count < _MIN_READINGS:
        return None

    mean_db, mean_s, slope_db = _line(readings.values, readings.seconds)
    return DecayTrend(mean_db=mean_db, mean_s=mean_s, slope_db=slope_db)
