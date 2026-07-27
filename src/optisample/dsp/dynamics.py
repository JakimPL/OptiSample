from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from optisample.config.dynamics import DynamicsConfig
from optisample.dsp.levels import db_to_gain, gain_to_db, peak_amplitude

Signal = NDArray[np.float64]

_UNITY: Final = 1.0
_HALF_KNEE: Final = 2.0  # the knee is stated as a full width, so each side of the threshold gets half
_NARROWEST_KNEE_DB: Final = 1e-9  # keeps a knee asked to be square a width the quadratic can divide by


def _one_pole(signal: Signal, sample_rate: int, time_constant_s: float) -> Signal:
    """Lag ``signal`` by one pole, covering ``1 - 1/e`` of a step within ``time_constant_s``."""
    decay = float(np.exp(-_UNITY / (time_constant_s * sample_rate)))
    return np.asarray(lfilter([_UNITY - decay], [_UNITY, -decay], signal), dtype=np.float64)


def _level(signal: Signal, sample_rate: int, window_s: float) -> Signal:
    """The signal's running amplitude: its square lagged over ``window_s``, rooted back to amplitude."""
    return np.sqrt(np.maximum(_one_pole(signal * signal, sample_rate, window_s), 0.0))


def _reduction_db(over_db: Signal, config: DynamicsConfig) -> Signal:
    """The gain reduction the curve asks for at each level, stated as decibels over the threshold.

    Under the knee the curve stays flat, over it each decibel of excess keeps ``1 / ratio`` of itself,
    and across the knee's width the two meet along the quadratic sharing a slope with both ends.
    """
    kept = _UNITY / config.ratio - _UNITY
    knee = max(config.knee_db, _NARROWEST_KNEE_DB)
    edge = knee / _HALF_KNEE
    bend = kept * np.square(over_db + edge) / (_HALF_KNEE * knee)
    return np.asarray(
        np.select([over_db <= -edge, over_db >= edge], [np.zeros_like(over_db), kept * over_db], bend),
        dtype=np.float64,
    )


def compress(signal: Signal, sample_rate: int, config: DynamicsConfig) -> Signal:
    """Narrow ``signal``'s crest factor: hold back whatever rises past the threshold under its own peak.

    A feed-forward soft-knee compressor. The level is read as a lagged RMS, the curve turns each level
    into the reduction it asks for, and a second lag smooths that reduction before it is applied, so the
    gain follows the material rather than each sample. Both lags are one-pole and symmetric, which keeps
    the whole pass a handful of vectorized steps and makes it reproduce exactly wherever it runs.

    Reading the threshold against the clip's own peak makes the result scale-invariant: a recording
    compresses the same way however hot it was captured, and the normalization that follows sets the
    level. A silent clip is returned as it stands.
    """
    data = np.asarray(signal, dtype=np.float64)
    peak = peak_amplitude(data)
    if peak <= 0.0:
        return data.copy()

    over_db = gain_to_db(_level(data, sample_rate, config.rms_window_s) / peak) - config.threshold_db
    reduction = _one_pole(_reduction_db(over_db, config), sample_rate, config.gain_smoothing_s)
    return np.asarray(data * db_to_gain(reduction), dtype=np.float64)
