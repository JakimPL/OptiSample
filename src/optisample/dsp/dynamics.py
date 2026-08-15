from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import lfilter

from optisample.config.dynamics import DynamicsConfig, HoldConfig
from optisample.dsp.envelope import LevelReading, local_level
from optisample.dsp.level import db_to_gain, gain_to_db, mean_energy, peak_amplitude

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


def _over_reduction_db(over_db: Signal, config: HoldConfig) -> Signal:
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


def reduction_db(level: Signal, *, reference: float, config: HoldConfig) -> Signal:
    """How far each moment of ``level`` is held back, in decibels, read against ``reference``.

    The one place the hold curve lives, so a caller states where the level comes from and how it is
    referenced while the knee, the ratio and the threshold are read the same way wherever the curve is
    asked for. Reading the threshold against a reference the level carries with it makes the answer
    scale-invariant: the same material is held back the same way however hot it stands.
    """
    return _over_reduction_db(gain_to_db(level / reference) - config.threshold_db, config)


def held_back(signal: Signal, reading: LevelReading, config: HoldConfig) -> Signal:
    """The gain each frame of ``signal`` is held back by, read against the level the whole of it carries.

    The detector is the level curve the recording's own pitch settles
    (:func:`~optisample.dsp.envelope.local_level`), so the weighting's reach either side of a frame is both
    the attack and the release at once: symmetric, zero-phase, and as long as the material asks for --
    twenty milliseconds for anything above G#2, widening toward eighty at the bottom of the band. A
    transient is therefore taken down as it arrives rather than after it, and since the gain is a function
    of a curve already held to that band it moves the level while leaving the waveform inside a cycle
    exactly as it stands.

    The reference is the signal's own root mean square, so ``threshold_db`` states how far **above** the
    body of the material the hold opens. Material standing where it sits throughout is therefore passed
    through as it is, and what the curve answers is the excursions alone -- which is what the pass is for,
    since a level a written curve already stated costs the depth nothing.
    """
    level = local_level(signal, reading)
    body = float(np.sqrt(mean_energy(signal)))
    if body <= 0.0:
        return np.ones_like(level)

    return db_to_gain(reduction_db(level, reference=body, config=config))


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

    held = reduction_db(_level(data, sample_rate, config.rms_window_s), reference=peak, config=config)
    reduction = _one_pole(held, sample_rate, config.gain_smoothing_s)
    return np.asarray(data * db_to_gain(reduction), dtype=np.float64)
