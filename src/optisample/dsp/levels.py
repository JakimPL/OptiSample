from typing import Final, cast, overload

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]

_DB_PER_DECADE: Final = 20.0  # amplitude decibels: a tenfold amplitude ratio is 20 dB
_SILENT_AMPLITUDE: Final = 1e-12  # the amplitude silence reads as, so its level stays a finite -240 dB


@overload
def db_to_gain(delta_db: float) -> float: ...
@overload
def db_to_gain(delta_db: Signal) -> Signal: ...
def db_to_gain(delta_db: float | Signal) -> float | Signal:
    """Amplitude gain for a level change of ``delta_db`` decibels (``10 ** (dB / 20)``).

    Loudness normalization, the velocity->volume map, the stored headroom and the compressor's gain
    curve are all linear in amplitude, so a level delta in dB becomes a multiplicative gain this way
    (``+6 dB`` ~ 2x). Accepts a scalar delta or a per-element array of deltas, returning the matching
    type.
    """
    return cast(float | Signal, 10.0 ** (delta_db / _DB_PER_DECADE))


def gain_to_db(gain: Signal) -> Signal:
    """Level of each amplitude in ``gain``, in decibels -- the inverse of :func:`db_to_gain`.

    Amplitudes under ``_SILENT_AMPLITUDE`` read as that floor, so a silent stretch carries a level a
    threshold can still be subtracted from.
    """
    floored = np.maximum(np.asarray(gain, dtype=np.float64), _SILENT_AMPLITUDE)
    return np.asarray(_DB_PER_DECADE * np.log10(floored), dtype=np.float64)


def peak_amplitude(signal: Signal) -> float:
    """Largest absolute sample in ``signal``; ``0.0`` when it holds none."""
    data = np.asarray(signal, dtype=np.float64)
    return float(np.max(np.abs(data))) if data.size else 0.0


def mean_energy(signal: Signal) -> float:
    """Mean square of ``signal`` -- the energy it carries per frame; ``0.0`` when it holds none.

    Full scale reads 1.0, so the measure states a stretch of audio against the loudest one storable and
    a level in dB is :func:`gain_to_db` of its square root.
    """
    data = np.asarray(signal, dtype=np.float64)
    return float(np.mean(data**2)) if data.size else 0.0
