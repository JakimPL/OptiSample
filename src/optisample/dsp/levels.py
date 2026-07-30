from dataclasses import dataclass
from typing import Final, cast, overload

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]

_DB_PER_DECADE: Final = 20.0  # amplitude decibels: a tenfold amplitude ratio is 20 dB
_SILENT_AMPLITUDE: Final = 1e-12  # the amplitude silence reads as, so its level stays a finite -240 dB
_LEVEL_WINDOW_S: Final = 0.05  # the stretch one reading of the material's level averages over
_MIN_READINGS: Final = 2  # readings a line can be drawn through, which is what states a trend


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


@overload
def gain_to_db(gain: float) -> float: ...
@overload
def gain_to_db(gain: Signal) -> Signal: ...
def gain_to_db(gain: float | Signal) -> float | Signal:
    """Level of each amplitude in ``gain``, in decibels -- the inverse of :func:`db_to_gain`.

    Amplitudes under ``_SILENT_AMPLITUDE`` read as that floor, so a silent stretch carries a level a
    threshold can still be subtracted from. Accepts one amplitude or a per-element array of them,
    returning the matching type.
    """
    return cast(float | Signal, _DB_PER_DECADE * np.log10(np.maximum(gain, _SILENT_AMPLITUDE)))


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


@dataclass(frozen=True)
class LevelTrend:
    """The straight line through a stretch's level readings, in seconds from that stretch's own first frame.

    Taking the line through every reading lets the body of the material set the slope: a note holding its
    level until a short release at the very end states the shallow fall it spent its length making, and a
    struck note states the steep one. A line also reads at a moment past the readings it was drawn through,
    which is where a ramp fitted to the material has to land.
    """

    mean_level: float
    mean_s: float
    slope: float

    @overload
    def at(self, moment_s: float) -> float: ...
    @overload
    def at(self, moment_s: Signal) -> Signal: ...
    def at(self, moment_s: float | Signal) -> float | Signal:
        """The level the line reads at ``moment_s`` seconds into the stretch it was drawn through."""
        return self.mean_level + self.slope * (moment_s - self.mean_s)


def _block_levels(signal: Signal, block: int) -> tuple[Signal, Signal]:
    """The level of each whole ``block``-frame window of ``signal``, and the frame each one centres on."""
    count = signal.size // block
    windows = np.asarray(signal[: count * block], dtype=np.float64).reshape(count, block)
    return np.sqrt(np.mean(windows**2, axis=1)), (np.arange(count, dtype=np.float64) + 0.5) * block


def level_trend(signal: Signal, sample_rate: int) -> LevelTrend | None:
    """The line ``signal``'s own level readings make, in seconds from its first frame.

    The level is read in short windows so the line follows the material's envelope past the phase of its
    waveform, and every reading is given to the fit, which is what lets the body of a stretch set its slope.

    Returns ``None`` for a stretch holding fewer than ``_MIN_READINGS`` whole windows, which is too little
    for a line to be drawn through.
    """
    block = max(1, round(_LEVEL_WINDOW_S * sample_rate))
    data = np.asarray(signal, dtype=np.float64)
    if data.size < _MIN_READINGS * block:
        return None

    levels, centres = _block_levels(data, block)
    seconds = centres / sample_rate
    mean_s = float(np.mean(seconds))
    centered = seconds - mean_s
    return LevelTrend(
        mean_level=float(np.mean(levels)),
        mean_s=mean_s,
        slope=float(np.sum(centered * levels) / np.sum(centered**2)),
    )
