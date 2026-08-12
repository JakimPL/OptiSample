from __future__ import annotations

from math import ceil, log
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

from optisample.config.subsonic import SubsonicConfig

Signal = NDArray[np.float64]
Sections = NDArray[np.float64]

_PASSES: Final = 2  # the sections are run forward and back, which doubles the depth one pass reaches
_DECIBELS_PER_DECADE: Final = 10.0  # decibels a power ratio of ten spans, which the rejection is stated in
_POLES_PER_ORDER: Final = 2.0  # the exponent one order adds to the response the depth is solved out of
_SHALLOWEST: Final = 1  # orders the curve holds at the least, which is where a gentle rejection lands
_SETTLE_PERIODS: Final = 3.0  # cycles of the cutoff the padding runs, which is where the roll-off comes to rest
_READABLE: Final = 1  # frames of padding a reading needs, so a recording shorter than that stands as it is


def subsonic_order(config: SubsonicConfig) -> int:
    """How many poles reach ``config.rejection_db`` at ``config.rejection_hz``, run forward and back.

    A maximally flat response stands ``10·log10(1 + (cutoff / f)^(2·order))`` decibels down at ``f`` and
    the second pass doubles that, so the order follows from the two points the config states. Taking the
    shallowest order that reaches the depth keeps the curve as gentle as one arriving there can be, which
    is what leaves the material above the cutoff carrying the shape it was recorded with.
    """
    ratio = config.cutoff_hz / config.rejection_hz
    reach = 10.0 ** (config.rejection_db / (_DECIBELS_PER_DECADE * _PASSES)) - 1.0
    return max(_SHALLOWEST, ceil(log(reach) / (_POLES_PER_ORDER * log(ratio))))


def subsonic_sections(sample_rate: int, config: SubsonicConfig) -> Sections:
    """The curve as second-order sections at ``sample_rate``, the form it stays numerically stable in.

    Raises:
        ValueError: when ``config.cutoff_hz`` stands at or above half of ``sample_rate``, leaving the
            curve no band to pass.
    """
    order = subsonic_order(config)
    return np.asarray(butter(order, config.cutoff_hz, btype="highpass", fs=sample_rate, output="sos"), np.float64)


def remove_subsonic(signal: Signal, sample_rate: int, config: SubsonicConfig) -> Signal:
    """``signal`` carrying the band above ``config.cutoff_hz``, read forward and then back.

    Running the sections both ways leaves the phase of everything they pass exactly as it was captured, so
    a waveform arrives shaped the way it was recorded and what it gives up is the depth beneath hearing --
    which is what keeps a seam, an onset and a decay reading off the audio the way they read off the file.
    Each end is padded for the cycles of the cutoff the roll-off comes to rest in, so the curve meets the
    recording's own material already settled. A recording captured on several channels is read channel by
    channel, each arriving as the one on its own would.

    A recording of a single frame is answered as it stands, a depth taking frames to measure.
    """
    data = np.asarray(signal, dtype=np.float64)
    padding = min(round(_SETTLE_PERIODS * sample_rate / config.cutoff_hz), len(data) - 1)
    if padding < _READABLE:
        return data.copy()

    filtered = sosfiltfilt(subsonic_sections(sample_rate, config), data, axis=0, padlen=padding)
    return np.asarray(filtered, dtype=np.float64)
