from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.dsp.level import db_to_gain, gain_to_db, peak_amplitude
from optisample.metrics.base import Signal
from optisample.metrics.preprocess import integrated_loudness

_HEARD_LUFS: Final = -20.0  # the loudness one question is met at, so a whole set plays at one volume
_HEARD_PEAK_DB: Final = -1.0  # the ceiling a lifted question is held under, so everything it plays stays clean


@dataclass(frozen=True)
class HeardPair:
    """One question as a listener meets it: three recordings of one length, carrying one lift between them.

    ``gain_db`` is that lift, which the manifest states so the level each encoding delivers
    (:attr:`~optisample.calibrate.ranking.renditions.Rendition.loudness_lufs`) and the level it was heard
    at stay reconcilable.
    """

    reference: Signal
    first: Signal
    second: Signal
    gain_db: float


def _lift_db(reference: Signal, sides: tuple[Signal, ...], sample_rate: int) -> float:
    """How far one question is lifted: to the stated loudness where its peak allows, to the ceiling where it binds."""
    headroom = _HEARD_PEAK_DB - gain_to_db(max(peak_amplitude(one) for one in (reference, *sides)))
    loudness = integrated_loudness(reference, sample_rate)
    if not np.isfinite(loudness):
        return headroom

    return min(_HEARD_LUFS - loudness, headroom)


def heard_pair(reference: Signal, first: Signal, second: Signal, sample_rate: int) -> HeardPair:
    """The three recordings of one question, held to the stretch they share and lifted to one level.

    The stretch is the shortest of the three, which is the recording wherever the material holds a note
    past the end of what was captured. That is where a comparison is possible: the metric reads the pair
    over the same shared stretch (:func:`~optisample.metrics.preprocess.match_length`), so holding the
    audio to it puts the listener and the reading on the same seconds.

    The lift is one gain carried by all three, so the level gap between the sides -- the confound a
    blinded preference is most exposed to, and the one the report reads back -- stands exactly as the
    encodings produced it, while a question the material plays softly is met at the loudness of one it
    leans on. That is what lets a session run at a fixed volume from the first block to the last.
    """
    length = min(reference.size, first.size, second.size)
    held = tuple(np.asarray(one[:length], dtype=np.float64) for one in (reference, first, second))
    gain_db = _lift_db(held[0], held[1:], sample_rate)
    gain = db_to_gain(gain_db)
    return HeardPair(
        reference=held[0] * gain,
        first=held[1] * gain,
        second=held[2] * gain,
        gain_db=gain_db,
    )
