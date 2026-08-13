from __future__ import annotations

from enum import Enum

from optisample.music import semitone_ratio


class Clock(Enum):
    """Which timeline a level's seconds are counted on.

    A level is a gain over time, and two levels multiply only where their moments mean the same thing. The
    two clocks a tracker puts a note on are what this tells apart: what a recording did, and what a player
    puts out.

    ``RECORDED`` counts seconds from a recording's own onset at the rate it was captured at, which is where
    :func:`~optisample.dsp.envelope.decompose` and
    :func:`~optisample.dsp.level.readings.level_readings` read a level. ``PLAYED`` counts seconds of output
    for a note sounding at some key, which is where a stored sample's own decline and a written volume
    envelope both land.

    A written envelope is natively ``PLAYED``: a tracker walks its breakpoints on the tick clock whatever
    key is struck, so the curve holds its shape while the material under it stretches. That asymmetry is
    what :func:`played_scale` states, and holding it in the type is what keeps a recording's level from
    being multiplied onto a played note as though the two ran at one speed.
    """

    RECORDED = "recorded"
    PLAYED = "played"


def played_scale(*, root_pitch: int, pitch: int) -> float:
    """How long one recorded second lasts once the material is sounded at ``pitch``.

    A tracker repitches by resampling, so a recording played above the key it was captured at runs
    proportionally faster and each of its seconds occupies less of the output. Scaling a level's moments by
    this is what carries it from :attr:`Clock.RECORDED` onto :attr:`Clock.PLAYED`.
    """
    return semitone_ratio(root_pitch - pitch)
