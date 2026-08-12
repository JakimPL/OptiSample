from __future__ import annotations

from typing import Final

import numpy as np

from optisample.carrier.store import NO_CURVE, StoredCarrier
from optisample.dsp.envelope import Signal
from optisample.optimize.export.envelope import sounding_gain
from trackmod.core.envelopes.envelope import Envelope

HELD_ROUNDS: Final = 4  # rounds a loop is wrapped for, enough for a seam step to become a rhythm rather than a click


def carrier_audition(
    carrier: StoredCarrier,
    envelope: Envelope | None,
    *,
    tempo: int,
    rounds: int = HELD_ROUNDS,
) -> Signal:
    """One stored waveform played the way a tracker plays it: the wrap it makes, under the curve above it.

    This is the pair the whole route is built on, made audible: the waveform holds the timbre at one
    loudness and the envelope brings the level down over it, so what is heard is what a player puts out
    rather than either half on its own. A looped sample is wrapped ``rounds`` times past its stored span,
    which is long enough for a seam step or a level pulse to be heard as a rhythm; a sample stored whole
    plays out as it stands.
    """
    stored = carrier.stored
    if stored.loop is None:
        played = np.asarray(stored.pcm, dtype=np.float64)
    else:
        region = stored.pcm[stored.loop.start : stored.loop.end]
        played = np.concatenate([stored.pcm, np.tile(region, rounds)])

    if envelope is NO_CURVE:
        return played

    gain = sounding_gain(envelope, tempo=tempo, frames=int(played.size), sample_rate=stored.sample_rate)
    return np.asarray(played * gain, dtype=np.float64)
