from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from optisample.dsp.decay import LinearDecay
from optisample.dsp.levels import gain_to_db
from trackmod.core.envelopes.curve import Breakpoint, timed_envelope
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.envelopes.span import EnvelopeSpan
from trackmod.limits.bound import Bound
from trackmod.spec.levels import MAX_VOLUME, MIN_VOLUME

NO_ENVELOPE: Final = None  # what an instrument whose samples carry every level they play at leaves behind

_ONSET_S: Final = 0.0
_NO_DISPERSION: Final = 0.0  # what one shape costs the samples sharing it while none of them states a decline


def _node_value(gain: float) -> int:
    """The step a volume envelope holds ``gain`` at, on the 0-64 grid both formats write their nodes on."""
    return round(MAX_VOLUME * gain)


def _decline_db_per_s(decay: LinearDecay) -> float:
    """How steeply a note is played down, in decibels a second, which is what one shared shape has to match."""
    return gain_to_db(decay.final_gain) / decay.span_s


def shared_decay(decays: Sequence[LinearDecay | None]) -> LinearDecay | None:
    """The one decline every voice an instrument starts is played down by.

    A volume envelope belongs to the instrument, so the samples written into one slot answer to a single
    shape however differently each was recorded to decline. The member falling at the median rate is taken
    whole, which keeps the written curve a decline some recording actually makes rather than an average of
    several that no key plays. :func:`decay_dispersion` states what the others give up for it.

    Returns ``None`` where no sample in the slot states a decline, which leaves that instrument's voices at
    the level their own material carries.
    """
    stated = [decay for decay in decays if decay is not None]
    if not stated:
        return NO_ENVELOPE

    return sorted(stated, key=_decline_db_per_s)[len(stated) // 2]


def decay_dispersion(decays: Sequence[LinearDecay | None], shared: LinearDecay) -> float:
    """The widest decibel gap ``shared`` leaves a sample of the same instrument by the end of its own ramp.

    Each member states where its note lands; the shared curve puts it somewhere else by the time that
    member's ramp is through. The largest of those gaps is what one envelope per instrument costs, and it is
    the reading that says whether the keys sharing one are better written as several instruments.
    """
    rate = _decline_db_per_s(shared)
    stated = [decay for decay in decays if decay is not None]
    return max(
        (abs(gain_to_db(decay.final_gain) - rate * decay.span_s) for decay in stated),
        default=_NO_DISPERSION,
    )


def volume_envelope(
    decay: LinearDecay,
    *,
    tempo: int,
    release_s: float,
    tick_bound: Bound,
    value_bound: Bound,
) -> Envelope:
    """The curve an instrument plays its voices down by, written for a module running at ``tempo``.

    Four breakpoints say the whole of a :class:`~optisample.dsp.decay.LinearDecay`: full volume at the
    onset, still full where the stored material stops following the recording, the level the note has
    fallen to once the ramp is through, and silence a release later. The third is the sustain point, so a
    held note stays at the level the recording reached and a released one goes on to the fourth and dies.
    Both formats sustain on a single point, so the span names one.

    The release is spelled as a breakpoint rather than as an instrument fadeout because the two formats
    begin a fade in different places -- FastTracker 2 at the key off, Impulse Tracker where the volume
    envelope ends -- while a curve reaching zero says the same thing to both
    (:mod:`trackmod.core.instruments.fade`).

    Ticks are what a format counts envelope time in, so the curve holds only for the tempo it was written
    for -- which is why that tempo travels with a bank
    (:class:`~optisample.artifacts.bank.BankDocument`).
    """
    breakpoints = (
        Breakpoint(seconds=_ONSET_S, value=MAX_VOLUME),
        Breakpoint(seconds=decay.start_s, value=MAX_VOLUME),
        Breakpoint(seconds=decay.end_s, value=_node_value(decay.final_gain)),
        Breakpoint(seconds=decay.end_s + release_s, value=MIN_VOLUME),
    )
    held = len(breakpoints) - 2
    return timed_envelope(
        breakpoints,
        tempo=tempo,
        tick_bound=tick_bound,
        value_bound=value_bound,
        sustain=EnvelopeSpan(begin=held, end=held),
    )
