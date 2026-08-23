from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

from optisample.carrier.shape import written_shape
from optisample.carrier.source import CarrierSource
from optisample.config.codec import EncodeConfig
from optisample.dsp.envelope import decompose, level_reading
from optisample.dsp.surrogate import (
    NO_LOOPS,
    UNLOOPED,
    EncodeContext,
    EncodingParams,
    Signal,
    StoredSample,
    encode,
)
from optisample.io.tracker.envelope import NO_ENVELOPE, EnvelopeGrid, carried_signal, sounding_level
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.music import midi_to_freq
from trackmod.core.envelopes.envelope import Envelope

Envelopes = Mapping[SampleKey, Envelope | None]
NO_ENVELOPES: Final[Envelopes] = {}  # what a run storing recordings prices against, read-only by its type

_ALONE: Final = 1.0  # the say one recording carries in the curve it alone states


@dataclass(frozen=True)
class CurveSettings:
    """What reading the curve one recording states is carried out with.

    ``config`` states the weighting a level is read under, ``target`` how many corners the format leaves a
    shape and ``grid`` the ticks and steps those corners land on -- the three that settle what a written
    curve can say, so a clip is priced against a curve of exactly the resolution a module carries.
    """

    config: EncodeConfig
    target: ExportTarget
    grid: EnvelopeGrid


def clip_envelope(signal: Signal, key: SampleKey, sample_rate: int, settings: CurveSettings) -> Envelope | None:
    """The volume curve one recording alone states, written on the grid a module carries it on.

    A written instrument gives one curve to every sample it holds, so what a plan finally plays a key down
    by is the curve its whole slot agreed on. That is settled after the zones are, which leaves the sweep
    the curve each clip states on its own -- the closest a per-clip price can come to what the module
    writes, and what makes the gap between the two exactly the cost of sharing
    (:attr:`~optisample.carrier.shape.WrittenShape.dispersion_db`).
    """
    return written_shape(
        [
            CarrierSource(
                key=key,
                root_pitch=key.pitch,
                sample_rate=sample_rate,
                decomposition=decompose(
                    signal, level_reading(sample_rate, settings.config.envelope, midi_to_freq(key.pitch))
                ),
                loops=NO_LOOPS,
                loop_index=UNLOOPED,
                weight=_ALONE,
            )
        ],
        target=settings.target,
        grid=settings.grid,
    ).envelope


def clip_envelopes(audio: Mapping[SampleKey, Signal], sample_rate: int, settings: CurveSettings) -> Envelopes:
    """The curve each surviving recording states on its own, read once so every encoding of it prices alike."""
    return {key: clip_envelope(signal, key, sample_rate, settings) for key, signal in audio.items()}


@dataclass(frozen=True)
class PlayedCurve:
    """The curve a stored sample hands its level to, and the clock that curve's corners are counted in.

    The two travel together because neither states a level without the other: a tracker writes envelope
    corners in ticks, so what a curve plays a note down by at any moment is settled by the tempo it is
    read at.
    """

    envelope: Envelope | None
    tempo: int


def stored_carrier(
    signal: Signal,
    sample_rate: int,
    params: EncodingParams,
    context: EncodeContext,
    curve: PlayedCurve,
) -> StoredSample:
    """``signal`` stored the way a module carrying ``curve`` keeps it, and played back through it.

    The waveform reaches the encoder already divided by what the curve plays it down by
    (:func:`~optisample.io.tracker.envelope.carried_signal`), so the span it quantizes is level-flat and
    spends its whole grid on timbre; the same curve then rides on the stored sample
    (:func:`~optisample.io.tracker.envelope.sounding_level`), so the renderer puts out what a player puts
    out and the score reads the pair rather than the flattened waveform alone.

    A caller naming no curve stores the signal as it was played, its own PCM holding every level it sounds
    at -- which is what a run keeping recordings rather than carriers asks for, and what a recording the
    curve could not be read from falls back to.
    """
    stored = encode(
        carried_signal(signal, curve.envelope, tempo=curve.tempo, sample_rate=sample_rate),
        sample_rate,
        params,
        context,
    )
    if curve.envelope is NO_ENVELOPE:
        return stored

    return replace(stored, level=sounding_level(curve.envelope, tempo=curve.tempo))
