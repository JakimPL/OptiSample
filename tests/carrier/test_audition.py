from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.carrier.audition import HELD_ROUNDS, carrier_audition
from optisample.carrier.instrument import written_shape
from optisample.carrier.source import CarrierSource
from optisample.carrier.store import CarrierSettings, StoredCarrier, store_carrier
from optisample.dsp.loop import Loop
from optisample.io.tracker.envelope import NO_ENVELOPE
from tests.carrier.conftest import ROOT_PITCH, SR, TEMPO, Sourcer
from trackmod.core.envelopes.envelope import Envelope

Settings = Callable[..., CarrierSettings]

_LOOP = Loop(start=SR // 4, end=SR // 2)


def _stored(one: CarrierSource, settings: CarrierSettings, envelope: Envelope | None = NO_ENVELOPE) -> StoredCarrier:
    return store_carrier(one, envelope, settings=settings, rng=np.random.default_rng(0))


def test_a_looped_waveform_is_wrapped_past_the_span_it_stores(
    source: Sourcer,
    carrier_settings: Settings,
) -> None:
    """A wrap is heard over rounds, so the audition plays the region several times past the stored span."""
    settings = carrier_settings()
    carrier = _stored(source(loop=_LOOP), settings)
    played = carrier_audition(carrier, NO_ENVELOPE, tempo=TEMPO)

    stored = carrier.stored
    region = stored.loop.end - stored.loop.start
    assert played.size == stored.frames + HELD_ROUNDS * region


def test_a_waveform_stored_whole_plays_out_as_it_stands(source: Sourcer, carrier_settings: Settings) -> None:
    """There is no region to wrap, so the audition is the stored span itself."""
    settings = carrier_settings()
    carrier = _stored(source(), settings)
    played = carrier_audition(carrier, NO_ENVELOPE, tempo=TEMPO)

    assert played.size == carrier.stored.frames
    assert played == pytest.approx(carrier.stored.pcm)


def test_the_curve_brings_a_held_note_down_over_its_rounds(source: Sourcer, carrier_settings: Settings) -> None:
    """The pair made audible: the waveform holds one loudness and the envelope states the decline."""
    settings = carrier_settings()
    one = source(decay_db=36.0, loop=_LOOP)
    envelope = written_shape([one], target=settings.target, grid=settings.grid).envelope
    carrier = _stored(one, settings, envelope)
    played = carrier_audition(carrier, envelope, tempo=TEMPO)

    opening = float(np.sqrt(np.mean(played[: played.size // 4] ** 2)))
    closing = float(np.sqrt(np.mean(played[-played.size // 4 :] ** 2)))
    assert closing < opening
