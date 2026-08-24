from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.root import OptiConfig
from optisample.dsp.surrogate import EncodeContext, EncodingParams, render
from optisample.io.tracker.envelope import NO_ENVELOPE, envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.metrics.diagnostics import segmental_snr
from optisample.optimize.carrier import (
    CurveSettings,
    PlayedCurve,
    clip_envelope,
    clip_envelopes,
    played_curve,
    stored_carrier,
)

_RATE = 44_100
_PITCH = 60
_TEMPO = 125
_KEY = SampleKey(_PITCH, 100)
_NO_CURVE = PlayedCurve(NO_ENVELOPE, _TEMPO)
_EVERY_ATTACK = 0.0  # the gate admitting every recording, whatever its attack asks of a written curve
_NO_ATTACK = 1_000.0  # a gate no recording clears, which is how a recording is held to its own level


@pytest.fixture(name="curves")
def _curves(config: OptiConfig, target: ExportTarget) -> CurveSettings:
    grid = envelope_grid(target, tempo=_TEMPO, release_s=config.export.envelope.release_s)
    return CurveSettings(config=config.encode, target=target, grid=grid, min_attack_ticks=_EVERY_ATTACK)


@pytest.fixture(name="note")
def _note(piano_note: Callable[..., NDArray[np.float64]]) -> NDArray[np.float64]:
    return piano_note(_PITCH, dur=0.8, seed=11)


def _params(depth: int) -> EncodingParams:
    return EncodingParams(target_rate=_RATE, depth_bits=depth, trim_s=None)


def _curve(note: NDArray[np.float64], curves: CurveSettings) -> PlayedCurve:
    return PlayedCurve(clip_envelope(note, _KEY, _RATE, curves), _TEMPO)


def test_a_recording_that_declines_states_a_curve(note: NDArray[np.float64], curves: CurveSettings) -> None:
    """A struck note falls across itself, which is exactly the level an envelope has to carry."""
    assert clip_envelope(note, _KEY, _RATE, curves)


def test_naming_no_curve_stores_the_signal_as_it_was_played(
    note: NDArray[np.float64], make_encode_ctx: Callable[..., EncodeContext]
) -> None:
    """A run keeping recordings prices exactly what it did before carriers were priced at all."""
    stored = stored_carrier(note, _RATE, _params(16), make_encode_ctx(_PITCH), _NO_CURVE)
    assert stored.level.transparent


def test_a_carrier_hands_its_level_to_the_curve_it_is_played_through(
    note: NDArray[np.float64],
    curves: CurveSettings,
    make_encode_ctx: Callable[..., EncodeContext],
) -> None:
    """The waveform stores timbre and the curve beside it states the decline, so the PCM comes out flat."""
    curve = _curve(note, curves)
    carried = stored_carrier(note, _RATE, _params(16), make_encode_ctx(_PITCH), curve)
    played = stored_carrier(note, _RATE, _params(16), make_encode_ctx(_PITCH), _NO_CURVE)
    assert not carried.level.transparent
    head, tail = np.split(np.abs(carried.pcm), 2)
    flat, fell = np.split(np.abs(played.pcm), 2)
    assert head.max() / tail.max() < flat.max() / fell.max()  # the carrier decays less than the recording


def test_a_shallow_grid_spent_on_timbre_beats_one_spent_on_a_decline(
    note: NDArray[np.float64],
    config: OptiConfig,
    curves: CurveSettings,
    make_encode_ctx: Callable[..., EncodeContext],
) -> None:
    """The whole of why a carrier is worth pricing: at eight bits the level costs the quantizer its range."""
    curve = _curve(note, curves)
    carried = render(stored_carrier(note, _RATE, _params(8), make_encode_ctx(_PITCH, seed=3), curve), _RATE)
    played = render(stored_carrier(note, _RATE, _params(8), make_encode_ctx(_PITCH, seed=3), _NO_CURVE), _RATE)
    frames = min(carried.size, played.size, note.size)
    segmental = config.analysis.metrics.preprocess.segmental
    assert segmental_snr(note[:frames], carried[:frames], segmental) > segmental_snr(
        note[:frames], played[:frames], segmental
    )


def test_every_recording_a_run_holds_states_a_curve_of_its_own(
    note: NDArray[np.float64], curves: CurveSettings
) -> None:
    """One reading per survivor, so every encoding of a recording is priced against the same curve."""
    other = SampleKey(_PITCH + 12, 100)
    envelopes = clip_envelopes({_KEY: note, other: note}, _RATE, curves)
    assert set(envelopes) == {_KEY, other}


def test_a_recording_rising_faster_than_a_tick_states_no_curve(
    note: NDArray[np.float64], curves: CurveSettings
) -> None:
    """A curve turns its corners on ticks, so a level event shorter than one is a level it cannot state.

    Handing it over anyway gives the attack back as a ramp between corners and holds the level ahead of it
    up, which is exactly what a struck sound is recognised by.
    """
    assert clip_envelope(note, _KEY, _RATE, replace(curves, min_attack_ticks=_NO_ATTACK)) is NO_ENVELOPE


def test_a_recording_the_grid_has_room_for_keeps_the_curve_it_states(
    note: NDArray[np.float64], curves: CurveSettings
) -> None:
    """The gate answers for the material a curve cannot follow and leaves everything else where it was."""
    assert clip_envelope(note, _KEY, _RATE, curves)


def test_an_encoding_storing_a_recording_as_played_reaches_for_no_curve(
    note: NDArray[np.float64], curves: CurveSettings
) -> None:
    """Both storages are offered per span, so which one an encoding asks for is what settles its waveform."""
    envelopes = clip_envelopes({_KEY: note}, _RATE, curves)

    assert played_curve(envelopes, _KEY, _params(16), _TEMPO).envelope is NO_ENVELOPE
    assert played_curve(envelopes, _KEY, replace(_params(16), carrier=True), _TEMPO).envelope is envelopes[_KEY]
