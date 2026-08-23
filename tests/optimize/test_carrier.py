from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.root import OptiConfig
from optisample.dsp.surrogate import EncodeContext, EncodingParams, render
from optisample.io.tracker.envelope import NO_ENVELOPE, envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.metrics.diagnostics import segmental_snr
from optisample.optimize.carrier import CurveSettings, PlayedCurve, clip_envelope, clip_envelopes, stored_carrier

_RATE = 44_100
_PITCH = 60
_TEMPO = 125
_KEY = SampleKey(_PITCH, 100)
_NO_CURVE = PlayedCurve(NO_ENVELOPE, _TEMPO)


@pytest.fixture(name="curves")
def _curves(config: OptiConfig, target: ExportTarget) -> CurveSettings:
    grid = envelope_grid(target, tempo=_TEMPO, release_s=config.export.envelope.release_s)
    return CurveSettings(config=config.encode, target=target, grid=grid)


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
