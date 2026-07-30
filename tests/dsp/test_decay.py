from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.decay import LinearDecay, fit_linear_decay
from optisample.dsp.levels import db_to_gain
from optisample.dsp.loop import Loop

SR = 8_000
FREQ = 200.0
FRAMES = 3 * SR
LOOP = Loop(start=SR // 2, end=SR)
_WINDOW = round(0.05 * SR)  # the stretch one level reading covers, which is what a trend is read in
_RING_DB = -12.0  # the fall a struck note makes over the recording, as steep as the level gate admits


def _sine(envelope: NDArray[np.float64] | float, frames: int = FRAMES) -> NDArray[np.float64]:
    return envelope * np.sin(2.0 * np.pi * FREQ * np.arange(frames, dtype=np.float64) / SR)


def _ringing() -> NDArray[np.float64]:
    """A struck note: one pitch under a level falling at the steady rate in decibels a ringing note keeps.

    This is the material the fitted ramp reads a final level off, so a test measuring how close the ramp
    lands measures it on the decline the fit is drawn for.
    """
    seconds = np.arange(FRAMES, dtype=np.float64) / SR
    return _sine(db_to_gain(_RING_DB * seconds * SR / FRAMES))


def _level(signal: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(signal**2)))


# --- the ramp itself --------------------------------------------------------------------------------


def test_the_ramp_holds_the_stored_material_and_falls_away_past_it() -> None:
    decay = LinearDecay(start_s=1.0, end_s=3.0, final_gain=0.25)

    envelope = decay.envelope(4 * SR, SR)

    assert decay.span_s == pytest.approx(2.0)
    assert float(np.min(envelope[: SR + 1])) == 1.0  # unit gain over everything the sample stores
    assert float(envelope[2 * SR]) == pytest.approx(0.625)  # halfway down the ramp
    assert float(envelope[3 * SR]) == pytest.approx(0.25)
    assert float(envelope[-1]) == pytest.approx(0.25)  # held for as long as the note runs on


# --- what the recording says a loop should decline to -------------------------------------------------


def test_a_struck_note_declines_to_the_level_its_recording_ends_on() -> None:
    signal = _ringing()

    decay = fit_linear_decay(signal, SR, LOOP)

    assert decay is not None
    assert decay.start_s == pytest.approx(LOOP.start / SR)
    assert decay.end_s == pytest.approx(FRAMES / SR)
    held = _level(signal[LOOP.start : LOOP.start + _WINDOW])
    ends_on = _level(signal[-_WINDOW:])
    assert decay.final_gain * held == pytest.approx(ends_on, rel=0.1)


def test_the_ramp_starts_where_the_stored_region_stops_following_the_recording() -> None:
    """The region is stored at one level from ``loop.start`` on, so that is where the fall it gave up resumes."""
    signal = _sine(np.linspace(1.0, 0.1, FRAMES))

    early = fit_linear_decay(signal, SR, Loop(start=SR // 2, end=SR))
    late = fit_linear_decay(signal, SR, Loop(start=SR, end=2 * SR))

    assert early is not None and late is not None
    assert early.start_s < late.start_s
    assert early.final_gain < late.final_gain  # the earlier region holds a higher level to fall from


def test_a_note_falling_further_is_played_further_down() -> None:
    gentle = fit_linear_decay(_sine(np.linspace(1.0, 0.6, FRAMES)), SR, LOOP)
    steep = fit_linear_decay(_sine(np.linspace(1.0, 0.1, FRAMES)), SR, LOOP)

    assert gentle is not None and steep is not None
    assert steep.final_gain < gentle.final_gain


def test_a_note_held_to_a_short_release_reports_the_fall_its_length_made() -> None:
    # The body of the material sets the line, so a release at the very end tilts it rather than owning it.
    release = round(0.15 * SR)
    envelope = np.concatenate([np.ones(FRAMES - release), np.linspace(1.0, 0.0, release)])

    decay = fit_linear_decay(_sine(envelope), SR, LOOP)

    assert decay is not None
    assert decay.final_gain > 0.8


@pytest.mark.parametrize(
    ("name", "envelope"),
    [
        pytest.param("steady", np.ones(FRAMES), id="material holding its level asks for no ramp"),
        pytest.param("rising", np.linspace(0.2, 1.0, FRAMES), id="material still growing asks for no ramp"),
    ],
)
def test_material_stating_no_decline_carries_no_decay(name: str, envelope: NDArray[np.float64]) -> None:
    assert fit_linear_decay(_sine(envelope), SR, LOOP) is None


def test_a_loop_reaching_the_end_of_its_recording_declines_through_the_region_itself() -> None:
    """The region is stored flat, so the fall it made from ``loop.start`` on is the ramp that restores it."""
    signal = _ringing()

    decay = fit_linear_decay(signal, SR, Loop(start=SR, end=FRAMES))

    assert decay is not None
    assert decay.start_s == pytest.approx(1.0)
    assert decay.final_gain * _level(signal[SR : SR + _WINDOW]) == pytest.approx(_level(signal[-_WINDOW:]), rel=0.1)


def test_a_region_too_short_for_a_line_through_its_levels_carries_no_decay() -> None:
    """The level the ramp falls from is read off the region, so a region holding one reading states none."""
    signal = _sine(np.linspace(1.0, 0.1, FRAMES))

    assert fit_linear_decay(signal, SR, Loop(start=SR, end=SR + _WINDOW)) is None


def test_a_silent_loop_region_holds_no_level_to_decline_from() -> None:
    signal = np.concatenate([np.zeros(SR), _sine(np.linspace(1.0, 0.1, 2 * SR), frames=2 * SR)])

    assert fit_linear_decay(signal, SR, Loop(start=100, end=SR)) is None
