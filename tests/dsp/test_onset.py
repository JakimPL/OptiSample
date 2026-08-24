from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.onset import ATTACK_READING, attack_seconds, crest_db, onset_frame, onset_window

SR = 48_000
_FREQ = 400.0
_SECOND = SR
_SILENT_S = 0.05  # the run of silence a delayed take opens with, longer than the window reaches back
_STRUCK_S = 0.001  # an attack no written curve has room to state
_BOWED_S = 0.15  # an attack a curve states with room to spare
_TOLERANCE_S = 0.004  # what the smoothing and the tenth-of-peak threshold leave between asked and read
_RISE_SHARE = 0.8  # of a straight rise the tenth-to-nine-tenths reading spans
_LOUDER_DB = 3.0  # how much more crest a spike has to show than a steady tone for the reading to mean anything


def _tone(frames: int) -> NDArray[np.float64]:
    return np.sin(2.0 * np.pi * _FREQ * np.arange(frames, dtype=np.float64) / SR)


def _struck(attack_s: float, frames: int = _SECOND) -> NDArray[np.float64]:
    """A tone rising over ``attack_s`` and holding, which is the shape an attack reading is asked about."""
    rise = max(1, round(attack_s * SR))
    envelope = np.concatenate([np.linspace(0.0, 1.0, rise), np.ones(max(0, frames - rise))])
    return _tone(frames) * envelope[:frames]


def _delayed(signal: NDArray[np.float64], seconds: float) -> NDArray[np.float64]:
    return np.concatenate([np.zeros(round(seconds * SR)), signal])


@pytest.mark.parametrize("attack_s", [_STRUCK_S, 0.02, 0.05])
def test_an_attack_reads_as_long_as_the_rise_it_was_given(attack_s: float) -> None:
    """The reading is what the gate is built on, so it answers the rise the material actually makes.

    A straight rise is timed from a tenth of its peak to nine tenths of it, so what comes back is that
    much of the stretch the material took to get there.
    """
    assert attack_seconds(_struck(attack_s), SR, ATTACK_READING) == pytest.approx(
        _RISE_SHARE * attack_s, abs=_TOLERANCE_S
    )


def test_an_attack_longer_than_the_window_reads_as_the_window_it_was_read_over() -> None:
    """The reading is bounded by the stretch it is taken over, which is what the gate asks of it.

    A curve is refused where an attack runs shorter than a tick, so what the reading has to separate is the
    brief from the unhurried; material rising for longer than the window reads as the window and clears
    every gate that could be asked for.
    """
    window_s = ATTACK_READING.pre_s + ATTACK_READING.span_s

    assert attack_seconds(_struck(_BOWED_S), SR, ATTACK_READING) == pytest.approx(window_s, abs=_TOLERANCE_S * 3)


def test_a_take_is_placed_where_its_own_material_begins() -> None:
    """Each signal states its own onset, so silence ahead of a take moves the window rather than the reading."""
    struck = _struck(_BOWED_S)

    assert onset_frame(_delayed(struck, _SILENT_S), SR, ATTACK_READING) - onset_frame(
        struck, SR, ATTACK_READING
    ) == pytest.approx(_SILENT_S * SR, abs=_TOLERANCE_S * SR)


def test_a_delayed_take_reads_the_same_attack_as_the_take_it_was_delayed_from() -> None:
    """What the reading is for is comparing shapes, so a codec's own delay leaves it where it was."""
    struck = _struck(_BOWED_S)

    assert attack_seconds(_delayed(struck, _SILENT_S), SR, ATTACK_READING) == pytest.approx(
        attack_seconds(struck, SR, ATTACK_READING), abs=_TOLERANCE_S
    )


def test_the_window_spans_the_stretch_either_side_of_the_onset_it_names() -> None:
    """Every reading is taken over this stretch, so its length is what the config asked for."""
    window = onset_window(_struck(_BOWED_S), SR, ATTACK_READING)

    assert window.size == round((ATTACK_READING.pre_s + ATTACK_READING.span_s) * SR)


def test_a_spike_shows_more_crest_than_the_tone_it_stands_in() -> None:
    """A transient is a peak a frame-averaged spectrum spreads out, which is what the crest states."""
    steady = _tone(_SECOND)
    spiked = np.array(steady)
    spiked[: round(_STRUCK_S * SR)] = 0.0
    spiked[round(_STRUCK_S * SR)] = 4.0

    assert crest_db(spiked, SR, ATTACK_READING) > crest_db(steady, SR, ATTACK_READING) + _LOUDER_DB


def test_silence_states_no_attack_at_all() -> None:
    """A recording carrying nothing has no onset to place, and answers rather than dividing by its own peak."""
    assert attack_seconds(np.zeros(_SECOND), SR, ATTACK_READING) == 0.0
    assert onset_frame(np.zeros(_SECOND), SR, ATTACK_READING) == 0
