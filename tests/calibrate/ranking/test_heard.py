from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.calibrate.ranking import heard_pair
from optisample.dsp.levels import gain_to_db, peak_amplitude
from optisample.metrics.preprocess import integrated_loudness

_RATE = 44_100
_SECONDS = 1.0
_TARGET_LUFS = -20.0
_CEILING_DB = -1.0
_QUIET_DB = -45.0
_SIDE_GAP_DB = 6.0


def _tone(seconds: float, level_db: float) -> NDArray[np.float64]:
    """A steady tone at a stated level, which is the simplest thing a loudness meter reads."""
    moments = np.arange(int(seconds * _RATE), dtype=np.float64) / _RATE
    return np.sin(2.0 * np.pi * 440.0 * moments) * 10.0 ** (level_db / 20.0)


@pytest.fixture
def quiet_question() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """A question the material plays too softly to hear, one side a stated gap under the other."""
    return (
        _tone(_SECONDS, _QUIET_DB),
        _tone(_SECONDS, _QUIET_DB),
        _tone(_SECONDS, _QUIET_DB - _SIDE_GAP_DB),
    )


def test_a_question_the_material_plays_softly_is_met_at_the_stated_loudness(
    quiet_question: tuple[NDArray[np.float64], ...],
) -> None:
    heard = heard_pair(*quiet_question, _RATE)

    assert integrated_loudness(heard.reference, _RATE) == pytest.approx(_TARGET_LUFS, abs=0.1)


def test_the_gap_between_the_sides_stands_as_the_encodings_produced_it(
    quiet_question: tuple[NDArray[np.float64], ...],
) -> None:
    heard = heard_pair(*quiet_question, _RATE)

    apart = integrated_loudness(heard.first, _RATE) - integrated_loudness(heard.second, _RATE)
    assert apart == pytest.approx(_SIDE_GAP_DB, abs=0.01)


def test_the_lift_a_question_carries_is_the_one_it_states(quiet_question: tuple[NDArray[np.float64], ...]) -> None:
    reference, first, second = quiet_question
    heard = heard_pair(reference, first, second, _RATE)

    assert heard.gain_db == pytest.approx(_TARGET_LUFS - integrated_loudness(reference, _RATE), abs=0.1)


def test_a_question_already_loud_enough_is_held_under_the_ceiling() -> None:
    loud = _tone(_SECONDS, -3.0)

    heard = heard_pair(loud, loud, loud, _RATE)

    assert gain_to_db(peak_amplitude(heard.reference)) <= _CEILING_DB + 0.01


def test_a_peak_the_lift_would_drive_past_the_ceiling_settles_at_it() -> None:
    reference = np.concatenate([_tone(0.01, 0.0), _tone(_SECONDS, _QUIET_DB)])

    heard = heard_pair(reference, reference, reference, _RATE)

    assert gain_to_db(peak_amplitude(heard.reference)) == pytest.approx(_CEILING_DB, abs=0.01)
    assert integrated_loudness(heard.reference, _RATE) < _TARGET_LUFS


def test_a_question_holding_silence_is_left_where_it_stands() -> None:
    silence = np.zeros(int(_SECONDS * _RATE), dtype=np.float64)

    heard = heard_pair(silence, silence, silence, _RATE)

    assert peak_amplitude(heard.reference) == pytest.approx(0.0)


def test_the_three_recordings_run_the_stretch_they_share() -> None:
    reference = _tone(0.5, _QUIET_DB)
    side = _tone(_SECONDS, _QUIET_DB)

    heard = heard_pair(reference, side, side, _RATE)

    assert {heard.reference.size, heard.first.size, heard.second.size} == {reference.size}


def test_the_stretch_a_question_runs_is_the_shortest_of_its_three(
    quiet_question: tuple[NDArray[np.float64], ...],
) -> None:
    reference, first, _ = quiet_question
    short = _tone(0.25, _QUIET_DB)

    heard = heard_pair(reference, first, short, _RATE)

    assert heard.reference.size == short.size
