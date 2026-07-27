from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.levels import db_to_gain, gain_to_db, peak_amplitude

_DOUBLE_DB = 6.020599913279624  # the level change an exactly doubled amplitude reads as
_SILENT_LEVEL_DB = -240.0  # what the floor under silence puts a zero amplitude at


# --- decibels and amplitude ---------------------------------------------------------------------------


def test_no_level_change_leaves_the_amplitude_alone() -> None:
    assert db_to_gain(0.0) == 1.0


def test_a_doubled_amplitude_is_six_decibels() -> None:
    assert db_to_gain(_DOUBLE_DB) == pytest.approx(2.0)


def test_a_delta_per_element_answers_per_element() -> None:
    """The velocity->volume map converts a whole curve at once, so the array form has to hold."""
    deltas = np.array([0.0, _DOUBLE_DB, -_DOUBLE_DB], dtype=np.float64)
    assert np.allclose(db_to_gain(deltas), [1.0, 2.0, 0.5])


def test_reading_a_level_back_returns_the_amplitude_it_came_from() -> None:
    amplitudes = np.array([1.0, 0.5, 0.25, 0.001], dtype=np.float64)
    assert np.allclose(db_to_gain(gain_to_db(amplitudes)), amplitudes)


def test_silence_reads_as_the_floor_rather_than_no_level_at_all() -> None:
    """A compressor subtracts a threshold from every level it reads, so a silent frame needs a number."""
    assert gain_to_db(np.zeros(4, dtype=np.float64)) == pytest.approx(_SILENT_LEVEL_DB)


# --- what a signal peaks at ---------------------------------------------------------------------------


def test_a_signal_peaks_at_its_largest_swing_either_way(sine: Callable[..., NDArray[np.float64]]) -> None:
    assert peak_amplitude(-0.4 * sine(440.0)) == pytest.approx(0.4)


def test_a_signal_holding_nothing_peaks_at_nothing() -> None:
    assert peak_amplitude(np.zeros(0, dtype=np.float64)) == 0.0
