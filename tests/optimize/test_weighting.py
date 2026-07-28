import numpy as np
import pytest

from optisample.metrics.base import Signal
from optisample.optimize.weighting import energy_weight

_FULL_SCALE = 1.0
_THIRTY_DB_DOWN = 10.0 ** (-30.0 / 20.0)  # the pp-to-ff span a piano actually covers


def tone(amplitude: float, frames: int = 400) -> Signal:
    """A constant-amplitude stretch, so its energy is exactly the square of the amplitude."""
    return np.full(frames, amplitude, dtype=np.float64)


def test_full_scale_is_the_unit() -> None:
    """The exponent alone says how steeply a quiet note counts for less, whatever value it takes."""
    for exponent in (0.0, 0.3, 0.5, 1.0):
        assert energy_weight(tone(_FULL_SCALE), exponent) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("exponent", "expected"),
    [
        (0.0, 1.0),  # every note priced alike
        (0.5, 10.0**-1.5),  # amplitude-proportional
        (1.0, 10.0**-3.0),  # energy-proportional
        (0.3, 10.0**-0.9),  # the loudness an ear reports for that energy
    ],
)
def test_the_exponent_sets_how_far_a_quiet_note_falls(exponent: float, expected: float) -> None:
    assert energy_weight(tone(_THIRTY_DB_DOWN), exponent) == pytest.approx(expected)


def test_the_weight_follows_the_energy_of_the_span_it_is_given() -> None:
    """Two spans of the same recording weigh what each of them carries, rather than what the file does."""
    signal = np.concatenate([tone(1.0, 100), tone(0.0, 300)])
    assert energy_weight(signal[:100], 1.0) == pytest.approx(1.0)
    assert energy_weight(signal, 1.0) == pytest.approx(0.25)


def test_a_silent_span_costs_nothing_where_energy_is_priced() -> None:
    assert energy_weight(tone(0.0), 0.5) == 0.0


def test_a_silent_span_still_counts_where_energy_is_left_out() -> None:
    """At exponent zero the objective reads playing time alone, so a level of zero changes nothing."""
    assert energy_weight(tone(0.0), 0.0) == 1.0


def test_an_empty_span_carries_no_energy() -> None:
    assert energy_weight(np.zeros(0, dtype=np.float64), 1.0) == 0.0
