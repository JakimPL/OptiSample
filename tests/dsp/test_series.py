from __future__ import annotations

import numpy as np
import pytest

from optisample.dsp.series import autocorrelation, hann_kernel, refined_lag, weighted_mean

_PERIOD = 40  # points one round of the repeating series below spans


@pytest.mark.parametrize("span", [0, 1, 4, 9, 64])
def test_a_weighting_carries_one_unit_of_weight_read_from_both_sides(span: int) -> None:
    """A unit sum leaves a mean on the scale of what it averages, and an odd count centres the reading."""
    kernel = hann_kernel(span)

    assert kernel.size % 2 == 1
    assert float(np.sum(kernel)) == pytest.approx(1.0)
    assert np.allclose(kernel, kernel[::-1])


def test_a_series_holding_one_level_reads_as_that_level_at_both_ends() -> None:
    """Dividing by the weight that lands inside the series is what holds its ends where the material is."""
    values = np.full(64, 3.0)

    assert np.allclose(weighted_mean(values, hann_kernel(9)), 3.0)


def test_a_weighted_mean_follows_the_movement_the_series_makes() -> None:
    """A step is answered somewhere between the two levels, so the curve carries the material's own shape."""
    values = np.concatenate([np.zeros(64), np.ones(64)])

    smoothed = weighted_mean(values, hann_kernel(17))

    assert smoothed[0] == pytest.approx(0.0, abs=1e-9)
    assert smoothed[-1] == pytest.approx(1.0, abs=1e-9)
    assert 0.4 < smoothed[64] < 0.6


def test_autocorrelation_of_silence_is_zero() -> None:
    assert np.array_equal(autocorrelation(np.zeros(64)), np.zeros(64))


def test_autocorrelation_peaks_at_the_lag_the_series_repeats_on() -> None:
    series = np.sin(2.0 * np.pi * np.arange(400, dtype=np.float64) / _PERIOD)

    correlation = autocorrelation(series)

    assert correlation[0] == pytest.approx(1.0)
    assert int(np.argmax(correlation[10:100])) + 10 == pytest.approx(_PERIOD, abs=1)


def test_a_series_holding_one_value_correlates_with_nothing() -> None:
    """Centring leaves a flat series with no movement to match, which is material a period has no purchase on."""
    assert np.array_equal(autocorrelation(np.full(32, 5.0)), np.zeros(32))


def test_a_peak_between_two_points_is_read_between_them() -> None:
    correlation = np.array([0.0, 0.5, 1.0, 0.8, 0.0])

    assert refined_lag(correlation, 2) == pytest.approx(2.2143, abs=1e-3)


@pytest.mark.parametrize(
    ("correlation", "lag"),
    [
        pytest.param(np.array([0.2, 0.9, 0.4]), 0, id="the first lag searched has no neighbour before it"),
        pytest.param(np.array([0.2, 0.9, 0.4]), 2, id="the last has none after it"),
        pytest.param(np.ones(5), 2, id="a flat top makes no parabola to place a lag on"),
    ],
)
def test_a_peak_with_no_parabola_through_it_is_read_at_its_own_point(correlation: np.ndarray, lag: int) -> None:
    assert refined_lag(correlation, lag) == float(lag)
