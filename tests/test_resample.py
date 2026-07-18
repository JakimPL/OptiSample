from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.resample import resample_num, resample_to, resampled_frame_count

SR = 44_100


def sine(freq: float, dur: float = 0.5, sr: int = SR) -> NDArray[np.float64]:
    t = np.arange(int(dur * sr), dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * freq * t)


def rms(signal: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0


def test_resample_to_identity_when_rate_unchanged() -> None:
    signal = sine(1000.0)
    assert np.array_equal(resample_to(signal, SR, SR), signal)


def test_resample_num_identity_and_empty_cases() -> None:
    signal = sine(1000.0)
    assert np.array_equal(resample_num(signal, signal.size), signal)
    assert resample_num(signal, 0).size == 0
    assert resample_num(np.zeros(0, dtype=np.float64), 100).size == 100  # empty source -> zeros


def test_resample_num_hits_exact_length() -> None:
    assert resample_num(sine(1000.0), 12_345).size == 12_345


def test_downsample_preserves_in_band_tone() -> None:
    down = resample_to(sine(500.0), SR, 8_000)  # 500 Hz well under the 4 kHz Nyquist
    assert down.size == pytest.approx(int(0.5 * 8_000), abs=2)
    assert rms(down) == pytest.approx(np.sqrt(0.5), abs=0.05)


def test_downsample_removes_above_new_nyquist() -> None:
    down = resample_to(sine(6000.0), SR, 8_000)  # 6 kHz is above the 4 kHz Nyquist -> gone
    assert rms(down) < 0.05


def test_round_trip_recovers_in_band_signal() -> None:
    signal = sine(500.0)
    round_trip = resample_to(resample_to(signal, SR, 8_000), 8_000, SR)
    length = min(signal.size, round_trip.size)
    # Compare the steady middle section (FFT resampling rings slightly at the edges).
    lo, hi = length // 5, 4 * length // 5
    error = signal[lo:hi] - round_trip[lo:hi]
    assert rms(error) < 0.1 * rms(signal[lo:hi])


def test_resampled_frame_count_matches_actual() -> None:
    signal = sine(1000.0)
    for target in (8_000, 22_050, 32_000):
        assert resample_to(signal, SR, target).size == resampled_frame_count(signal.size, SR, target)


def test_invalid_rates_raise() -> None:
    with pytest.raises(ValueError):
        resample_to(sine(1000.0), 0, 8_000)
    with pytest.raises(ValueError):
        resampled_frame_count(100, 44_100, -1)
