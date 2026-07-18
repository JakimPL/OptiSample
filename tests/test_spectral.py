from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.spectral import (
    MelParams,
    StftParams,
    band_energy,
    bandlimit,
    mel_filterbank,
    spectral_centroid,
    spectral_flux,
    spectral_rolloff,
    stft_magnitude,
)

SR = 8_000


def tone(freq: float, dur: float = 0.5, amp: float = 1.0) -> NDArray[np.float64]:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def test_stft_magnitude_shape() -> None:
    magnitude = stft_magnitude(tone(1000), StftParams(n_fft=256, hop_length=64))
    assert magnitude.shape[1] == 256 // 2 + 1
    assert magnitude.shape[0] >= 1


def test_mel_filterbank_shape_and_nonnegative() -> None:
    filters = mel_filterbank(SR, MelParams(n_fft=512, n_mels=20))
    assert filters.shape == (20, 512 // 2 + 1)
    assert np.all(filters >= 0.0)
    assert np.all(filters.sum(axis=1) > 0.0)  # no degenerate empty filters


def test_centroid_tracks_tone_frequency() -> None:
    assert spectral_centroid(tone(1500), SR) == pytest.approx(1500.0, abs=200.0)


def test_centroid_higher_for_brighter_tone() -> None:
    assert spectral_centroid(tone(2500), SR) > spectral_centroid(tone(600), SR)


def test_rolloff_near_tone_frequency() -> None:
    assert spectral_rolloff(tone(1200), SR) == pytest.approx(1200.0, abs=250.0)


def test_band_energy_is_localized() -> None:
    signal = tone(1000)
    assert band_energy(signal, SR, 800, 1200) > 50.0 * band_energy(signal, SR, 1800, 3000)


def test_bandlimit_removes_out_of_band_tone() -> None:
    signal = tone(1000)
    removed = bandlimit(signal, SR, 1500, 3000)  # 1000 Hz excluded
    assert float(np.sum(removed**2)) < 1e-3 * float(np.sum(signal**2))


def test_flux_variance_higher_for_evolving_signal() -> None:
    static = tone(1000, dur=1.0)
    t = np.arange(static.size, dtype=np.float64) / SR
    evolving = (1.0 + 0.8 * np.sin(2.0 * np.pi * 3.0 * t)) * np.sin(2.0 * np.pi * 1000.0 * t)
    assert float(np.std(spectral_flux(evolving))) > float(np.std(spectral_flux(static)))
