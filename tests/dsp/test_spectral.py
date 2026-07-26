from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dsp import MelParams, SpectralConfig, StftParams
from optisample.dsp.spectral import (
    band_energy,
    bandlimit,
    content_edge_hz,
    mel_filterbank,
    spectral_centroid,
    spectral_flux,
    spectral_rolloff,
    stft_magnitude,
)

SR = 8_000
_BAND_HZ = 100.0  # the smoothing width the content-edge readings are stated to


def tone(freq: float, dur: float = 0.5, amp: float = 1.0) -> NDArray[np.float64]:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return amp * np.sin(2.0 * np.pi * freq * t)


def test_stft_magnitude_shape() -> None:
    magnitude = stft_magnitude(tone(1000), StftParams(n_fft=256, hop_length=64))
    assert magnitude.shape[1] == 256 // 2 + 1
    assert magnitude.shape[0] >= 1


def test_mel_filterbank_shape_and_nonnegative() -> None:
    filters = mel_filterbank(SR, MelParams(n_fft=512, hop_length=256, n_mels=20, fmin=0.0, fmax=None))
    assert filters.shape == (20, 512 // 2 + 1)
    assert np.all(filters >= 0.0)
    assert np.all(filters.sum(axis=1) > 0.0)  # no degenerate empty filters


def test_centroid_tracks_tone_frequency(spectral_config: SpectralConfig) -> None:
    assert spectral_centroid(tone(1500), SR, spectral_config.stft) == pytest.approx(1500.0, abs=200.0)


def test_centroid_higher_for_brighter_tone(spectral_config: SpectralConfig) -> None:
    stft = spectral_config.stft
    assert spectral_centroid(tone(2500), SR, stft) > spectral_centroid(tone(600), SR, stft)


def test_rolloff_near_tone_frequency(spectral_config: SpectralConfig) -> None:
    rolloff = spectral_rolloff(tone(1200), SR, spectral_config.stft, spectral_config.rolloff_percent)
    assert rolloff == pytest.approx(1200.0, abs=250.0)


def test_band_energy_is_localized() -> None:
    signal = tone(1000)
    assert band_energy(signal, SR, 800, 1200) > 50.0 * band_energy(signal, SR, 1800, 3000)


def test_content_edge_lands_on_the_highest_frequency_a_signal_carries() -> None:
    assert content_edge_hz(tone(1200), SR, 80.0, _BAND_HZ) == pytest.approx(1200.0, abs=_BAND_HZ)


def test_content_edge_reaches_a_partial_far_below_the_fundamental_in_energy() -> None:
    """The reading a share-of-energy rolloff misses: a quiet partial is still content up there."""
    mixed = tone(500) + 0.01 * tone(3000)
    assert content_edge_hz(mixed, SR, 80.0, _BAND_HZ) == pytest.approx(3000.0, abs=_BAND_HZ)


def test_a_shallower_floor_stops_short_of_the_quiet_partial() -> None:
    mixed = tone(500) + 0.01 * tone(3000)  # the partial sits 40 dB down in power
    assert content_edge_hz(mixed, SR, 20.0, _BAND_HZ) < 3000.0


def test_content_edge_of_silence_is_zero() -> None:
    assert content_edge_hz(np.zeros(1024, dtype=np.float64), SR, 80.0, _BAND_HZ) == 0.0


def test_a_band_wider_than_the_spectrum_reads_the_whole_signal_at_once() -> None:
    edge = content_edge_hz(tone(1200), SR, 80.0, float(SR))
    assert edge == pytest.approx(SR / 4.0, rel=0.01)  # the midpoint of the one band there is


def test_bandlimit_removes_out_of_band_tone() -> None:
    signal = tone(1000)
    removed = bandlimit(signal, SR, 1500, 3000)  # 1000 Hz excluded
    assert float(np.sum(removed**2)) < 1e-3 * float(np.sum(signal**2))


def test_flux_variance_higher_for_evolving_signal(spectral_config: SpectralConfig) -> None:
    stft = spectral_config.stft
    static = tone(1000, dur=1.0)
    t = np.arange(static.size, dtype=np.float64) / SR
    evolving = (1.0 + 0.8 * np.sin(2.0 * np.pi * 3.0 * t)) * np.sin(2.0 * np.pi * 1000.0 * t)
    assert float(np.std(spectral_flux(evolving, stft))) > float(np.std(spectral_flux(static, stft)))
