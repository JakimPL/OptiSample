import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.spectral import MelParams, SpectralConfig, StftParams
from optisample.dsp.spectral import (
    band_energy,
    band_masks,
    bandlimit,
    content_edge_hz,
    mel_filterbank,
    spectral_centroid,
    spectral_flux,
    spectral_rolloff,
    split_bands,
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


# --- band splitting ------------------------------------------------------------------------------

_CROSSOVERS = (250.0, 1_000.0)
_CROSSOVER_OCTAVES = 0.5


def test_the_masks_of_a_split_sum_to_one_at_every_bin() -> None:
    """Summing to one is what lets a caller weigh the bands apart and still hold the whole material."""
    masks = band_masks(SR, 2_048, _CROSSOVERS, _CROSSOVER_OCTAVES)

    assert masks.shape == (len(_CROSSOVERS) + 1, 2_048 // 2 + 1)
    assert np.all(masks >= 0.0)
    assert np.allclose(masks.sum(axis=0), 1.0)


def test_a_split_signal_adds_back_up_to_itself() -> None:
    signal = tone(300) + tone(1_800, amp=0.4)

    bands = split_bands(signal, SR, _CROSSOVERS, _CROSSOVER_OCTAVES)

    assert np.allclose(bands.sum(axis=0), signal, atol=1e-12)


def test_a_tone_lands_in_the_band_its_frequency_names() -> None:
    bands = split_bands(tone(1_800), SR, _CROSSOVERS, _CROSSOVER_OCTAVES)
    energies = np.sum(bands**2, axis=1)

    assert int(np.argmax(energies)) == 2  # the band above the 1 kHz crossover
    assert float(energies[0] + energies[1]) < 1e-3 * float(energies[2])


def test_a_tone_sitting_on_a_crossover_is_shared_by_the_bands_it_divides() -> None:
    """Splitting the amplitude hands the two sides their share, so together they carry the tone whole."""
    bands = split_bands(tone(1_000), SR, _CROSSOVERS, _CROSSOVER_OCTAVES)
    energies = np.sum(bands**2, axis=1)

    assert float(energies[1]) == pytest.approx(float(energies[2]), rel=0.05)


def test_a_stretch_too_short_to_read_a_crossover_apart_weighs_its_bands_together() -> None:
    """A climb narrower than a few bins separates nothing, so the reading that stretch supports is one band."""
    coarse = band_masks(SR, 32, _CROSSOVERS, _CROSSOVER_OCTAVES)  # 250 Hz per bin
    fine = band_masks(SR, 4_096, _CROSSOVERS, _CROSSOVER_OCTAVES)

    assert coarse.shape[0] == 1
    assert fine.shape[0] == len(_CROSSOVERS) + 1


def test_a_crossover_reaching_past_nyquist_leaves_the_band_below_it_open() -> None:
    masks = band_masks(SR, 4_096, (1_000.0, 3_800.0), _CROSSOVER_OCTAVES)

    assert masks.shape[0] == 2  # 3.8 kHz climbs past the 4 kHz Nyquist, so it names no band of its own


def test_listing_no_crossover_weighs_the_whole_spectrum_as_one_band() -> None:
    signal = tone(300) + tone(1_800, amp=0.4)

    bands = split_bands(signal, SR, (), _CROSSOVER_OCTAVES)

    assert bands.shape[0] == 1
    assert np.allclose(bands[0], signal, atol=1e-12)


def test_flux_variance_higher_for_evolving_signal(spectral_config: SpectralConfig) -> None:
    stft = spectral_config.stft
    static = tone(1000, dur=1.0)
    t = np.arange(static.size, dtype=np.float64) / SR
    evolving = (1.0 + 0.8 * np.sin(2.0 * np.pi * 3.0 * t)) * np.sin(2.0 * np.pi * 1000.0 * t)
    assert float(np.std(spectral_flux(evolving, stft))) > float(np.std(spectral_flux(static, stft)))
