from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.cluster.harmonics import harmonic_bands, harmonic_peaks, harmonic_profile
from optisample.config.cluster import DescriptorConfig
from optisample.config.spectral import StftParams
from tests.cluster.conftest import SR, Signal

ROOT_HZ = 220.0
PARAMS = StftParams(n_fft=4096, hop_length=1024)

_LOUD_GAIN = 4.0  # the gain a copy of a take is captured at, a power of two so the scaling stays exact


def test_each_partial_is_read_in_a_band_of_its_own(descriptor_config: DescriptorConfig) -> None:
    """Half a pitch either side tiles the spectrum, so neighbouring partials stay told apart."""
    bands = harmonic_bands(SR, PARAMS, ROOT_HZ, descriptor_config)

    assert len(bands) == descriptor_config.harmonics
    assert all(low < high for low, high in bands)
    assert all(earlier[1] <= later[0] for earlier, later in zip(bands, bands[1:]))


def test_a_band_reaching_past_the_transform_reads_nothing(descriptor_config: DescriptorConfig) -> None:
    """Above Nyquist there is no room to carry a partial, so its band comes back empty and reads zero."""
    bands = harmonic_bands(SR, PARAMS, 8_000.0, descriptor_config)
    empty = [(low, high) for low, high in bands if high <= low]

    assert empty
    assert np.allclose(harmonic_peaks(np.ones((3, PARAMS.n_fft // 2 + 1)), bands)[:, -1], 0.0)


def test_a_partial_a_little_off_its_multiple_is_still_read_by_the_band_it_names(
    descriptor_config: DescriptorConfig,
) -> None:
    """A struck string rings its partials wider apart than whole multiples, and the band follows them."""
    times = np.arange(SR, dtype=np.float64) / SR
    stretched = np.sin(2.0 * np.pi * 3.2 * ROOT_HZ * times)
    profile = harmonic_profile(stretched, SR, params=PARAMS, root_hz=ROOT_HZ, config=descriptor_config)

    assert int(np.argmax(profile.mean(axis=0))) == 2  # the third partial, counted from the fundamental


def test_the_reading_states_a_balance_past_the_level_the_take_was_captured_at(
    descriptor_config: DescriptorConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    signal = ringing_note(ROOT_HZ)
    quiet = harmonic_profile(signal, SR, params=PARAMS, root_hz=ROOT_HZ, config=descriptor_config)
    loud = harmonic_profile(_LOUD_GAIN * signal, SR, params=PARAMS, root_hz=ROOT_HZ, config=descriptor_config)

    assert np.allclose(loud, quiet, atol=1e-9)


def test_every_frame_states_its_partials_around_their_own_mean(
    descriptor_config: DescriptorConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    profile = harmonic_profile(ringing_note(ROOT_HZ), SR, params=PARAMS, root_hz=ROOT_HZ, config=descriptor_config)

    assert profile.shape[1] == descriptor_config.harmonics
    assert np.allclose(profile.mean(axis=1), 0.0, atol=1e-9)


def test_a_frame_states_its_balance_across_a_range_of_its_own(
    descriptor_config: DescriptorConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """Flooring under each frame's own loudest partial bounds how far apart two readings can stand."""
    profile = harmonic_profile(ringing_note(ROOT_HZ), SR, params=PARAMS, root_hz=ROOT_HZ, config=descriptor_config)

    spread = profile.max(axis=1) - profile.min(axis=1)

    assert float(np.max(spread)) <= descriptor_config.harmonic_range_db + 1e-9


def test_a_silent_take_reads_one_flat_balance(descriptor_config: DescriptorConfig) -> None:
    profile = harmonic_profile(np.zeros(SR), SR, params=PARAMS, root_hz=ROOT_HZ, config=descriptor_config)

    assert np.allclose(profile, 0.0)


def test_two_notes_an_octave_apart_sharing_a_balance_read_alike(
    descriptor_config: DescriptorConfig,
    ringing_note: Callable[..., Signal],
) -> None:
    """Indexing by partial number leaves the register out, so a group here means a sound.

    The two takes are read over transforms sized to their own pitch, which resolves each note's partials
    to the same number of bins and puts both readings on one footing.
    """
    low_params = StftParams(n_fft=8192, hop_length=2048)
    low = harmonic_profile(ringing_note(ROOT_HZ), SR, params=low_params, root_hz=ROOT_HZ, config=descriptor_config)
    high = harmonic_profile(
        ringing_note(2.0 * ROOT_HZ), SR, params=PARAMS, root_hz=2.0 * ROOT_HZ, config=descriptor_config
    )

    assert float(np.max(np.abs(high.mean(axis=0) - low.mean(axis=0)))) == pytest.approx(0.0, abs=1.0)
