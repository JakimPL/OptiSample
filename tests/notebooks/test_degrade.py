from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from notebooks.utils import degrade
from optisample.dsp.spectral import band_energy

SR = 44_100


def test_quantize_snaps_to_step_grid(tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone()
    quantized = degrade.quantize(signal, 8)
    step = 2.0 / (2**8)
    residual = quantized / step
    assert np.allclose(residual, np.round(residual))
    assert np.unique(quantized).size <= 2**8


def test_lowpass_removes_high_frequency_energy(tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone(freq=8000.0)
    filtered = degrade.lowpass(signal, SR, 3000.0)
    assert band_energy(filtered, SR, 3000.0, SR / 2.0) < 1e-6 * band_energy(signal, SR, 3000.0, SR / 2.0)


def test_gain_scales_amplitude(tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone()
    assert np.allclose(degrade.gain(signal, 0.25), 0.25 * signal)


@pytest.mark.parametrize("antialias", [True, False])
def test_resample_roundtrip_preserves_length(antialias: bool, tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone()
    out = degrade.resample_roundtrip(signal, SR, 22_050, antialias=antialias)
    assert out.size == signal.size


def test_naive_resample_aliases_more_than_antialiased(tone: Callable[..., NDArray[np.float64]]) -> None:
    # A tone above the downsampled Nyquist aliases badly without a filter.
    signal = tone(freq=16_000.0)
    clean = degrade.resample_roundtrip(signal, SR, 22_050, antialias=True)
    aliased = degrade.resample_roundtrip(signal, SR, 22_050, antialias=False)
    err_clean = float(np.mean((signal - clean) ** 2))
    err_aliased = float(np.mean((signal - aliased) ** 2))
    assert err_aliased > err_clean


def test_degrade_spec_labels_and_dispatch(tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone()
    specs = {
        "8-bit": degrade.DegradeSpec(kind="quantize", bits=8),
        "lowpass 3 kHz": degrade.DegradeSpec(kind="lowpass", cutoff_hz=3000.0),
        "gain x0.3": degrade.DegradeSpec(kind="gain", factor=0.3),
        "resample 22050 Hz": degrade.DegradeSpec(kind="resample", target_sr=22_050),
    }
    for expected_label, spec in specs.items():
        assert spec.label == expected_label
        assert degrade.apply(spec, signal, SR).size == signal.size


def test_smoke_specs_cover_the_p1_set() -> None:
    labels = [spec.label for spec in degrade.smoke_specs()]
    assert labels == ["16-bit", "8-bit", "lowpass 3 kHz", "gain x0.3"]
