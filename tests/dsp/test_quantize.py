from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dsp import QuantizeConfig
from optisample.dsp.quantize import apply_gain, normalize_peak, quantization_step, requantize
from optisample.dsp.spectral import band_energy
from optisample.metrics.diagnostics import snr

SR = 44_100


def test_quantization_step_values() -> None:
    assert quantization_step(16) == 2.0**-15
    assert quantization_step(8) == 2.0**-7


def test_quantization_step_rejects_unsupported_depth() -> None:
    with pytest.raises(ValueError):
        quantization_step(24)


def test_normalize_peak_hits_target(sine: Callable[..., NDArray[np.float64]]) -> None:
    norm, gain = normalize_peak(0.3 * sine(440.0), target_peak=1.0)
    assert float(np.max(np.abs(norm))) == pytest.approx(1.0)
    assert gain == pytest.approx(1.0 / 0.3, rel=1e-6)


def test_normalize_peak_silence_is_identity(quantize_config: QuantizeConfig) -> None:
    silence = np.zeros(128, dtype=np.float64)
    norm, gain = normalize_peak(silence, quantize_config.target_peak)
    assert gain == 1.0
    assert np.array_equal(norm, silence)


def test_apply_gain_scales_linearly(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)
    assert np.allclose(apply_gain(signal, 0.5), 0.5 * signal)


@pytest.mark.parametrize("bits", [8, 16])
def test_undithered_snr_matches_theory(bits: int, sine: Callable[..., NDArray[np.float64]]) -> None:
    # A full-scale sine requantized to n bits → SNR ≈ 6.02 n + 1.76 dB.
    signal = sine(997.0)
    quantized = requantize(signal, bits, dither=False)
    assert snr(signal, quantized) == pytest.approx(6.02 * bits + 1.76, abs=1.5)


def test_undithered_output_lies_on_grid(sine: Callable[..., NDArray[np.float64]]) -> None:
    step = quantization_step(8)
    quantized = requantize(sine(440.0), 8, dither=False)
    ratios = quantized / step
    assert np.allclose(ratios, np.round(ratios))


def test_dither_lowers_snr_but_stays_close_to_theory(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(997.0)
    undithered = snr(signal, requantize(signal, 8, dither=False))
    dithered = snr(signal, requantize(signal, 8, dither=True, rng=np.random.default_rng(0)))
    assert dithered < undithered
    assert dithered > 6.02 * 8 + 1.76 - 8.0  # TPDF costs a few dB, not tens


def test_requantize_clips_to_signed_range(sine: Callable[..., NDArray[np.float64]]) -> None:
    step = quantization_step(8)
    quantized = requantize(1.5 * sine(440.0), 8, dither=True, rng=np.random.default_rng(0))
    assert quantized.max() <= 1.0 - step + 1e-9
    assert quantized.min() >= -1.0 - 1e-9


def test_noise_shaping_moves_error_toward_high_frequencies(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(997.0)
    err_plain = signal - requantize(signal, 8, dither=True, rng=np.random.default_rng(0))
    err_shaped = signal - requantize(signal, 8, dither=True, noise_shaping=True, rng=np.random.default_rng(0))
    low_band_plain = band_energy(err_plain, SR, 0.0, SR / 8.0)
    low_band_shaped = band_energy(err_shaped, SR, 0.0, SR / 8.0)
    assert low_band_shaped < low_band_plain


def test_requantize_empty_signal_is_empty() -> None:
    assert requantize(np.zeros(0, dtype=np.float64), 8).size == 0


def test_requantize_default_rng_is_deterministic(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)
    assert np.array_equal(requantize(signal, 8), requantize(signal, 8))


def test_requantize_dither_varies_with_seed(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)
    first = requantize(signal, 8, rng=np.random.default_rng(1))
    second = requantize(signal, 8, rng=np.random.default_rng(2))
    assert not np.array_equal(first, second)
