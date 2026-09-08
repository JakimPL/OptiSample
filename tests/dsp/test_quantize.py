from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod import BitDepth

from optisample.config.codec import QuantizeConfig
from optisample.dsp.level import db_to_gain
from optisample.dsp.quantize import (
    apply_gain,
    headroom_peak,
    normalize_peak,
    quantization_step,
    release_fade,
    requantize,
)
from optisample.dsp.spectral import band_energy
from optisample.metrics.diagnostics import snr

SR = 44_100
_FADE_FRAMES = 441  # 10 ms at the test rate, the shipped ramp


def test_quantization_step_values() -> None:
    assert quantization_step(BitDepth.SIXTEEN) == 2.0**-15
    assert quantization_step(BitDepth.EIGHT) == 2.0**-7


def test_a_step_is_one_of_the_integers_its_depth_stores() -> None:
    """The float grid and the stored integer range are the same fact read from opposite ends."""
    for depth in BitDepth:
        assert quantization_step(depth) * depth.scale == 1.0


def test_normalize_peak_hits_target(sine: Callable[..., NDArray[np.float64]]) -> None:
    norm, gain = normalize_peak(0.3 * sine(440.0), target_peak=1.0)
    assert float(np.max(np.abs(norm))) == pytest.approx(1.0)
    assert gain == pytest.approx(1.0 / 0.3, rel=1e-6)


def test_normalize_peak_silence_is_identity(quantize_config: QuantizeConfig) -> None:
    silence = np.zeros(128, dtype=np.float64)
    norm, gain = normalize_peak(silence, headroom_peak(quantize_config.headroom_db))
    assert gain == 1.0
    assert np.array_equal(norm, silence)


def test_a_named_reference_scales_every_clip_by_one_factor(sine: Callable[..., NDArray[np.float64]]) -> None:
    """What a format storing no per-sample gain needs: the margin between two recordings, kept in the PCM."""
    loud, quiet = sine(440.0), 0.25 * sine(440.0)
    reference = float(np.max(np.abs(loud)))
    scaled_loud, loud_gain = normalize_peak(loud, 1.0, reference_peak=reference)
    scaled_quiet, quiet_gain = normalize_peak(quiet, 1.0, reference_peak=reference)
    assert loud_gain == quiet_gain
    assert float(np.max(np.abs(scaled_quiet))) == pytest.approx(0.25 * float(np.max(np.abs(scaled_loud))))


def test_a_reference_of_silence_leaves_a_clip_as_it_stands(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)
    scaled, gain = normalize_peak(signal, 1.0, reference_peak=0.0)
    assert gain == 1.0
    assert np.array_equal(scaled, signal)


def test_headroom_leaves_the_peak_that_far_under_full_scale() -> None:
    assert headroom_peak(0.0) == pytest.approx(1.0)
    assert headroom_peak(6.0) == pytest.approx(db_to_gain(-6.0))
    assert headroom_peak(0.5) < 1.0


def test_apply_gain_scales_linearly(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)
    assert np.allclose(apply_gain(signal, 0.5), 0.5 * signal)


def test_a_release_ramp_closes_a_span_on_silence(sine: Callable[..., NDArray[np.float64]]) -> None:
    """The last frame lands on zero, which is what a tracker reaching the end of the sample plays."""
    signal = sine(440.0)
    faded = release_fade(signal, _FADE_FRAMES)

    assert faded[-1] == pytest.approx(0.0)
    assert np.allclose(faded[: signal.size - _FADE_FRAMES], signal[: signal.size - _FADE_FRAMES])


def test_a_release_ramp_descends_over_the_frames_it_is_given() -> None:
    """The ramp is linear, so each frame of the closing stretch keeps a smaller share of what it held."""
    ramp_frames = 8
    faded = release_fade(np.ones(64, dtype=np.float64), ramp_frames)

    assert np.allclose(faded[-ramp_frames:], np.linspace(1.0, 0.0, ramp_frames))
    assert np.all(np.diff(faded[-ramp_frames:]) < 0.0)


def test_a_span_shorter_than_the_ramp_is_faded_over_its_whole_length() -> None:
    faded = release_fade(np.ones(4, dtype=np.float64), _FADE_FRAMES)

    assert faded[0] == pytest.approx(1.0)
    assert faded[-1] == pytest.approx(0.0)


def test_a_ramp_of_no_frames_leaves_the_span_as_it_stands(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)

    assert np.array_equal(release_fade(signal, 0), signal)


@pytest.mark.parametrize("depth", tuple(BitDepth))
def test_undithered_snr_matches_theory(depth: BitDepth, sine: Callable[..., NDArray[np.float64]]) -> None:
    # A full-scale sine requantized to n bits → SNR ≈ 6.02 n + 1.76 dB.
    signal = sine(997.0)
    quantized = requantize(signal, depth, dither=False)
    assert snr(signal, quantized) == pytest.approx(6.02 * depth + 1.76, abs=1.5)


def test_undithered_output_lies_on_grid(sine: Callable[..., NDArray[np.float64]]) -> None:
    step = quantization_step(BitDepth.EIGHT)
    quantized = requantize(sine(440.0), BitDepth.EIGHT, dither=False)
    ratios = quantized / step
    assert np.allclose(ratios, np.round(ratios))


def test_dither_lowers_snr_but_stays_close_to_theory(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(997.0)
    undithered = snr(signal, requantize(signal, BitDepth.EIGHT, dither=False))
    dithered = snr(signal, requantize(signal, BitDepth.EIGHT, dither=True, rng=np.random.default_rng(0)))
    assert dithered < undithered
    assert dithered > 6.02 * 8 + 1.76 - 8.0  # TPDF costs a few dB, not tens


def test_requantize_clips_to_signed_range(sine: Callable[..., NDArray[np.float64]]) -> None:
    step = quantization_step(BitDepth.EIGHT)
    quantized = requantize(1.5 * sine(440.0), BitDepth.EIGHT, dither=True, rng=np.random.default_rng(0))
    assert quantized.max() <= 1.0 - step + 1e-9
    assert quantized.min() >= -1.0 - 1e-9


def test_noise_shaping_moves_error_toward_high_frequencies(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(997.0)
    err_plain = signal - requantize(signal, BitDepth.EIGHT, dither=True, rng=np.random.default_rng(0))
    err_shaped = signal - requantize(
        signal, BitDepth.EIGHT, dither=True, noise_shaping=True, rng=np.random.default_rng(0)
    )
    low_band_plain = band_energy(err_plain, SR, 0.0, SR / 8.0)
    low_band_shaped = band_energy(err_shaped, SR, 0.0, SR / 8.0)
    assert low_band_shaped < low_band_plain


def test_requantize_empty_signal_is_empty() -> None:
    assert requantize(np.zeros(0, dtype=np.float64), BitDepth.EIGHT).size == 0


def test_requantize_default_rng_is_deterministic(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)
    assert np.array_equal(requantize(signal, BitDepth.EIGHT), requantize(signal, BitDepth.EIGHT))


def test_requantize_dither_varies_with_seed(sine: Callable[..., NDArray[np.float64]]) -> None:
    signal = sine(440.0)
    first = requantize(signal, BitDepth.EIGHT, rng=np.random.default_rng(1))
    second = requantize(signal, BitDepth.EIGHT, rng=np.random.default_rng(2))
    assert not np.array_equal(first, second)
