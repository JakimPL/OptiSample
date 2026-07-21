from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import resample_poly

from optisample.config.metrics import MetricsConfig
from optisample.dsp.spectral import bandlimit
from optisample.metrics.diagnostics import (
    band_snr,
    hf_loss_db,
    loop_seam,
    loudness_delta,
    segmental_snr,
    si_sdr,
    snr,
)

SR = 44_100


def sine(freq: float, dur: float, sr: int = SR, amp: float = 1.0) -> NDArray[np.float64]:
    t = np.arange(int(dur * sr), dtype=np.float64) / sr
    return amp * np.sin(2.0 * np.pi * freq * t)


@pytest.mark.parametrize("bits", [8, 12, 16])
def test_quantization_snr_matches_theory(bits: int, quantize: Callable[..., NDArray[np.float64]]) -> None:
    # Full-scale sine quantized to n bits → SNR ≈ 6.02 n + 1.76 dB.
    signal = sine(997.0, 1.0, amp=1.0)
    assert snr(signal, quantize(signal, bits)) == pytest.approx(6.02 * bits + 1.76, abs=1.2)


def test_snr_identical_is_infinite() -> None:
    signal = sine(440.0, 0.5)
    assert snr(signal, signal) == np.inf


def test_si_sdr_is_scale_invariant() -> None:
    reference = sine(440.0, 0.5)
    candidate = reference + 0.01 * sine(1300.0, 0.5)
    assert si_sdr(reference, candidate) == pytest.approx(si_sdr(reference, 5.0 * candidate), abs=1e-6)


def test_si_sdr_collinear_is_huge() -> None:
    reference = sine(440.0, 0.5)
    assert si_sdr(reference, 2.0 * reference) > 100.0


def test_si_sdr_silent_reference_is_neg_inf() -> None:
    assert si_sdr(np.zeros(1000), np.ones(1000)) == -np.inf


def test_segmental_snr_identical_hits_ceiling(metrics_config: MetricsConfig) -> None:
    signal = sine(440.0, 0.5)
    ceiling = metrics_config.preprocess.segmental.clip_high_db
    assert segmental_snr(signal, signal, metrics_config.preprocess.segmental) == pytest.approx(ceiling)


def test_segmental_snr_degraded_below_ceiling(
    metrics_config: MetricsConfig, quantize: Callable[..., NDArray[np.float64]]
) -> None:
    signal = sine(440.0, 0.5)
    ceiling = metrics_config.preprocess.segmental.clip_high_db
    assert segmental_snr(signal, quantize(signal, 5), metrics_config.preprocess.segmental) < ceiling


def test_segmental_snr_silence_is_zero(metrics_config: MetricsConfig) -> None:
    assert segmental_snr(np.zeros(4096), np.zeros(4096), metrics_config.preprocess.segmental) == 0.0


def test_loudness_delta_positive_when_reference_louder() -> None:
    loud = sine(440.0, 1.0, amp=0.9)
    soft = sine(440.0, 1.0, amp=0.1)
    assert loudness_delta(loud, soft, SR) > 0.0


def test_hf_loss_positive_when_lowpassed() -> None:
    signal = sine(200.0, 0.5) + sine(6000.0, 0.5)
    lowpassed = bandlimit(signal, SR, 0.0, 2000.0)  # removes the 6 kHz partial
    assert hf_loss_db(signal, lowpassed, SR, cutoff_hz=2000.0) > 20.0
    assert hf_loss_db(signal, signal, SR, cutoff_hz=2000.0) == pytest.approx(0.0, abs=1e-6)


def test_band_snr_detects_aliasing() -> None:
    sr = 8_000
    reference = sine(500.0, 1.0, sr) + sine(3000.0, 1.0, sr)
    clean = resample_poly(resample_poly(reference, 1, 2), 2, 1)  # anti-aliased down then up
    aliased = resample_poly(reference[::2], 2, 1)  # naive decimation folds 3 kHz → 1 kHz
    clean_snr = band_snr(reference, clean, sr, 0.0, 1900.0)
    aliased_snr = band_snr(reference, aliased, sr, 0.0, 1900.0)
    assert clean_snr > aliased_snr + 15.0
    assert clean_snr > 20.0


def test_loop_seam_perfect_versus_bad() -> None:
    sr = 8_000
    length = 1000
    signal = sine(100.0, length / sr, sr)  # period is exactly 80 samples
    perfect = loop_seam(signal, 80, 880)  # span of 10 whole periods
    bad = loop_seam(signal, 80, 110)  # ends mid-period
    assert perfect.total < 1e-6
    assert bad.total > 0.1


def test_loop_seam_invalid_bounds_raise() -> None:
    signal = sine(100.0, 0.1, 8_000)
    with pytest.raises(ValueError):
        loop_seam(signal, 0, 50)  # loop_start must be > 0
    with pytest.raises(ValueError):
        loop_seam(signal, 10, signal.size)  # loop_end must be < len
