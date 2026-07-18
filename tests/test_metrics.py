from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.dsp.spectral import bandlimit
from optisample.metrics import (
    MetricContext,
    MultiResolutionStft,
    SpectralShape,
    available,
    default_composite,
    evaluate,
    get,
    integrated_loudness,
    loudness_normalize,
    prepare,
    register,
    unregister,
)
from optisample.metrics.base import Signal

SR = 16_000
_METRIC_NAMES = ("mrstft", "logmel_l1", "mcd", "spectral_shape")


def harmonic(f0: float, dur: float = 0.7, n_partials: int = 6, amp: float = 0.8) -> NDArray[np.float64]:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    signal = np.zeros_like(t)
    for k in range(1, n_partials + 1):
        signal += (1.0 / k) * np.sin(2.0 * np.pi * f0 * k * t)
    return signal / float(np.max(np.abs(signal))) * amp


def quantize(signal: NDArray[np.float64], bits: int) -> NDArray[np.float64]:
    step = 2.0 / (2**bits)
    return np.round(signal / step) * step


@dataclass(frozen=True)
class _DummyMetric:
    name: str = "tmp_metric"

    def distance(self, reference: Signal, candidate: Signal, ctx: MetricContext) -> float:
        del reference, candidate, ctx
        return 0.0


def test_registry_exposes_defaults() -> None:
    for name in _METRIC_NAMES:
        assert name in available()
        assert get(name).name == name


def test_get_unknown_metric_raises() -> None:
    with pytest.raises(KeyError):
        get("does_not_exist")


def test_register_duplicate_raises() -> None:
    with pytest.raises(ValueError):
        register(MultiResolutionStft())  # "mrstft" already registered


def test_register_and_unregister_round_trip() -> None:
    register(_DummyMetric())
    assert "tmp_metric" in available()
    unregister("tmp_metric")
    assert "tmp_metric" not in available()


def test_each_metric_zero_for_identical() -> None:
    signal = harmonic(220.0)
    ctx = MetricContext(sample_rate=SR)
    for name in _METRIC_NAMES:
        assert get(name).distance(signal, signal, ctx) == pytest.approx(0.0, abs=1e-6)


def test_evaluate_identical_has_zero_fidelity() -> None:
    signal = harmonic(220.0)
    report = evaluate(signal, signal, SR)
    assert report.fidelity == pytest.approx(0.0, abs=1e-6)
    assert report.diagnostics["snr_db"] == np.inf
    assert set(report.breakdown) == set(_METRIC_NAMES)


def test_composite_is_monotone_with_quantization() -> None:
    signal = harmonic(220.0)
    coarse = evaluate(signal, quantize(signal, 4), SR).fidelity
    fine = evaluate(signal, quantize(signal, 12), SR).fidelity
    assert coarse > fine > 0.0


def test_mrstft_increases_with_bandlimiting() -> None:
    signal = harmonic(220.0)
    ctx = MetricContext(sample_rate=SR)
    metric = MultiResolutionStft()
    lowpassed = bandlimit(signal, SR, 0.0, 800.0)  # strips upper harmonics
    assert metric.distance(signal, lowpassed, ctx) > metric.distance(signal, signal, ctx)


def test_spectral_shape_flags_static_loop() -> None:
    base = harmonic(220.0, dur=1.0)
    t = np.arange(base.size, dtype=np.float64) / SR
    evolving = base * (1.0 + 0.7 * np.sin(2.0 * np.pi * 2.0 * t))
    shape = SpectralShape()
    against_static = shape.components(evolving, base, SR)["flux_variance"]
    against_self = shape.components(evolving, evolving, SR)["flux_variance"]
    assert against_static > against_self


def test_default_composite_accepts_custom_weights() -> None:
    composite = default_composite({"mrstft": 1.0})
    assert len(composite.components) == 1
    signal = harmonic(220.0)
    assert composite.distance(signal, signal, MetricContext(SR)) == pytest.approx(0.0, abs=1e-6)


def test_loudness_normalize_hits_target() -> None:
    signal = harmonic(220.0, dur=1.0)
    normalized = loudness_normalize(signal, SR, target_lufs=-20.0)
    assert integrated_loudness(normalized, SR) == pytest.approx(-20.0, abs=0.5)


def test_loudness_normalize_leaves_silence_untouched() -> None:
    silence = np.zeros(SR, dtype=np.float64)
    np.testing.assert_array_equal(loudness_normalize(silence, SR), silence)


def test_integrated_loudness_of_silence_is_neg_inf() -> None:
    assert integrated_loudness(np.zeros(SR, dtype=np.float64), SR) == -np.inf


def test_prepare_matches_length_and_loudness() -> None:
    loud = harmonic(220.0, dur=1.0)
    soft = 0.05 * harmonic(220.0, dur=1.0)
    ref, cand, ctx = prepare(loud, soft, SR)
    assert ref.size == cand.size
    assert ctx.normalized is True
    assert integrated_loudness(ref, SR) == pytest.approx(integrated_loudness(cand, SR), abs=0.5)


def test_prepare_short_signal_uses_fallback() -> None:
    short = harmonic(220.0, dur=0.1)  # below the BS.1770 block → RMS fallback, must not crash
    ref, cand, _ = prepare(short, short.copy(), SR)
    assert ref.size == cand.size
