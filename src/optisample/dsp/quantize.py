"""Bit-depth reduction with peak-normalization, TPDF dither, and error-feedback noise shaping.

Storing a sample peak-normalized ("hot") *before* requantizing maximizes the effective bit usage:
the quantization noise floor sits ~1 LSB below full scale regardless of the source level, so a quiet
source requantized as-is would waste bits. The makeup gain (the inverse of the normalization) is
recorded by the caller so the velocity->volume map can restore the intended playback level later.

Quantization uses a mid-tread signed grid with step ``2**(1 - bits)`` over ``[-1, 1)``, matching IT's
signed PCM. On a full-scale sine the undithered SNR approaches the classic ``6.02 * bits + 1.76`` dB;
TPDF dither trades a few dB of that for a signal-independent (click-free) noise floor, and first-order
error-feedback noise shaping moves noise power out of the low band toward Nyquist.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Signal = NDArray[np.float64]

VALID_DEPTHS = (8, 16)


def quantization_step(bits: int) -> float:
    """Grid spacing of a signed ``bits``-bit quantizer over ``[-1, 1)`` (one LSB)."""
    if bits not in VALID_DEPTHS:
        raise ValueError(f"IT samples are 8- or 16-bit, got {bits}")
    return 2.0 ** (1 - bits)


def normalize_peak(signal: Signal, target_peak: float = 1.0) -> tuple[Signal, float]:
    """Scale so ``max|x| == target_peak``; return ``(normalized, gain)``. Silence -> ``(copy, 1.0)``."""
    data = np.asarray(signal, dtype=np.float64)
    largest = float(np.max(np.abs(data))) if data.size else 0.0
    if largest <= 0.0:
        return data.copy(), 1.0
    gain = target_peak / largest
    return np.asarray(data * gain, dtype=np.float64), gain


def apply_gain(signal: Signal, gain: float) -> Signal:
    """Scale amplitude by ``gain`` (the makeup gain is ``1 / normalize_peak``'s gain)."""
    return np.asarray(np.asarray(signal, dtype=np.float64) * gain, dtype=np.float64)


def _tpdf_dither(size: int, step: float, rng: np.random.Generator) -> Signal:
    """Triangular-PDF dither in ``(-step, step)`` = the difference of two uniform LSBs."""
    return np.asarray(step * (rng.random(size) - rng.random(size)), dtype=np.float64)


def _quantize_grid(values: Signal, step: float) -> Signal:
    """Round to the signed grid and clip to ``[-1, 1 - step]`` (IT's asymmetric signed range)."""
    grid = np.round(np.asarray(values, dtype=np.float64) / step) * step
    return np.asarray(np.clip(grid, -1.0, 1.0 - step), dtype=np.float64)


def _noise_shape(data: Signal, dither: Signal, step: float) -> Signal:
    """First-order error diffusion: carry the rounding residual forward so its spectrum is high-pass."""
    out = np.empty_like(data)
    carry = 0.0
    for index in range(data.size):
        desired = float(data[index]) + float(dither[index]) + carry
        quantized = float(np.clip(round(desired / step) * step, -1.0, 1.0 - step))
        carry = desired - quantized
        out[index] = quantized
    return out


def requantize(
    signal: Signal,
    bits: int,
    *,
    dither: bool = True,
    noise_shaping: bool = False,
    rng: np.random.Generator | None = None,
) -> Signal:
    """Requantize ``signal`` to ``bits`` bits over ``[-1, 1)`` with optional TPDF dither / noise shaping.

    ``rng`` defaults to a fixed seed so encoding is reproducible; pass one to vary the dither.
    """
    step = quantization_step(bits)
    data = np.asarray(signal, dtype=np.float64)
    if data.size == 0:
        return data.copy()
    generator = rng if rng is not None else np.random.default_rng(0)
    noise = _tpdf_dither(data.size, step, generator) if dither else np.zeros(data.size, dtype=np.float64)
    if noise_shaping:
        return _noise_shape(data, noise, step)
    return _quantize_grid(data + noise, step)
