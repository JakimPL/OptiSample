"""Encoding-preview degradations, kept deliberately simple.

These reproduce the transforms used in the P1 smoke test — bit-depth reduction, bandlimiting,
level change, and a down/up resample round-trip — so their effect can be *heard* and *seen* next
to the metric that is supposed to catch them. They are previews only; P2 replaces them with the
real encoder (dithered/noise-shaped requantization, proper resampling, trim-to-duration).

Every degradation returns a signal the same length as its input, so the metrics' length-matching
never has to trim and frame alignment is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

from optisample.dsp.spectral import bandlimit

Signal = NDArray[np.float64]

DegradeKind = Literal["quantize", "lowpass", "gain", "resample"]


def quantize(signal: Signal, bits: int) -> Signal:
    """Mid-tread uniform quantization to ``bits`` bits over the full-scale [-1, 1] range."""
    step = 2.0 / (2**bits)
    return np.asarray(np.round(np.asarray(signal, dtype=np.float64) / step) * step, dtype=np.float64)


def lowpass(signal: Signal, sample_rate: int, cutoff_hz: float) -> Signal:
    """Brick-wall lowpass via the shared spectral primitive (energy above ``cutoff_hz`` removed)."""
    return bandlimit(np.asarray(signal, dtype=np.float64), sample_rate, 0.0, cutoff_hz)


def gain(signal: Signal, factor: float) -> Signal:
    """Scale amplitude by ``factor`` (timbre-preserving; a pure level change)."""
    return np.asarray(np.asarray(signal, dtype=np.float64) * factor, dtype=np.float64)


def resample_roundtrip(signal: Signal, sample_rate: int, target_sr: int, *, antialias: bool = True) -> Signal:
    """Downsample to ``target_sr`` then back, at the original length.

    With ``antialias`` the down/up steps are band-limited (the honest "store fewer samples" preview);
    without it, naive stride decimation folds high frequencies down as aliasing (which in-band SNR
    is meant to expose).
    """
    data = np.asarray(signal, dtype=np.float64)
    length = data.size
    if target_sr >= sample_rate or length == 0:
        return data.copy()
    n_down = max(1, int(round(length * target_sr / sample_rate)))
    if antialias:
        down = np.asarray(resample(data, n_down), dtype=np.float64)
        return np.asarray(resample(down, length), dtype=np.float64)
    stride = max(1, int(round(sample_rate / target_sr)))
    down = data[::stride]
    return np.asarray(np.interp(np.linspace(0.0, 1.0, length), np.linspace(0.0, 1.0, down.size), down), np.float64)


@dataclass(frozen=True)
class DegradeSpec:
    """A named, parameterized degradation the notebook can build from UI controls."""

    kind: DegradeKind
    bits: int = 8
    cutoff_hz: float = 3000.0
    factor: float = 0.3
    target_sr: int = 22_050
    antialias: bool = True

    @property
    def label(self) -> str:
        if self.kind == "quantize":
            return f"{self.bits}-bit"
        if self.kind == "lowpass":
            return f"lowpass {self.cutoff_hz / 1000.0:g} kHz"
        if self.kind == "gain":
            return f"gain x{self.factor:g}"
        return f"resample {self.target_sr} Hz{'' if self.antialias else ' (naive)'}"


def apply(spec: DegradeSpec, signal: Signal, sample_rate: int) -> Signal:
    """Apply a :class:`DegradeSpec` to ``signal``."""
    if spec.kind == "quantize":
        return quantize(signal, spec.bits)
    if spec.kind == "lowpass":
        return lowpass(signal, sample_rate, spec.cutoff_hz)
    if spec.kind == "gain":
        return gain(signal, spec.factor)
    return resample_roundtrip(signal, sample_rate, spec.target_sr, antialias=spec.antialias)


def smoke_specs() -> list[DegradeSpec]:
    """The canonical P1 smoke-test set: transparent 16-bit, audible 8-bit, a 3 kHz cut, a level drop."""
    return [
        DegradeSpec(kind="quantize", bits=16),
        DegradeSpec(kind="quantize", bits=8),
        DegradeSpec(kind="lowpass", cutoff_hz=3000.0),
        DegradeSpec(kind="gain", factor=0.3),
    ]
