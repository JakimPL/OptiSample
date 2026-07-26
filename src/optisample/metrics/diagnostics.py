from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.dsp import StftParams
from optisample.config.metrics import SegmentalSnrConfig
from optisample.dsp.spectral import band_energy, bandlimit, frame, spectral_flux
from optisample.metrics.base import Signal
from optisample.metrics.preprocess import integrated_loudness, match_length

_EPS: Final = 1e-12


def snr(reference: Signal, candidate: Signal) -> float:
    """Global signal-to-noise ratio in dB (``+inf`` if identical). Level-sensitive by design."""
    reference, candidate = match_length(reference, candidate)
    noise_power = float(np.sum((reference - candidate) ** 2))
    signal_power = float(np.sum(reference**2))
    if noise_power <= 0.0:
        return np.inf
    return 10.0 * float(np.log10((signal_power + _EPS) / noise_power))


def segmental_snr(
    reference: Signal,
    candidate: Signal,
    config: SegmentalSnrConfig,
) -> float:
    """Mean per-frame SNR (dB), clipped to the config's dB range and ignoring silent frames."""
    reference, candidate = match_length(reference, candidate)
    ref_frames = frame(reference, config.frame_length, config.hop_length)
    err_frames = frame(reference - candidate, config.frame_length, config.hop_length)
    signal_power = np.sum(ref_frames**2, axis=1)
    noise_power = np.sum(err_frames**2, axis=1)
    active = signal_power > _EPS
    if not np.any(active):
        return 0.0
    ratio = 10.0 * np.log10((signal_power[active] + _EPS) / (noise_power[active] + _EPS))
    return float(np.mean(np.clip(ratio, config.clip_low_db, config.clip_high_db)))


def si_sdr(reference: Signal, candidate: Signal) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (gain-robust; ``+inf`` if collinear)."""
    reference, candidate = match_length(reference, candidate)
    energy = float(np.sum(reference**2))
    if energy <= 0.0:
        return -np.inf
    scale = float(np.dot(candidate, reference)) / energy
    target = scale * reference
    distortion = float(np.sum((candidate - target) ** 2))
    if distortion <= 0.0:
        return np.inf
    return 10.0 * float(np.log10((float(np.sum(target**2)) + _EPS) / distortion))


def loudness_delta(reference: Signal, candidate: Signal, sample_rate: int) -> float:
    """Reference minus candidate integrated loudness (LU) — the raw level/velocity gap."""
    return integrated_loudness(reference, sample_rate) - integrated_loudness(candidate, sample_rate)


def hf_loss_db(
    reference: Signal,
    candidate: Signal,
    sample_rate: int,
    cutoff_hz: float,
) -> float:
    """Energy lost above ``cutoff_hz`` in dB (positive = candidate is duller than reference)."""
    nyquist = sample_rate / 2.0
    ref_energy = band_energy(reference, sample_rate, cutoff_hz, nyquist)
    cand_energy = band_energy(candidate, sample_rate, cutoff_hz, nyquist)
    return 10.0 * float(np.log10((ref_energy + _EPS) / (cand_energy + _EPS)))


def band_snr(
    reference: Signal,
    candidate: Signal,
    sample_rate: int,
    f_low: float,
    f_high: float,
) -> float:
    """SNR (dB) restricted to ``[f_low, f_high)`` — a low value flags aliasing folded into that band."""
    reference, candidate = match_length(reference, candidate)
    return snr(
        bandlimit(reference, sample_rate, f_low, f_high),
        bandlimit(candidate, sample_rate, f_low, f_high),
    )


@dataclass(frozen=True)
class LoopSeam:
    """Discontinuity at a loop boundary (in sample-amplitude units); larger = more audible click."""

    amplitude_jump: float
    derivative_jump: float

    @property
    def total(self) -> float:
        return self.amplitude_jump + self.derivative_jump


def loop_seam(signal: Signal, loop_start: int, loop_end: int) -> LoopSeam:
    """Amplitude + derivative mismatch when the loop wraps from ``loop_end`` back to ``loop_start``."""
    if not 0 < loop_start < loop_end < signal.size:
        raise ValueError("require 0 < loop_start < loop_end < len(signal)")
    amplitude_jump = abs(float(signal[loop_end]) - float(signal[loop_start]))
    slope_in = float(signal[loop_end]) - float(signal[loop_end - 1])
    slope_out = float(signal[loop_start + 1]) - float(signal[loop_start])
    return LoopSeam(amplitude_jump=amplitude_jump, derivative_jump=abs(slope_out - slope_in))


def flux_variance(signal: Signal, params: StftParams) -> float:
    """Standard deviation of spectral flux — near zero for a static loop, higher for evolving audio."""
    return float(np.std(spectral_flux(signal, params)))
