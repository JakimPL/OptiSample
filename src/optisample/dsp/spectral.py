"""Spectral analysis primitives shared by the metrics.

Kept dependency-light (numpy + scipy) and fully typed. STFT/mel parameters are bundled into
small frozen param objects so callers pass one config, not a long argument list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct

Signal = NDArray[np.float64]

_LOG_FLOOR = 1e-10


@dataclass(frozen=True)
class StftParams:
    """Short-time Fourier transform settings."""

    n_fft: int = 1024
    hop_length: int = 256


@dataclass(frozen=True)
class MelParams:
    """Mel-spectrogram settings (includes its own STFT sizing)."""

    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    fmin: float = 0.0
    fmax: float | None = None

    def stft(self) -> StftParams:
        return StftParams(n_fft=self.n_fft, hop_length=self.hop_length)


def frame(signal: Signal, frame_length: int, hop_length: int) -> Signal:
    """Slice ``signal`` into overlapping rectangular frames ``(n_frames, frame_length)`` (right-padded)."""
    data = np.asarray(signal, dtype=np.float64)
    if data.size < frame_length:
        data = np.pad(data, (0, frame_length - data.size))
    n_frames = 1 + (data.size - frame_length) // hop_length
    offsets = hop_length * np.arange(n_frames)[:, None]
    indices = np.arange(frame_length)[None, :] + offsets
    return np.asarray(np.take(data, indices), dtype=np.float64)


def stft_magnitude(signal: Signal, params: StftParams = StftParams()) -> Signal:
    """Magnitude STFT with a Hann window: ``(n_frames, n_fft // 2 + 1)``."""
    window = np.hanning(params.n_fft)
    frames = frame(signal, params.n_fft, params.hop_length) * window
    spectrum = np.fft.rfft(frames, n=params.n_fft, axis=1)
    return np.abs(spectrum).astype(np.float64)


def _hz_to_mel(hertz: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return np.asarray(2595.0 * np.log10(1.0 + np.asarray(hertz, dtype=np.float64) / 700.0), dtype=np.float64)


def _mel_to_hz(mel: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(700.0 * (10.0 ** (mel / 2595.0) - 1.0), dtype=np.float64)


def mel_filterbank(sample_rate: int, params: MelParams = MelParams()) -> Signal:
    """Triangular mel filterbank ``(n_mels, n_fft // 2 + 1)`` on the FFT frequency grid."""
    fmax = params.fmax if params.fmax is not None else sample_rate / 2.0
    fft_freqs = np.fft.rfftfreq(params.n_fft, 1.0 / sample_rate)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(params.fmin), _hz_to_mel(fmax), params.n_mels + 2))
    filters = np.zeros((params.n_mels, fft_freqs.size), dtype=np.float64)
    for band in range(1, params.n_mels + 1):
        left, center, right = float(edges[band - 1]), float(edges[band]), float(edges[band + 1])
        rising = (fft_freqs - left) / max(center - left, _LOG_FLOOR)
        falling = (right - fft_freqs) / max(right - center, _LOG_FLOOR)
        filters[band - 1] = np.clip(np.minimum(rising, falling), 0.0, None)
    return filters


def melspectrogram(signal: Signal, sample_rate: int, params: MelParams = MelParams()) -> Signal:
    """Mel power spectrogram ``(n_frames, n_mels)``."""
    power = stft_magnitude(signal, params.stft()) ** 2
    filters = mel_filterbank(sample_rate, params)
    return np.asarray(power @ filters.T, dtype=np.float64)


def mfcc(
    signal: Signal, sample_rate: int, n_mfcc: int = 13, params: MelParams = MelParams(), top_db: float = 80.0
) -> Signal:
    """Mel-frequency cepstral coefficients ``(n_frames, n_mfcc)`` (DCT-II of log-mel energies).

    The log-mel is floored ``top_db`` dB below its peak so near-silent bins (and any noise
    filling them) do not dominate the cepstrum. Flooring is done in the natural-log domain,
    keeping mel-cepstral-distortion's ``10 / ln 10`` dB constant valid downstream.
    """
    log_mel = np.log(np.maximum(melspectrogram(signal, sample_rate, params), _LOG_FLOOR))
    floor = float(np.max(log_mel)) - top_db * float(np.log(10.0)) / 10.0
    coeffs = np.asarray(dct(np.maximum(log_mel, floor), type=2, norm="ortho", axis=1), dtype=np.float64)
    return np.asarray(coeffs[:, :n_mfcc], dtype=np.float64)


def _magnitude_and_freqs(signal: Signal, sample_rate: int, params: StftParams) -> tuple[Signal, Signal]:
    magnitude = stft_magnitude(signal, params)
    freqs = np.fft.rfftfreq(params.n_fft, 1.0 / sample_rate)
    return magnitude, np.asarray(freqs, dtype=np.float64)


def spectral_centroid(signal: Signal, sample_rate: int, params: StftParams = StftParams()) -> float:
    """Energy-weighted mean frequency (Hz), averaged over frames — a brightness proxy."""
    magnitude, freqs = _magnitude_and_freqs(signal, sample_rate, params)
    total = np.sum(magnitude, axis=1)
    centroid = np.where(total > 0.0, (magnitude @ freqs) / np.maximum(total, _LOG_FLOOR), 0.0)
    return float(np.mean(centroid))


def spectral_rolloff(
    signal: Signal, sample_rate: int, roll_percent: float = 0.85, params: StftParams = StftParams()
) -> float:
    """Frequency (Hz) below which ``roll_percent`` of the energy lies, averaged over frames."""
    magnitude, freqs = _magnitude_and_freqs(signal, sample_rate, params)
    cumulative = np.cumsum(magnitude, axis=1)
    total = cumulative[:, -1]
    threshold = roll_percent * total[:, None]
    index = np.argmax(cumulative >= threshold, axis=1)
    rolloff = np.where(total > 0.0, freqs[index], 0.0)
    return float(np.mean(rolloff))


def spectral_flatness(signal: Signal, params: StftParams = StftParams()) -> float:
    """Geometric/arithmetic mean-power ratio, averaged over frames (1.0 ≈ noise, ~0 ≈ tonal)."""
    power = stft_magnitude(signal, params) ** 2 + _LOG_FLOOR
    geometric = np.exp(np.mean(np.log(power), axis=1))
    arithmetic = np.mean(power, axis=1)
    return float(np.mean(geometric / np.maximum(arithmetic, _LOG_FLOOR)))


def spectral_flux(signal: Signal, params: StftParams = StftParams()) -> Signal:
    """Per-frame L2 magnitude change ``(n_frames - 1,)`` — how fast the spectrum evolves."""
    magnitude = stft_magnitude(signal, params)
    if magnitude.shape[0] < 2:
        return np.zeros(1, dtype=np.float64)
    return np.sqrt(np.sum(np.diff(magnitude, axis=0) ** 2, axis=1)).astype(np.float64)


def band_energy(signal: Signal, sample_rate: int, f_low: float, f_high: float) -> float:
    """Total spectral energy in ``[f_low, f_high)`` (whole-signal FFT, Parseval-proportional)."""
    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def bandlimit(signal: Signal, sample_rate: int, f_low: float, f_high: float) -> Signal:
    """Zero every frequency outside ``[f_low, f_high)`` and return the time-domain signal."""
    length = signal.size
    spectrum = np.fft.rfft(np.asarray(signal, dtype=np.float64))
    freqs = np.fft.rfftfreq(length, 1.0 / sample_rate)
    spectrum = spectrum * ((freqs >= f_low) & (freqs < f_high))
    return np.fft.irfft(spectrum, n=length).astype(np.float64)
