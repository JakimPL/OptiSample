from typing import Final

import numpy as np
import pyloudnorm as pyln

from optisample.dsp.levels import db_to_gain
from optisample.metrics.base import MetricContext, Signal

_MIN_LOUDNESS_SECONDS: Final = 0.4  # BS.1770 integrated-loudness block size


def match_length(reference: Signal, candidate: Signal) -> tuple[Signal, Signal]:
    """Truncate both signals to their shared length."""
    length = min(reference.size, candidate.size)
    return reference[:length], candidate[:length]


def integrated_loudness(signal: Signal, sample_rate: int) -> float:
    """Integrated loudness (LUFS). Falls back to RMS-dBFS for clips shorter than the BS.1770 block."""
    data = np.asarray(signal, dtype=np.float64)
    if data.size >= int(_MIN_LOUDNESS_SECONDS * sample_rate) + 1:
        return float(pyln.Meter(sample_rate).integrated_loudness(data))
    rms = float(np.sqrt(np.mean(data**2))) if data.size else 0.0
    return -np.inf if rms <= 0.0 else 20.0 * float(np.log10(rms))


def loudness_normalize(
    signal: Signal,
    sample_rate: int,
    target_lufs: float,
) -> Signal:
    """Scale ``signal`` to ``target_lufs`` (no-op for silence)."""
    loudness = integrated_loudness(signal, sample_rate)
    if not np.isfinite(loudness):
        return np.asarray(signal, dtype=np.float64)

    gain = db_to_gain(target_lufs - loudness)
    return np.asarray(signal * gain, dtype=np.float64)


def prepare(
    reference: Signal,
    candidate: Signal,
    sample_rate: int,
    target_lufs: float,
    *,
    normalize: bool = True,
) -> tuple[Signal, Signal, MetricContext]:
    """Length-match (and optionally loudness-normalize) two signals for comparison."""
    reference, candidate = match_length(reference, candidate)
    if normalize:
        reference = loudness_normalize(reference, sample_rate, target_lufs)
        candidate = loudness_normalize(candidate, sample_rate, target_lufs)

    return (
        reference,
        candidate,
        MetricContext(
            sample_rate=sample_rate,
            normalized=normalize,
        ),
    )
