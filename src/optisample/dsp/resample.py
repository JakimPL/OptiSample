import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample

Signal = NDArray[np.float64]


def resample_num(signal: Signal, num: int) -> Signal:
    """Resample to exactly ``num`` samples (band-limited). ``num <= 0`` -> empty; identity if unchanged."""
    data = np.asarray(signal, dtype=np.float64)
    if num <= 0:
        return np.zeros(0, dtype=np.float64)

    if data.size == 0:
        return np.zeros(num, dtype=np.float64)

    if num == data.size:
        return data.copy()

    return np.asarray(resample(data, num), dtype=np.float64)


def resample_to(signal: Signal, orig_rate: int, target_rate: int) -> Signal:
    """Convert ``signal`` from ``orig_rate`` to ``target_rate`` preserving duration and pitch."""
    if orig_rate <= 0 or target_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {orig_rate} -> {target_rate}")

    data = np.asarray(signal, dtype=np.float64)
    if target_rate == orig_rate:
        return data.copy()

    return resample_num(data, round(data.size * target_rate / orig_rate))


def resampled_frame_count(orig_frames: int, orig_rate: int, target_rate: int) -> int:
    """Frame count after converting ``orig_frames`` from ``orig_rate`` to ``target_rate``."""
    if orig_rate <= 0 or target_rate <= 0:
        raise ValueError(f"sample rates must be positive, got {orig_rate} -> {target_rate}")

    return round(orig_frames * target_rate / orig_rate)
