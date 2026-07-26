import io

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

Signal = NDArray[np.float64]

_DEFAULT_PEAK = 0.95


def to_wav_bytes(
    signal: Signal,
    sample_rate: int,
    *,
    normalize: bool = True,
    peak: float = _DEFAULT_PEAK,
) -> bytes:
    """Encode ``signal`` as 16-bit PCM WAV bytes (peak-normalized unless ``normalize=False``)."""
    data = np.asarray(signal, dtype=np.float64)
    if normalize and data.size:
        largest = float(np.max(np.abs(data)))
        if largest > 0.0:
            data = data / largest * peak

    buffer = io.BytesIO()
    sf.write(buffer, np.clip(data, -1.0, 1.0), int(sample_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()
