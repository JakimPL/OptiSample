from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray


def write_wav(
    path: Path | str,
    samples: NDArray[np.floating],
    sample_rate: int,
) -> None:
    """Write mono (1-D) or multi-channel (N, C) float samples to a 32-bit float WAV."""
    sf.write(str(path), samples, int(sample_rate), subtype="FLOAT")


def read_wav(path: Path | str) -> tuple[NDArray[np.float64], int]:
    """Read a WAV as float64 samples plus its sample rate (mono returned as 1-D)."""
    data, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    return np.asarray(data, dtype=np.float64), int(sample_rate)
