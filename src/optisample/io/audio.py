from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray


@dataclass(frozen=True)
class WavInfo:
    """What a WAV's header states about its contents: how many frames it holds, and at what rate."""

    frames: int
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.sample_rate


def probe_wav(path: Path | str) -> WavInfo:
    """Read a WAV's frame count and rate from its header, leaving the PCM on disk.

    Ranking candidate recordings by length is a per-file question asked of every duplicate in the
    recorded grid, so the header alone answers it at a fraction of what decoding each file would cost.
    """
    info = sf.info(str(path))
    return WavInfo(frames=int(info.frames), sample_rate=int(info.samplerate))


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


def mono(samples: NDArray[np.floating]) -> NDArray[np.float64]:
    """``samples`` as the single channel every stage reads: a multi-channel recording averaged across its own.

    A tracker sounds one waveform per voice, so a recording reaches the pipeline as one channel however
    many it was captured on, and taking the mean keeps whatever both channels hold in common at the level
    they hold it. A recording already on one channel is answered as it stands.
    """
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim > 1:
        return np.asarray(np.mean(data, axis=1), dtype=np.float64)

    return data
