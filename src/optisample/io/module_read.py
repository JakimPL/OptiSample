import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xmodits
from numpy.typing import NDArray

from optisample.io.audio import read_wav


@dataclass(frozen=True)
class RippedSample:
    """One sample recovered from a module file: its index, name, tagged rate and float PCM."""

    index: int
    name: str
    sample_rate: int
    pcm: NDArray[np.float64]

    @property
    def frames(self) -> int:
        return int(self.pcm.size)


def _parse_name(stem: str) -> tuple[int, str]:
    """Split xmodits' ``"NN - name"`` filename stem into ``(index, name)`` (index 0 if unprefixed)."""
    head, separator, tail = stem.partition(" - ")
    if separator and head.strip().isdigit():
        return int(head), tail

    return 0, stem


def read_module_samples(path: Path | str) -> list[RippedSample]:
    """Extract every sample from ``path`` with xmodits and return them as float PCM, ordered by index.

    xmodits recognises the module format from the file itself, so this reads whichever format the
    exporter wrote and gives an account of the stored samples independent of the writer that made them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp)
        xmodits.dump(str(Path(path)), str(destination), format="wav")
        ripped: list[RippedSample] = []
        for wav in sorted(destination.glob("*.wav")):
            data, rate = read_wav(wav)
            index, name = _parse_name(wav.stem)
            pcm = np.asarray(data, dtype=np.float64).ravel()
            ripped.append(
                RippedSample(
                    index=index,
                    name=name,
                    sample_rate=rate,
                    pcm=pcm,
                )
            )

    return sorted(ripped, key=lambda sample: sample.index)
