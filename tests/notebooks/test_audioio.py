import io
from collections.abc import Callable

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from notebooks.utils import audioio

SR = 44_100


def test_to_wav_bytes_roundtrips_and_normalizes(tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone(amplitude=0.05)  # quiet input
    data = audioio.to_wav_bytes(signal, SR)
    assert data[:4] == b"RIFF"
    back, sample_rate = sf.read(io.BytesIO(data), dtype="float64")
    assert sample_rate == SR
    assert back.shape[0] == signal.size
    assert float(np.max(np.abs(back))) == pytest.approx(0.95, abs=0.02)


def test_to_wav_bytes_without_normalization_keeps_level(tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone(amplitude=0.2)
    back, _ = sf.read(io.BytesIO(audioio.to_wav_bytes(signal, SR, normalize=False)), dtype="float64")
    assert float(np.max(np.abs(back))) == pytest.approx(0.2, abs=0.01)
