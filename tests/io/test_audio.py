from __future__ import annotations

from pathlib import Path

import numpy as np

from optisample.io.audio import read_wav, write_wav


def test_wav_round_trip(tmp_path: Path) -> None:
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    signal = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    path = tmp_path / "tone.wav"

    write_wav(path, signal, sample_rate)
    restored, restored_rate = read_wav(path)

    assert restored_rate == sample_rate
    assert restored.shape == signal.shape
    # 32-bit float WAV is effectively lossless for this range.
    np.testing.assert_allclose(restored, signal, atol=1e-6)


def test_read_returns_float64_mono(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    write_wav(path, np.zeros(128, dtype=np.float64), 8_000)
    data, _ = read_wav(path)
    assert data.dtype == np.float64
    assert data.ndim == 1
