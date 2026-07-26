from pathlib import Path

import numpy as np
import pytest

from optisample.io.audio import probe_wav, read_wav, write_wav


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


def test_probe_reports_the_length_the_header_states(tmp_path: Path) -> None:
    path = tmp_path / "half_second.wav"
    write_wav(path, np.zeros(11_025, dtype=np.float64), 22_050)

    info = probe_wav(path)

    assert info.frames == 11_025
    assert info.sample_rate == 22_050
    assert info.duration_s == pytest.approx(0.5)


def test_probe_agrees_with_reading_the_pcm(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    write_wav(path, np.zeros(1_234, dtype=np.float64), 16_000)

    data, rate = read_wav(path)
    info = probe_wav(path)

    assert (info.frames, info.sample_rate) == (data.size, rate)
