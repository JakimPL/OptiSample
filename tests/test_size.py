from __future__ import annotations

import pytest

from optisample.metrics.size import (
    FILE_HEADER_BYTES,
    INSTRUMENT_HEADER_BYTES,
    SAMPLE_HEADER_BYTES,
    SampleSize,
    bytes_to_kib,
    kib_to_bytes,
    max_frames_for_budget,
    module_bytes,
)


def test_pcm_and_total_bytes() -> None:
    sample = SampleSize(frames=1000, depth_bits=16)
    assert sample.pcm_bytes == 2000
    assert sample.total_bytes == 2000 + SAMPLE_HEADER_BYTES


def test_16bit_is_twice_8bit() -> None:
    assert SampleSize(1000, 16).pcm_bytes == 2 * SampleSize(1000, 8).pcm_bytes


def test_stereo_doubles_mono() -> None:
    assert SampleSize(1000, 16, channels=2).pcm_bytes == 2 * SampleSize(1000, 16, channels=1).pcm_bytes


@pytest.mark.parametrize("depth", [0, 24, 32])
def test_invalid_depth_rejected(depth: int) -> None:
    with pytest.raises(ValueError):
        _ = SampleSize(1000, depth).pcm_bytes


def test_invalid_channels_rejected() -> None:
    with pytest.raises(ValueError):
        _ = SampleSize(1000, 16, channels=0).pcm_bytes


def test_module_bytes_includes_overhead() -> None:
    samples = [SampleSize(1000, 16), SampleSize(500, 8)]
    expected = (
        FILE_HEADER_BYTES + 2 * INSTRUMENT_HEADER_BYTES + (2000 + SAMPLE_HEADER_BYTES) + (500 + SAMPLE_HEADER_BYTES)
    )
    assert module_bytes(samples, n_instruments=2) == expected


def test_max_frames_for_budget() -> None:
    # 2080 B budget, 16-bit mono → 2000 usable bytes → 1000 frames.
    assert max_frames_for_budget(2080, 16) == 1000
    # Header alone exceeds the budget → nothing fits.
    assert max_frames_for_budget(50, 16) == 0


def test_kib_round_trip() -> None:
    assert kib_to_bytes(128.0) == 131072
    assert bytes_to_kib(131072) == pytest.approx(128.0)
