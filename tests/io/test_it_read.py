from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from optisample.io.it_read import RippedSample, _parse_name, read_it_samples
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.module.protocol import TrackerModule

_SINE_FRAMES = 400
_RAMP_FRAMES = 240
_TOLERANCE = {BitDepth.SIXTEEN: 0.02, BitDepth.EIGHT: 0.05}


@pytest.fixture
def written_samples() -> tuple[Sample, ...]:
    """Two samples of different depth and rate, so the round trip has something to disagree about."""
    time = np.arange(_SINE_FRAMES) / _SINE_FRAMES
    sine = 0.8 * np.sin(2 * np.pi * 3 * time)
    ramp = np.linspace(-0.7, 0.7, _RAMP_FRAMES, dtype=np.float64)
    return (
        Sample(name="sine", pcm=sine, rate=20_000, depth=BitDepth.SIXTEEN),
        Sample(name="ramp", pcm=ramp, rate=8_000, depth=BitDepth.EIGHT),
    )


def test_parse_name_reads_index_prefix_and_falls_back() -> None:
    assert _parse_name("01 - piano C4") == (1, "piano C4")
    assert _parse_name("bare-name") == (0, "bare-name")  # no numeric prefix


def test_read_it_samples_recovers_every_sample(
    tmp_path: Path,
    probe_module: Callable[[Sequence[Sample]], TrackerModule],
    written_samples: tuple[Sample, ...],
) -> None:
    pytest.importorskip("xmodits")
    module = probe_module(written_samples)
    path = tmp_path / f"rt{module.extension}"
    module.save(path)

    ripped = read_it_samples(path)
    assert [sample.index for sample in ripped] == [1, 2]  # ordered by index
    assert all(isinstance(sample, RippedSample) for sample in ripped)
    for original, recovered in zip(written_samples, ripped):
        assert recovered.frames == original.frames
        assert recovered.sample_rate == original.rate  # the tagged rate is the rate the sample stores
        assert np.max(np.abs(recovered.pcm - np.asarray(original.pcm))) < _TOLERANCE[original.depth]
