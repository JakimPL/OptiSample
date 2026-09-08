from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from trackmod import BitDepth, Sample

from optisample.config.tracker import TrackerFormat
from optisample.io.module_read import RippedSample, _parse_name, read_module_samples
from optisample.io.tracker.target import ExportTarget
from tests.io.conftest import ProbeModule

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


def written(
    samples: tuple[Sample, ...],
    tracker_format: TrackerFormat,
    tmp_path: Path,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
) -> list[RippedSample]:
    """Write ``samples`` as ``tracker_format`` and rip them back out with the independent parser."""
    module = probe_module(samples, retarget(tracker_format))
    path = tmp_path / f"rt{module.extension}"
    module.save(path)
    return read_module_samples(path)


@pytest.mark.parametrize("tracker_format", tuple(TrackerFormat))
def test_read_module_samples_recovers_every_stored_waveform(
    tracker_format: TrackerFormat,
    tmp_path: Path,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
    written_samples: tuple[Sample, ...],
) -> None:
    pytest.importorskip("xmodits")
    ripped = written(written_samples, tracker_format, tmp_path, retarget, probe_module)
    assert [sample.index for sample in ripped] == [1, 2]  # ordered by index
    assert all(isinstance(sample, RippedSample) for sample in ripped)
    for original, recovered in zip(written_samples, ripped):
        assert recovered.name == original.name
        assert recovered.frames == original.frames
        assert np.max(np.abs(recovered.pcm - np.asarray(original.pcm))) < _TOLERANCE[original.depth]


def test_an_impulse_tracker_module_tags_each_sample_with_the_rate_it_stores(
    tmp_path: Path,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
    written_samples: tuple[Sample, ...],
) -> None:
    """The rate is a per-sample field in this format, so a reader recovers it outright.

    FastTracker 2 expresses pitch as a transposition of the key instead, and a reader that reports a
    rate for one of its samples is quoting what it would sound at that format's own reference key.
    """
    pytest.importorskip("xmodits")
    ripped = written(written_samples, TrackerFormat.IT, tmp_path, retarget, probe_module)
    assert [sample.sample_rate for sample in ripped] == [sample.rate for sample in written_samples]
