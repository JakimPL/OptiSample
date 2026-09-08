from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from trackmod import BitDepth, Sample, load_voices
from trackmod.trackers.xm.spec.tuning import FINETUNE_UNITS

from optisample.config.tracker import TrackerFormat
from optisample.io.tracker.target import ExportTarget
from optisample.io.tracker.voices import routed
from tests.io.conftest import ProbeModule

_SINE_FRAMES = 400
_RAMP_FRAMES = 240
_SINE_CYCLES = 3
_CENTS_PER_SEMITONE = 100.0
_FINETUNE_CENTS = _CENTS_PER_SEMITONE / FINETUNE_UNITS  # the finest pitch step FastTracker 2 spells


@pytest.fixture
def written_samples() -> tuple[Sample, ...]:
    """Two samples of different depth and rate, so the round trip has something to disagree about."""
    time = np.arange(_SINE_FRAMES) / _SINE_FRAMES
    sine = 0.8 * np.sin(2 * np.pi * _SINE_CYCLES * time)
    ramp = np.linspace(-0.7, 0.7, _RAMP_FRAMES, dtype=np.float64)
    return (
        Sample(name="sine", pcm=sine, rate=20_000, depth=BitDepth.SIXTEEN),
        Sample(name="ramp", pcm=ramp, rate=8_000, depth=BitDepth.EIGHT),
    )


def recovered(
    samples: tuple[Sample, ...],
    tracker_format: TrackerFormat,
    tmp_path: Path,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
) -> tuple[Sample, ...]:
    """Write ``samples`` as ``tracker_format``, then read the file back off disk as the voices it holds."""
    module = probe_module(samples, retarget(tracker_format))
    path = tmp_path / f"rt{module.extension}"
    module.save(path)
    return routed(load_voices(path), held_by=f"module {path.name!r}").samples


@pytest.mark.parametrize("tracker_format", tuple(TrackerFormat))
def test_a_written_module_reads_back_as_the_samples_it_was_given(
    tracker_format: TrackerFormat,
    tmp_path: Path,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
    written_samples: tuple[Sample, ...],
) -> None:
    """Every field a stored sample carries survives the file, and its waveform lands on the stored grid.

    A written waveform is rounded to the depth it is stored at, so what comes back sits within one step
    of what went in -- which is the whole distance the file itself accounts for.
    """
    read_back = recovered(written_samples, tracker_format, tmp_path, retarget, probe_module)

    assert len(read_back) == len(written_samples)
    for original, sample in zip(written_samples, read_back):
        assert sample.name == original.name
        assert sample.frames == original.frames
        assert sample.depth == original.depth
        assert np.max(np.abs(np.asarray(sample.pcm) - np.asarray(original.pcm))) < 1.0 / original.depth.scale


def test_an_impulse_tracker_module_states_the_rate_each_sample_stores(
    tmp_path: Path,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
    written_samples: tuple[Sample, ...],
) -> None:
    """The rate is a per-sample field in this format, so a reader recovers it outright."""
    read_back = recovered(written_samples, TrackerFormat.IT, tmp_path, retarget, probe_module)
    assert [sample.rate for sample in read_back] == [sample.rate for sample in written_samples]


def test_a_fast_tracker_module_reaches_each_rate_through_the_key_it_transposes(
    tmp_path: Path,
    retarget: Callable[[TrackerFormat], ExportTarget],
    probe_module: ProbeModule,
    written_samples: tuple[Sample, ...],
) -> None:
    """This format spells pitch as a transposition of the reference key, on a lattice of its own.

    A rate is written as the semitones and the finetune steps that reach it, so the rate read back is the
    one that lattice can spell, which lands within a step of the rate the sample was given.
    """
    read_back = recovered(written_samples, TrackerFormat.XM, tmp_path, retarget, probe_module)
    for original, sample in zip(written_samples, read_back):
        drift_cents = _CENTS_PER_SEMITONE * np.log2(sample.rate / original.rate)
        assert abs(drift_cents) < _FINETUNE_CENTS
        assert (sample.relative_note, sample.finetune) != (0, 0)  # the transposition carries the rate
