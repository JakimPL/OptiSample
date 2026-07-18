from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisample.io.it_read import RippedSample, _parse_name, read_it_samples
from optisample.io.it_writer import (
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITSample,
    identity_note_map,
    write_it,
)


def _module() -> ITModule:
    time = np.arange(400) / 400.0
    sine = 0.8 * np.sin(2 * np.pi * 3 * time)
    ramp = np.linspace(-0.7, 0.7, 240, dtype=np.float64)
    samples = (
        ITSample(name="sine", pcm=sine, depth_bits=16, c5speed=20_000),
        ITSample(name="ramp", pcm=ramp, depth_bits=8, c5speed=8_000),
    )
    instrument = ITInstrument(name="i", note_map=identity_note_map({60: 1, 62: 2}))
    pattern = ITPattern(rows=4, cells=((0, 0, ITCell(note=60, instrument=1)),))
    return ITModule(name="rt", samples=samples, instruments=(instrument,), patterns=(pattern,), orders=(0,))


def test_parse_name_reads_index_prefix_and_falls_back() -> None:
    assert _parse_name("01 - piano C4") == (1, "piano C4")
    assert _parse_name("bare-name") == (0, "bare-name")  # no numeric prefix


def test_read_it_samples_recovers_every_sample(tmp_path: Path) -> None:
    pytest.importorskip("xmodits")
    module = _module()
    it_path = tmp_path / "rt.it"
    write_it(it_path, module)

    ripped = read_it_samples(it_path)
    assert [sample.index for sample in ripped] == [1, 2]  # ordered by index
    assert all(isinstance(sample, RippedSample) for sample in ripped)
    for original, recovered in zip(module.samples, ripped):
        assert recovered.frames == original.frames
        assert recovered.sample_rate == original.c5speed  # xmodits tags each WAV with the C5Speed
        tolerance = 0.02 if original.depth_bits == 16 else 0.05
        assert np.max(np.abs(recovered.pcm - np.asarray(original.pcm))) < tolerance
