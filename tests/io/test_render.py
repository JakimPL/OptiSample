from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from optisample.config.render import RenderConfig
from optisample.io import render as render_mod
from optisample.io.it_writer import (
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    identity_note_map,
    write_it,
)
from optisample.io.render import filter_taps, openmpt123_available, render_it, render_module

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")


def _module(playback: ITPlayback, frames: int = 4096, c5speed: int = 22_050) -> ITModule:
    time = np.arange(frames) / frames
    tone = 0.8 * np.sin(2 * np.pi * 40 * time)  # a handful of cycles across the sample
    sample = ITSample(name="tone", pcm=tone, depth_bits=16, c5speed=c5speed)
    instrument = ITInstrument(name="i", note_map=identity_note_map({60: 1}))
    pattern = ITPattern(rows=16, cells=((0, 0, ITCell(note=60, instrument=1, volume=64)), (12, 0, ITCell(note=254))))
    return ITModule(
        name="probe", samples=(sample,), instruments=(instrument,), patterns=(pattern,), orders=(0,), playback=playback
    )


def test_filter_taps_maps_each_interpolation(render_config: RenderConfig) -> None:
    taps = {
        name: filter_taps(render_config.model_copy(update={"interpolation": name}))
        for name in ("none", "linear", "cubic", "sinc")
    }
    assert taps == {"none": 1, "linear": 2, "cubic": 4, "sinc": 8}


def test_render_config_rejects_unknown_interpolation() -> None:
    with pytest.raises(ValidationError):
        RenderConfig(sample_rate=48_000, interpolation="bogus", gain_db=0.0)  # type: ignore[arg-type]


def test_openmpt123_available_returns_bool() -> None:
    assert isinstance(openmpt123_available(), bool)


def test_missing_binary_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, playback: ITPlayback, render_config: RenderConfig
) -> None:
    monkeypatch.setattr(render_mod.shutil, "which", lambda _name: None)
    it_path = tmp_path / "x.it"
    write_it(it_path, _module(playback))
    with pytest.raises(RuntimeError, match="not found on PATH"):
        render_it(it_path, render_config)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        render_module(_module(playback), render_config)


def test_render_failure_surfaces_stderr(
    monkeypatch: pytest.MonkeyPatch, playback: ITPlayback, render_config: RenderConfig
) -> None:
    monkeypatch.setattr(render_mod.shutil, "which", lambda _name: "/usr/bin/openmpt123")

    def _fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(render_mod.subprocess, "run", _fail)
    with pytest.raises(RuntimeError, match="render failed .*boom"):
        render_module(_module(playback), render_config)


@requires_openmpt
def test_render_module_produces_mono_audio_at_requested_rate(playback: ITPlayback, render_config: RenderConfig) -> None:
    audio, rate = render_module(_module(playback), render_config.model_copy(update={"sample_rate": 44_100}))
    assert rate == 44_100
    assert audio.ndim == 1  # mono
    assert audio.size > 0
    assert np.max(np.abs(audio)) > 0.0  # the note actually sounded


@requires_openmpt
def test_render_it_reads_a_file_and_matches_render_module(
    tmp_path: Path, playback: ITPlayback, render_config: RenderConfig
) -> None:
    module = _module(playback)
    it_path = tmp_path / "probe.it"
    write_it(it_path, module)
    from_file, rate_file = render_it(it_path, render_config)
    from_module, rate_module = render_module(module, render_config)
    assert rate_file == rate_module
    np.testing.assert_array_equal(from_file, from_module)  # same bytes in -> same audio out


@requires_openmpt
def test_render_leaves_no_files_beside_the_source(
    tmp_path: Path, playback: ITPlayback, render_config: RenderConfig
) -> None:
    it_path = tmp_path / "probe.it"
    write_it(it_path, _module(playback))
    render_it(it_path, render_config)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["probe.it"]  # rendered .wav stayed in the temp dir
