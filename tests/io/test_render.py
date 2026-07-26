from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from optisample.config.render import RenderConfig
from optisample.io import render as render_mod
from optisample.io.render import filter_taps, openmpt123_available, render_file, render_module
from trackmod.core.samples.depth import BitDepth
from trackmod.core.samples.sample import Sample
from trackmod.module.protocol import TrackerModule

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

_RATE = 22_050


@pytest.fixture
def module(probe_module: Callable[[Sequence[Sample]], TrackerModule], tone: Callable[..., np.ndarray]) -> TrackerModule:
    """A one-note module carrying a short tone, small enough to render in a test."""
    return probe_module([Sample(name="tone", pcm=tone(), rate=_RATE, depth=BitDepth.SIXTEEN)])


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module: TrackerModule, render_config: RenderConfig
) -> None:
    path = tmp_path / f"x{module.extension}"
    module.save(path)
    monkeypatch.setattr(render_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        render_file(path, render_config)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        render_module(module, render_config)


def test_render_failure_surfaces_stderr(
    monkeypatch: pytest.MonkeyPatch, module: TrackerModule, render_config: RenderConfig
) -> None:
    monkeypatch.setattr(render_mod.shutil, "which", lambda _name: "/usr/bin/openmpt123")

    def _fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(render_mod.subprocess, "run", _fail)
    with pytest.raises(RuntimeError, match="render failed .*boom"):
        render_module(module, render_config)


@requires_openmpt
def test_render_module_produces_mono_audio_at_requested_rate(
    module: TrackerModule, render_config: RenderConfig
) -> None:
    audio, rate = render_module(module, render_config.model_copy(update={"sample_rate": 44_100}))
    assert rate == 44_100
    assert audio.ndim == 1  # mono
    assert audio.size > 0
    assert np.max(np.abs(audio)) > 0.0  # the note actually sounded


@requires_openmpt
def test_render_file_reads_a_file_and_matches_render_module(
    tmp_path: Path, module: TrackerModule, render_config: RenderConfig
) -> None:
    path = tmp_path / f"probe{module.extension}"
    module.save(path)
    from_file, rate_file = render_file(path, render_config)
    from_module, rate_module = render_module(module, render_config)
    assert rate_file == rate_module
    np.testing.assert_array_equal(from_file, from_module)  # same bytes in -> same audio out


@requires_openmpt
def test_render_leaves_no_files_beside_the_source(
    tmp_path: Path, module: TrackerModule, render_config: RenderConfig
) -> None:
    path = tmp_path / f"probe{module.extension}"
    module.save(path)
    render_file(path, render_config)
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]  # the .wav stayed in the temp dir
