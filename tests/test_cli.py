from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisample.cli import _dump_settings, build_parser, main
from optisample.config import OptiConfig, load_config
from optisample.io.audio import write_wav
from optisample.io.manifest import dump_manifest
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample
from optisample.synth import NoteSpec, render_sample

SR = 44_100
PITCHES = (60, 62, 64)

# render_sample is only a test-signal generator here, so its synth config is fixture-independent data.
SYNTH = load_config().synth


def _tiny_manifest(tmp_path: Path) -> Path:
    """A minimal on-disk project: three piano notes + a manifest that references them."""
    samples = []
    for pitch in PITCHES:
        rel = Path(f"p{pitch}.wav")
        signal = render_sample("piano", NoteSpec(pitch, 100, 0.0, 0.6, SR), np.random.default_rng(pitch), SYNTH)
        write_wav(tmp_path / rel, signal, SR)
        samples.append(SourceSample(file=rel, pitch=pitch, velocity=100))
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=2) for pitch in PITCHES]
    instrument = InstrumentSpec(id="piano", budget_kb=48.0, samples=samples, material=material)
    manifest_path = tmp_path / "manifest.yaml"
    dump_manifest(Manifest(project=ProjectSpec(name="demo"), instruments=[instrument]), manifest_path)
    return manifest_path


def test_dump_settings_maps_grid_and_flags(config: OptiConfig) -> None:
    args = build_parser().parse_args(
        [
            "optimize",
            "m.yaml",
            "--rate",
            "11025",
            "--depth",
            "8",
            "--no-render",
            "--strategy",
            "grouped",
            "--no-loop",
            "--seed",
            "3",
        ]
    )
    settings = _dump_settings(config, args)
    assert settings.optimize.sweep.rates == (11_025,)
    assert settings.optimize.sweep.depths == (8,)
    assert settings.optimize.sweep.loops == (False,)  # --no-loop disables looping
    assert settings.optimize.seed == 3
    assert settings.render_ground_truth is False
    assert settings.grouped is True and settings.ungrouped is False


def test_dump_settings_defaults_to_looping_full_grid_both_strategies(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.yaml"])
    settings = _dump_settings(config, args)
    assert settings.grouped and settings.ungrouped and settings.render_ground_truth
    assert settings.optimize.sweep.loops == (True,)  # looping is on by default


def test_optimize_command_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = _tiny_manifest(tmp_path)
    out = tmp_path / "artifacts"
    main(["optimize", str(manifest_path), "--out", str(out), "--rate", "11025", "--depth", "8", "--no-render"])
    assert (out / "piano" / "grouped" / "module.it").is_file()
    assert (out / "piano" / "ungrouped" / "plan.json").is_file()
    printed = capsys.readouterr().out
    assert "piano" in printed and "objective" in printed


def test_optimize_command_honors_single_strategy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = _tiny_manifest(tmp_path)
    out = tmp_path / "artifacts"
    main(
        [
            "optimize",
            str(manifest_path),
            "--out",
            str(out),
            "--rate",
            "11025",
            "--depth",
            "8",
            "--no-render",
            "--strategy",
            "ungrouped",
        ]
    )
    assert (out / "piano" / "ungrouped").is_dir()
    assert not (out / "piano" / "grouped").exists()  # --strategy ungrouped skips the grouped run
    assert "ungrouped: objective" in capsys.readouterr().out


def test_synth_command_generates_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["synth", str(tmp_path / "demo"), "--sample-rate", "22050"])
    assert (tmp_path / "demo" / "manifest.yaml").is_file()
    assert "manifest" in capsys.readouterr().out.lower()
