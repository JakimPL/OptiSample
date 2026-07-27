from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.cli import _dump_settings, build_parser, main
from optisample.config import OptiConfig
from optisample.config.reduce import DedupeKey
from optisample.config.tracker import TrackerFormat
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NoteRecord, dump_notes

SR = 44_100
PITCHES = (60, 62, 64)


@pytest.fixture
def tiny_notes(tmp_path: Path, piano_note: Callable[..., NDArray[np.float64]]) -> Path:
    """A minimal on-disk NoteExtractor project: three piano notes + a .notes.json referencing them.

    The WAVs live in the sibling ``piano/`` directory the ``optimize`` command resolves by default, so the
    instrument id defaults to ``piano`` and its artifacts land under ``<out>/piano/``.
    """
    samples_dir = tmp_path / "piano"
    samples_dir.mkdir()
    records = []
    for index, pitch in enumerate(PITCHES):
        write_wav(samples_dir / f"{index:04d}_p{pitch}_v100.wav", piano_note(pitch, 100, 0.6, seed=pitch), SR)
        records.append(NoteRecord(index=index, pitch=pitch, velocity=100, duration_s=0.5))
    notes_json = tmp_path / "piano.notes.json"
    dump_notes(records, notes_json)
    return notes_json


def test_dump_settings_maps_grid_and_flags(config: OptiConfig) -> None:
    args = build_parser().parse_args(
        [
            "optimize",
            "m.notes.json",
            "--budget-kb",
            "48",
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
    assert settings.optimize.target.format is config.tracker.format  # unnamed, so the configured format
    assert settings.render_ground_truth is False
    assert settings.grouped is True and settings.ungrouped is False


def test_the_format_flag_overrides_the_configured_format(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--format", "xm"])
    target = _dump_settings(config, args).optimize.target
    assert target.format is TrackerFormat.XM
    assert target.compliance is config.tracker.compliance  # only the format is overridden
    assert target.it.global_volume == config.tracker.it.global_volume


def test_the_reduction_flags_override_their_configured_sections(config: OptiConfig) -> None:
    args = build_parser().parse_args(
        [
            "optimize",
            "m.notes.json",
            "--budget-kb",
            "48",
            "--dedupe-key",
            "pitch",
            "--candidates",
            "7",
        ]
    )
    reduce = _dump_settings(config, args).optimize.reduce
    assert reduce.dedupe.key is DedupeKey.PITCH
    assert reduce.bandwidth.candidates == 7
    assert reduce.events == config.reduce.events  # only the named sections move
    assert reduce.dedupe.cc_quantum == config.reduce.dedupe.cc_quantum


def test_the_reduction_sections_stay_configured_when_no_flag_names_them(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert _dump_settings(config, args).optimize.reduce == config.reduce


def test_optimize_command_writes_artifacts(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "artifacts"
    main(
        [
            "optimize",
            str(tiny_notes),
            "--budget-kb",
            "48",
            "--out",
            str(out),
            "--rate",
            "11025",
            "--depth",
            "8",
            "--no-render",
        ]
    )
    assert (out / "piano" / "grouped" / "module.it").is_file()
    assert (out / "piano" / "ungrouped" / "plan.json").is_file()
    printed = capsys.readouterr().out
    assert "piano" in printed and "objective" in printed


def test_optimize_command_writes_the_format_it_was_asked_for(tmp_path: Path, tiny_notes: Path) -> None:
    out = tmp_path / "artifacts"
    main(
        [
            "optimize",
            str(tiny_notes),
            "--budget-kb",
            "48",
            "--out",
            str(out),
            "--format",
            "xm",
            "--rate",
            "11025",
            "--depth",
            "8",
            "--no-render",
            "--strategy",
            "ungrouped",
        ]
    )
    assert (out / "piano" / "ungrouped" / "module.xm").is_file()


def test_optimize_command_reports_timing(tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "artifacts"
    main(
        [
            "optimize",
            str(tiny_notes),
            "--budget-kb",
            "48",
            "--out",
            str(out),
            "--rate",
            "11025",
            "--depth",
            "8",
            "--no-render",
        ]
    )
    printed = capsys.readouterr().out
    assert "s]" in printed  # each strategy line carries its wall-clock, e.g. "[0.4s]"
    assert "total:" in printed


def test_optimize_command_profile_flag_still_writes_and_profiles(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "artifacts"
    main(
        [
            "optimize",
            str(tiny_notes),
            "--budget-kb",
            "48",
            "--out",
            str(out),
            "--rate",
            "11025",
            "--depth",
            "8",
            "--no-render",
            "--profile",
        ]
    )
    captured = capsys.readouterr()
    assert (out / "piano" / "ungrouped" / "plan.json").is_file()  # profiling does not change the run
    assert "function calls" in captured.err  # cProfile's report went to stderr


def test_optimize_command_honors_single_strategy(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "artifacts"
    main(
        [
            "optimize",
            str(tiny_notes),
            "--budget-kb",
            "48",
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


def test_synth_command_generates_notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["synth", str(tmp_path / "demo"), "--sample-rate", "22050"])
    assert sorted((tmp_path / "demo").glob("*.notes.json"))
    assert "notes.json" in capsys.readouterr().out.lower()
