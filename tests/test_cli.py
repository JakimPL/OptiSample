import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from optisample.cli import (
    _demo_settings,
    _dump_settings,
    _optimize_settings,
    build_parser,
    main,
)
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


def test_the_max_layers_flag_overrides_the_configured_cap(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-layers", "1"])
    layers = _dump_settings(config, args).optimize.layers
    assert layers.max_layers == 1
    assert layers.nodes == config.layers.nodes  # only the cap moves
    assert layers.min_gain == config.layers.min_gain


def test_the_layer_cap_stays_configured_when_no_flag_names_it(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert _dump_settings(config, args).optimize.layers == config.layers


def test_a_layer_cap_below_one_is_refused_by_the_schema(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-layers", "0"])
    with pytest.raises(ValidationError):
        _dump_settings(config, args)


def test_the_max_samples_flag_overrides_the_configured_cap(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-samples", "8"])
    settings = _dump_settings(config, args).optimize
    assert settings.max_samples == 8
    assert settings.sample_cap == 8
    assert settings.method == config.optimize.method  # only the cap moves
    assert settings.energy_exponent == config.optimize.energy_exponent


def test_the_sample_cap_stays_configured_when_no_flag_names_it(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert _dump_settings(config, args).optimize.max_samples == config.optimize.max_samples


def test_a_sample_cap_below_zero_is_refused_by_the_schema(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-samples", "-1"])
    with pytest.raises(ValidationError):
        _dump_settings(config, args)


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


def test_reduce_command_writes_a_dataset_and_its_reduction(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "reduced"
    main(["reduce", str(tiny_notes), "--budget-kb", "48", "--out", str(out), "--rate", "11025", "--depth", "8"])
    assert (out / "piano.notes.json").is_file()
    assert sorted(path.name for path in (out / "piano").glob("*.wav"))
    assert (out / "reduction" / "piano" / "reduction.json").is_file()
    assert (out / "reduction" / "piano" / "auditions").is_dir()
    printed = capsys.readouterr().out
    assert "samples" in printed and "auditions" in printed


def test_reduce_reports_and_leaves_out_the_recordings_that_never_sound(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed render keeps no slot in the dataset, and the run says so rather than quietly shrinking."""
    silent = tmp_path / "piano" / f"0000_p{PITCHES[0]}_v100.wav"
    write_wav(silent, np.full(round(0.6 * SR), 1.0e-6), SR)
    out = tmp_path / "reduced"
    main(["reduce", str(tiny_notes), "--budget-kb", "48", "--out", str(out), "--rate", "11025", "--depth", "8"])
    written = sorted(path.name for path in (out / "piano").glob("*.wav"))
    assert len(written) == len(PITCHES) - 1
    assert "carried no signal" in capsys.readouterr().out
    document = json.loads((out / "reduction" / "piano" / "reduction.json").read_text(encoding="utf-8"))
    assert document["screen"]["unplayable"] == [PITCHES[0]]
    assert document["screen"]["dropped_notes"] == 1


def test_a_reduced_dataset_optimizes_to_the_same_plan_as_its_source(tmp_path: Path, tiny_notes: Path) -> None:
    """The round trip the reduce stage exists for: allocating from the dataset reaches the same plan."""
    reduced = tmp_path / "reduced"
    flags = ["--budget-kb", "48", "--rate", "11025", "--depth", "8"]
    main(["reduce", str(tiny_notes), *flags, "--out", str(reduced)])
    main(["optimize", str(tiny_notes), *flags, "--out", str(tmp_path / "direct"), "--no-render"])
    main(["optimize", str(reduced / "piano.notes.json"), *flags, "--out", str(tmp_path / "again"), "--no-render"])
    direct = (tmp_path / "direct" / "piano" / "ungrouped" / "plan.json").read_text(encoding="utf-8")
    again = (tmp_path / "again" / "piano" / "ungrouped" / "plan.json").read_text(encoding="utf-8")
    assert direct == again


def test_subset_command_writes_a_dataset_the_other_commands_read(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "subset"
    main(["subset", str(tiny_notes), "--fraction", "0.67", "--out", str(out)])

    assert (out / "piano.notes.json").is_file()
    assert len(list((out / "piano").glob("*.wav"))) == 2
    printed = capsys.readouterr().out
    assert "2 of 3 notes" in printed
    main(
        ["optimize", str(out / "piano.notes.json"), "--budget-kb", "48", "--out", str(tmp_path / "art"), "--no-render"]
    )
    assert (tmp_path / "art" / "piano" / "ungrouped" / "plan.json").is_file()


def test_the_subset_command_names_its_output_after_the_instrument(tmp_path: Path, tiny_notes: Path) -> None:
    main(["subset", str(tiny_notes), "--fraction", "1.0", "--instrument-id", "Grand", "--out", str(tmp_path / "s")])

    assert (tmp_path / "s" / "Grand.notes.json").is_file()
    assert len(list((tmp_path / "s" / "Grand").glob("*.wav"))) == len(PITCHES)


def test_the_reduce_command_reads_the_same_ingest_flags_as_optimize(config: OptiConfig) -> None:
    """Both commands share one ingest parser, so a reduction knob means the same thing to either."""
    argv = ["m.notes.json", "--budget-kb", "48", "--dedupe-key", "pitch", "--candidates", "7", "--seed", "3"]
    reduced = build_parser().parse_args(["reduce", *argv])
    optimized = build_parser().parse_args(["optimize", *argv])
    assert _optimize_settings(config, reduced, config.layers, config.optimize).reduce == (
        _optimize_settings(config, optimized, config.layers, config.optimize).reduce
    )
    assert _optimize_settings(config, reduced, config.layers, config.optimize).seed == 3


def test_the_workers_flag_sets_how_far_a_run_fans_out(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--workers", "3"])
    assert _optimize_settings(config, args, config.layers, config.optimize).workers == 3


def test_a_run_fans_out_the_configured_way_when_no_flag_names_a_count(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert _optimize_settings(config, args, config.layers, config.optimize).workers == config.runtime.workers


def test_the_demo_reads_the_same_fan_out_flag_as_a_run(config: OptiConfig) -> None:
    """One flag on the shared parser, so ``--workers 1`` means an in-process run to either command."""
    args = build_parser().parse_args(["synth", "demo", "--workers", "1"])
    assert _demo_settings(config, args).workers == 1


def test_synth_command_generates_notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["synth", str(tmp_path / "demo"), "--sample-rate", "22050"])
    assert sorted((tmp_path / "demo").glob("*.notes.json"))
    assert "notes.json" in capsys.readouterr().out.lower()
