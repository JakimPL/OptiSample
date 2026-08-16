import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from optisample.artifacts.paths import instrument_files_dir
from optisample.artifacts.pipeline import PipelineStage
from optisample.calibrate.ranking import Fault, Verdict
from optisample.cli import main
from optisample.cli.parsers import build_parser
from optisample.cli.settings import (
    _demo_settings,
    _dump_settings,
    _ingest_settings,
    _optimize_settings,
    _pipeline_settings,
)
from optisample.config import OptiConfig
from optisample.config.reduce import DedupeKey
from optisample.config.tracker import TrackerFormat
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NoteRecord, dump_notes
from tests.conftest import TEST_CONFIG_DIR


def run(argv: list[str]) -> None:
    """One command carried out under the suite's own settings (:data:`~tests.conftest.TEST_CONFIG_DIR`).

    Every command loads its configured values from the directory ``--config`` names, so pointing each run
    at the repository's test settings is what makes the artifacts a test reads back reproduce whatever the
    bundled ``opticonfig`` currently ships. A test about a configured knob states it through the command's
    own flag, which leaves the flag beside the assertion it moves.
    """
    main([*argv, "--config", str(TEST_CONFIG_DIR)])


SR = 44_100
PITCHES = (60, 62, 64)
_HELD_S = 1.5  # past the length a listening set asks about, so the set has material to draw on
_SOUNDS_S = 0.75  # a note ringing long enough for a slice to draw on it
_BRIEF_S = 0.25  # a note ringing for less than the floor a slice is asked for
_ADMITS_BOTH = 0.5  # a floor standing between the two, so a ragged source loses exactly its short notes


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


@pytest.fixture
def ragged_notes(tmp_path: Path, piano_note: Callable[..., NDArray[np.float64]]) -> Path:
    """The same project with its first note ringing for less than a slice draws on."""
    samples_dir = tmp_path / "piano"
    samples_dir.mkdir()
    records = []
    for index, pitch in enumerate(PITCHES):
        write_wav(samples_dir / f"{index:04d}_p{pitch}_v100.wav", piano_note(pitch, 100, 0.6, seed=pitch), SR)
        sounds_s = _BRIEF_S if index == 0 else _SOUNDS_S
        records.append(NoteRecord(index=index, pitch=pitch, velocity=100, duration_s=sounds_s))

    notes_json = tmp_path / "piano.notes.json"
    dump_notes(records, notes_json)
    return notes_json


@pytest.fixture
def held_notes(tmp_path: Path, piano_note: Callable[..., NDArray[np.float64]]) -> Path:
    """The same project over notes held long enough for a listening set to be asked about them.

    A listening set leaves the short material unpriced, since a degradation shows itself over a decay, so
    a project exercising it plays past that floor.
    """
    samples_dir = tmp_path / "piano"
    samples_dir.mkdir()
    records = []
    for index, pitch in enumerate(PITCHES):
        write_wav(samples_dir / f"{index:04d}_p{pitch}_v100.wav", piano_note(pitch, 100, _HELD_S, seed=pitch), SR)
        records.append(NoteRecord(index=index, pitch=pitch, velocity=100, duration_s=_HELD_S))
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
    assert settings.optimize.sweep.depth == 8
    assert settings.optimize.loops is False  # --no-loop leaves the loop stage out of the run
    assert settings.optimize.seed == 3
    assert settings.optimize.target.format is config.export.tracker.format  # unnamed, so the configured format
    assert settings.render_ground_truth is False
    assert settings.grouped is True and settings.ungrouped is False


def test_the_format_flag_overrides_the_configured_format(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--format", "xm"])
    target = _dump_settings(config, args).optimize.target
    assert target.format is TrackerFormat.XM
    assert target.compliance is config.export.tracker.compliance  # only the format is overridden
    assert target.it.global_volume == config.export.tracker.it.global_volume


def test_the_reduction_flags_override_their_configured_sections(config: OptiConfig) -> None:
    args = build_parser().parse_args(
        [
            "optimize",
            "m.notes.json",
            "--budget-kb",
            "48",
            "--dedupe-key",
            "pitch",
            "--content-floor-db",
            "45",
        ]
    )
    reduce = _dump_settings(config, args).optimize.reduce
    assert reduce.dedupe.key is DedupeKey.PITCH
    assert reduce.bandwidth.content_floor_db == 45.0
    assert reduce.events == config.reduce.events  # only the named sections move
    assert reduce.dedupe.cc_quantum == config.reduce.dedupe.cc_quantum


def test_the_reduction_sections_stay_configured_when_no_flag_names_them(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert _dump_settings(config, args).optimize.reduce == config.reduce


def test_the_max_layers_flag_overrides_the_configured_cap(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-layers", "1"])
    layers = _dump_settings(config, args).optimize.layers
    assert layers.max_layers == 1
    assert layers.nodes == config.optimize.layers.nodes  # only the cap moves
    assert layers.min_gain == config.optimize.layers.min_gain


def test_the_layer_cap_stays_configured_when_no_flag_names_it(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert _dump_settings(config, args).optimize.layers == config.optimize.layers


def test_a_layer_cap_below_one_is_refused_by_the_schema(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-layers", "0"])
    with pytest.raises(ValidationError):
        _dump_settings(config, args)


def test_the_max_samples_flag_overrides_the_configured_cap(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-samples", "8"])
    settings = _dump_settings(config, args).optimize
    assert settings.max_samples == 8
    assert settings.sample_cap == 8
    assert settings.method == config.optimize.budget.method  # only the cap moves
    assert settings.energy_exponent == config.optimize.budget.energy_exponent


def test_the_sample_cap_stays_configured_when_no_flag_names_it(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert _dump_settings(config, args).optimize.max_samples == config.optimize.budget.max_samples


def test_a_sample_cap_below_zero_is_refused_by_the_schema(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--max-samples", "-1"])
    with pytest.raises(ValidationError):
        _dump_settings(config, args)


def test_optimize_command_writes_artifacts(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "artifacts"
    run(
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
    assert (out / "piano" / "grouped" / "piano.bank").is_file()
    assert (out / "piano" / "ungrouped" / "plan.json").is_file()
    printed = capsys.readouterr().out
    assert "piano" in printed and "objective" in printed


def test_optimize_command_writes_the_format_it_was_asked_for(tmp_path: Path, tiny_notes: Path) -> None:
    out = tmp_path / "artifacts"
    run(
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
    assert [path.suffix for path in sorted((out / "piano" / "ungrouped" / "instruments").iterdir())] == [".xi"]


def test_optimize_command_reports_timing(tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "artifacts"
    run(
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
    run(
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
    run(
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
    run(["reduce", str(tiny_notes), "--budget-kb", "48", "--out", str(out), "--rate", "11025", "--depth", "8"])
    assert (out / "piano.notes.json").is_file()
    assert sorted(path.name for path in (out / "piano").glob("*.wav"))
    assert (out / "reduction" / "piano" / "reduction.json").is_file()
    assert (out / "reduction" / "piano" / "auditions").is_dir()
    printed = capsys.readouterr().out
    assert "samples" in printed and "auditions" in printed


def test_listen_command_writes_a_blinded_set_and_its_answer_sheet(
    tmp_path: Path, held_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "listening"
    run(["listen", str(held_notes), "--budget-kb", "48", "--out", str(out), "--pairs", "4"])
    manifest = json.loads((out / "piano" / "pairs.json").read_text(encoding="utf-8"))
    assert manifest["pairs"]
    for record in manifest["pairs"]:
        assert sorted(path.name for path in (out / "piano" / record["directory"]).glob("*.wav")) == [
            "a.wav",
            "b.wav",
            "reference.wav",
        ]

    assert (out / "piano" / "labels.csv").is_file()
    assert "questions chosen from" in capsys.readouterr().out


def test_listen_scales_the_configured_quota_to_the_listening_asked_for(tmp_path: Path, held_notes: Path) -> None:
    asked = []
    for pairs in (4, 12):
        out = tmp_path / str(pairs)
        run(["listen", str(held_notes), "--budget-kb", "48", "--out", str(out), "--pairs", str(pairs)])
        asked.append(len(json.loads((out / "piano" / "pairs.json").read_text(encoding="utf-8"))["pairs"]))

    assert asked[0] < asked[1]


def test_listen_leaves_the_notes_too_short_to_judge_unasked(tmp_path: Path, tiny_notes: Path) -> None:
    out = tmp_path / "listening"

    run(["listen", str(tiny_notes), "--budget-kb", "48", "--out", str(out), "--pairs", "4"])

    assert json.loads((out / "piano" / "pairs.json").read_text(encoding="utf-8"))["pairs"] == []


def test_rank_command_reads_a_filled_in_sheet_and_writes_what_it_makes_of_the_metric(
    tmp_path: Path, held_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "listening"
    run(["listen", str(held_notes), "--budget-kb", "48", "--out", str(out), "--pairs", "4"])
    written = out / "piano"
    labels = written / "labels.csv"
    labels.write_text(
        labels.read_text(encoding="utf-8").replace(",,,", f",{Verdict.A_CLEARLY},{Fault.HISS},"),
        encoding="utf-8",
    )
    capsys.readouterr()

    run(["rank", str(written)])

    report = json.loads((written / "report.json").read_text(encoding="utf-8"))
    assert [metric["name"] for metric in report["metrics"]][:2] == ["objective", "composite"]
    assert report["outstanding"] == 0
    assert "confirmed" in capsys.readouterr().out


def test_rank_reports_on_what_a_part_filled_sheet_holds(tmp_path: Path, held_notes: Path) -> None:
    out = tmp_path / "listening"
    run(["listen", str(held_notes), "--budget-kb", "48", "--out", str(out), "--pairs", "4"])
    written = out / "piano"

    run(["rank", str(written)])

    report = json.loads((written / "report.json").read_text(encoding="utf-8"))
    assert (report["answered"], report["metrics"]) == (0, [])
    assert report["outstanding"] > 0


def test_reduce_reports_and_leaves_out_the_recordings_that_never_sound(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed render keeps no slot in the dataset, and the run says so rather than quietly shrinking."""
    silent = tmp_path / "piano" / f"0000_p{PITCHES[0]}_v100.wav"
    write_wav(silent, np.full(round(0.6 * SR), 1.0e-6), SR)
    out = tmp_path / "reduced"
    run(["reduce", str(tiny_notes), "--budget-kb", "48", "--out", str(out), "--rate", "11025", "--depth", "8"])
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
    run(["reduce", str(tiny_notes), *flags, "--out", str(reduced)])
    run(["optimize", str(tiny_notes), *flags, "--out", str(tmp_path / "direct"), "--no-render"])
    run(["optimize", str(reduced / "piano.notes.json"), *flags, "--out", str(tmp_path / "again"), "--no-render"])
    direct = (tmp_path / "direct" / "piano" / "ungrouped" / "plan.json").read_text(encoding="utf-8")
    again = (tmp_path / "again" / "piano" / "ungrouped" / "plan.json").read_text(encoding="utf-8")
    assert direct == again


def test_subset_command_writes_a_dataset_the_other_commands_read(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "subset"
    run(["subset", str(tiny_notes), "--fraction", "0.67", "--out", str(out)])

    assert (out / "piano.notes.json").is_file()
    assert len(list((out / "piano").glob("*.wav"))) == 2
    printed = capsys.readouterr().out
    assert "2 of 3 notes" in printed
    run(["optimize", str(out / "piano.notes.json"), "--budget-kb", "48", "--out", str(tmp_path / "art"), "--no-render"])
    assert (tmp_path / "art" / "piano" / "ungrouped" / "plan.json").is_file()


def test_the_subset_command_states_what_sounded_too_briefly_to_slice(
    tmp_path: Path, ragged_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "subset"
    run(
        [
            "subset",
            str(ragged_notes),
            "--fraction",
            "1.0",
            "--min-duration-s",
            str(_ADMITS_BOTH),
            "--out",
            str(out),
        ]
    )
    printed = capsys.readouterr().out

    assert "2 of 3 notes" in printed
    assert "1 of 3 notes sounded too briefly to slice" in printed


def test_the_subset_command_says_nothing_of_a_source_it_admits_whole(
    tmp_path: Path, ragged_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(["subset", str(ragged_notes), "--fraction", "1.0", "--min-duration-s", "0.0", "--out", str(tmp_path / "s")])

    assert "too briefly" not in capsys.readouterr().out


def test_the_subset_command_names_its_output_after_the_instrument(tmp_path: Path, tiny_notes: Path) -> None:
    run(["subset", str(tiny_notes), "--fraction", "1.0", "--instrument-id", "Grand", "--out", str(tmp_path / "s")])

    assert (tmp_path / "s" / "Grand.notes.json").is_file()
    assert len(list((tmp_path / "s" / "Grand").glob("*.wav"))) == len(PITCHES)


def test_a_directory_of_recordings_optimizes_with_no_manifest_naming_them(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pointing a command at the recordings alone is how a set of samples with no song behind it runs."""
    run(
        ["optimize", str(tiny_notes.parent / "piano"), "--budget-kb", "48", "--out", str(tmp_path / "a"), "--no-render"]
    )

    assert (tmp_path / "a" / "piano" / "ungrouped" / "plan.json").is_file()
    assert "objective" in capsys.readouterr().out


def test_a_chained_run_takes_a_directory_of_recordings_the_whole_way(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The slice a directory takes is a directory, so every stage downstream reads the shape it was given."""
    out = tmp_path / "artifacts"
    run(
        [
            "pipeline",
            str(tiny_notes.parent / "piano"),
            "--fraction",
            "0.67",
            "--budget-kb",
            "48",
            "--strategy",
            "ungrouped",
            "--no-render",
            "--out",
            str(out),
        ]
    )

    assert len(list((out / "0_subset" / "piano").glob("*.wav"))) == 2
    assert not (out / "0_subset" / "piano.notes.json").exists()
    assert (out / "3_optimized" / "piano" / "ungrouped" / "plan.json").is_file()
    assert "2 of 3 notes" in capsys.readouterr().out


def test_the_subset_command_slices_a_directory_of_recordings_into_a_directory(tmp_path: Path, tiny_notes: Path) -> None:
    out = tmp_path / "subset"
    run(["subset", str(tiny_notes.parent / "piano"), "--fraction", "0.67", "--out", str(out)])

    assert len(list((out / "piano").glob("*.wav"))) == 2
    assert not (out / "piano.notes.json").exists()


def test_pipeline_command_writes_a_directory_per_stage_it_ran(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "artifacts"
    run(
        [
            "pipeline",
            str(tiny_notes),
            "--budget-kb",
            "48",
            "--fraction",
            "0.67",
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
    assert (out / "0_subset" / "piano.notes.json").is_file()
    assert (out / "1_looped" / "piano.notes.json").is_file()
    assert (out / "2_reduced" / "piano.notes.json").is_file()
    assert (out / "3_optimized" / "piano" / "ungrouped" / "plan.json").is_file()
    printed = capsys.readouterr().out
    assert "2 of 3 notes" in printed  # the slice it took
    assert "auditions" in printed  # what the reduction wrote
    assert "ungrouped: objective" in printed and "total:" in printed


@pytest.mark.parametrize("stage", ["0_subset/piano", "1_looped/piano", "2_reduced/piano"])
def test_every_stage_of_a_chained_run_carries_its_recordings_as_instruments(
    stage: str, tmp_path: Path, tiny_notes: Path
) -> None:
    """A stage's own audio is playable in a tracker, so what one stage did to it is audible against the next."""
    out = tmp_path / "artifacts"
    run(
        # fmt: off
        [
            "pipeline", str(tiny_notes), "--budget-kb", "48", "--fraction", "0.67", "--out", str(out),
            "--rate", "11025", "--depth", "8", "--no-render", "--strategy", "ungrouped",
        ]
        # fmt: on
    )
    samples_dir = out / stage

    for extension in (".iti", ".xi"):
        written = list(instrument_files_dir(samples_dir, extension).glob(f"*{extension}"))
        assert [path.stem for path in sorted(written)] == [path.stem for path in sorted(samples_dir.glob("*.wav"))]


def test_a_plans_own_samples_are_carried_as_instruments_beside_them(tmp_path: Path, tiny_notes: Path) -> None:
    """One instrument per stored sample is how a single voice out of a plan is auditioned on its own."""
    out = tmp_path / "artifacts"
    run(
        # fmt: off
        [
            "optimize", str(tiny_notes), "--budget-kb", "48", "--out", str(out),
            "--no-render", "--strategy", "ungrouped",
        ]
        # fmt: on
    )
    samples_dir = out / "piano" / "ungrouped" / "samples"

    for extension in (".iti", ".xi"):
        written = list(instrument_files_dir(samples_dir, extension).glob(f"*{extension}"))
        assert [path.stem for path in sorted(written)] == [path.stem for path in sorted(samples_dir.glob("*.wav"))]


def test_the_instruments_command_fills_in_a_dataset_that_was_already_written(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tree written before is filled in from what it holds, which asks for no rerun of the stage that wrote it."""
    out = tmp_path / "subset"
    run(["subset", str(tiny_notes), "--fraction", "1.0", "--out", str(out)])
    shutil.rmtree(instrument_files_dir(out / "piano", ".iti"))

    run(["instruments", str(out / "piano.notes.json")])

    assert len(list(instrument_files_dir(out / "piano", ".iti").glob("*.iti"))) == len(PITCHES)
    assert "instruments -> ITI, XI" in capsys.readouterr().out


def test_the_pipeline_command_reaches_what_running_the_stages_one_at_a_time_reaches(
    tmp_path: Path, tiny_notes: Path
) -> None:
    """The chain exists to spare the typing, so it allocates the very plan the three commands do."""
    flags = ["--budget-kb", "48", "--rate", "11025", "--depth", "8"]
    allocate = ["--no-render", "--strategy", "ungrouped"]
    chained, apart = tmp_path / "chained", tmp_path / "apart"
    run(["pipeline", str(tiny_notes), *flags, *allocate, "--fraction", "0.67", "--out", str(chained)])
    run(["subset", str(tiny_notes), "--fraction", "0.67", "--out", str(apart / "0_subset")])
    run(["loop", str(apart / "0_subset" / "piano.notes.json"), *flags, "--out", str(apart / "1_looped")])
    run(["reduce", str(apart / "1_looped" / "piano.notes.json"), *flags, "--out", str(apart / "2_reduced")])
    run(["optimize", str(apart / "2_reduced" / "piano.notes.json"), *flags, *allocate, "--out", str(apart / "3")])
    plan = Path("piano") / "ungrouped" / "plan.json"
    assert (chained / "3_optimized" / plan).read_text(encoding="utf-8") == (apart / "3" / plan).read_text(
        encoding="utf-8"
    )


def test_the_pipeline_command_reduces_its_source_when_no_fraction_names_a_slice(
    tmp_path: Path, tiny_notes: Path
) -> None:
    out = tmp_path / "artifacts"
    flags = ["--budget-kb", "48", "--rate", "11025", "--depth", "8", "--no-render", "--strategy", "ungrouped"]
    run(["pipeline", str(tiny_notes), *flags, "--out", str(out)])
    assert not (out / "0_subset").exists()
    assert (out / "3_optimized" / "piano" / "ungrouped" / "plan.json").is_file()


def test_the_pipeline_command_stopped_before_optimizing_writes_no_plan(
    tmp_path: Path, tiny_notes: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A chain ended at ``--skip optimize`` keeps the datasets the allocation reads and nothing past them."""
    out = tmp_path / "artifacts"
    run(
        [
            "pipeline",
            str(tiny_notes),
            "--budget-kb",
            "48",
            "--fraction",
            "0.67",
            "--out",
            str(out),
            "--rate",
            "11025",
            "--depth",
            "8",
            "--no-render",
            "--strategy",
            "ungrouped",
            "--skip",
            "optimize",
        ]
    )
    assert (out / "0_subset" / "piano.notes.json").is_file()
    assert (out / "1_looped" / "piano.notes.json").is_file()
    assert (out / "2_reduced" / "piano.notes.json").is_file()
    assert not (out / "3_optimized").exists()
    printed = capsys.readouterr().out
    assert "stopped before optimize" in printed
    assert "total:" in printed


def test_the_allocation_caps_a_chained_run_states_reach_the_stage_that_allocates(config: OptiConfig) -> None:
    """The reduction runs at the configured caps, so its dataset stays the one any allocation reads back."""
    args = build_parser().parse_args(
        ["pipeline", "m.notes.json", "--budget-kb", "48", "--max-samples", "4", "--max-layers", "1"]
    )
    settings = _pipeline_settings(config, args)
    assert settings.dump.optimize.max_samples == 4
    assert settings.dump.optimize.layers.max_layers == 1
    assert settings.reduce.max_samples == config.optimize.budget.max_samples
    assert settings.reduce.layers == config.optimize.layers


def test_a_chained_run_slices_nothing_when_no_fraction_is_named(config: OptiConfig) -> None:
    args = build_parser().parse_args(["pipeline", "m.notes.json", "--budget-kb", "48"])
    assert _pipeline_settings(config, args).fraction is None


def test_the_skip_flag_names_the_stage_a_chain_stops_at(config: OptiConfig) -> None:
    """``--skip`` travels to the settings as the stage the chain ends before, and nothing where it is absent."""
    assert (
        _pipeline_settings(config, build_parser().parse_args(["pipeline", "m.notes.json", "--budget-kb", "48"])).skip
        is None
    )
    args = build_parser().parse_args(["pipeline", "m.notes.json", "--budget-kb", "48", "--skip", "reduce"])
    assert _pipeline_settings(config, args).skip is PipelineStage.REDUCE


def test_the_skip_flag_accepts_only_the_stages_the_chain_always_reaches() -> None:
    """The slice is omitted by leaving ``--fraction`` out, so the flag names the three stages past it."""
    for stage in ("loop", "reduce", "optimize"):
        args = build_parser().parse_args(["pipeline", "m.notes.json", "--budget-kb", "48", "--skip", stage])
        assert args.skip == stage

    with pytest.raises(SystemExit):
        build_parser().parse_args(["pipeline", "m.notes.json", "--budget-kb", "48", "--skip", "subset"])


def test_the_pipeline_command_reads_the_same_ingest_flags_as_optimize(config: OptiConfig) -> None:
    """One ingest parser across the commands, so a reduction knob means the same to a chain as to a stage."""
    argv = ["m.notes.json", "--budget-kb", "48", "--dedupe-key", "pitch", "--content-floor-db", "45", "--seed", "3"]
    chained = _pipeline_settings(config, build_parser().parse_args(["pipeline", *argv]))
    optimized = _dump_settings(config, build_parser().parse_args(["optimize", *argv]))
    assert chained.dump.optimize.reduce == optimized.optimize.reduce
    assert chained.reduce.reduce == optimized.optimize.reduce  # both stages of a chain, on the same knobs
    assert chained.dump.optimize.seed == 3


def test_the_reduce_command_reads_the_same_ingest_flags_as_optimize(config: OptiConfig) -> None:
    """Both commands share one ingest parser, so a reduction knob means the same thing to either."""
    argv = ["m.notes.json", "--budget-kb", "48", "--dedupe-key", "pitch", "--content-floor-db", "45", "--seed", "3"]
    reduced = build_parser().parse_args(["reduce", *argv])
    optimized = build_parser().parse_args(["optimize", *argv])
    assert _optimize_settings(config, reduced, config.optimize.layers, config.optimize.budget).reduce == (
        _optimize_settings(config, optimized, config.optimize.layers, config.optimize.budget).reduce
    )
    assert _optimize_settings(config, reduced, config.optimize.layers, config.optimize.budget).seed == 3


def test_the_roll_flags_state_a_directory_of_recordings_padding_in_seconds() -> None:
    """A directory names no rolls of its own, so the flags are read as the seconds it holds at each end."""
    argv = ["optimize", "piano", "--budget-kb", "48", "--pre-roll-ms", "20", "--post-roll-ms", "250"]
    settings = _ingest_settings(build_parser().parse_args(argv))

    assert (settings.pre_roll_s, settings.post_roll_s) == pytest.approx((0.02, 0.25))


def test_the_keep_tail_flag_asks_for_the_padding_past_each_release() -> None:
    argv = ["optimize", "m.notes.json", "--budget-kb", "48", "--keep-tail"]

    assert _ingest_settings(build_parser().parse_args(argv)).keep_tail


def test_the_workers_flag_sets_how_far_a_run_fans_out(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48", "--workers", "3"])
    assert _optimize_settings(config, args, config.optimize.layers, config.optimize.budget).workers == 3


def test_a_run_fans_out_the_configured_way_when_no_flag_names_a_count(config: OptiConfig) -> None:
    args = build_parser().parse_args(["optimize", "m.notes.json", "--budget-kb", "48"])
    assert (
        _optimize_settings(config, args, config.optimize.layers, config.optimize.budget).workers
        == config.runtime.workers
    )


def test_the_demo_reads_the_same_fan_out_flag_as_a_run(config: OptiConfig) -> None:
    """One flag on the shared parser, so ``--workers 1`` means an in-process run to either command."""
    args = build_parser().parse_args(["synth", "demo", "--workers", "1"])
    assert _demo_settings(config, args).workers == 1


def test_synth_command_generates_notes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(["synth", str(tmp_path / "demo"), "--sample-rate", "22050"])
    assert sorted((tmp_path / "demo").glob("*.notes.json"))
    assert "notes.json" in capsys.readouterr().out.lower()


# --- writing a run's recordings as clustered carrier instruments ---------------------------------------------


@pytest.fixture
def clustered_run(tiny_notes: Path, tmp_path: Path) -> Path:
    """A run root whose first stage holds the tiny project, which is what a clustered run reads."""
    root = tmp_path / "run"
    run(["subset", str(tiny_notes), "--fraction", "1.0", "--min-duration-s", "0.0", "--out", str(root / "0_subset")])
    return root


def test_cluster_writes_one_instrument_per_velocity_band(clustered_run: Path, tmp_path: Path) -> None:
    """A keymap names no dynamic, so each band the corpus is cut into is written as its own file."""
    out_dir = tmp_path / "clustered"
    run(
        [
            "cluster",
            str(clustered_run),
            "--stage",
            "subset",
            "--groups",
            "2",
            "--layers",
            "1",
            "--depth",
            "8",
            "--workers",
            "1",
            "--no-progress",
            "--out",
            str(out_dir),
        ]
    )

    written = sorted(instrument_files_dir(out_dir / "piano", ".iti").glob("*.iti"))
    assert len(written) == 1
    assert (out_dir / "piano.clustered.json").is_file()
    assert list((out_dir / "auditions").rglob("*.wav"))


def test_cluster_states_the_depth_and_groups_it_was_asked_for(clustered_run: Path, tmp_path: Path) -> None:
    """The flags reach the nested config, so the manifest reports what the run was actually told to do."""
    out_dir = tmp_path / "clustered"
    run(
        [
            "cluster",
            str(clustered_run),
            "--stage",
            "subset",
            "--groups",
            "3",
            "--layers",
            "1",
            "--depth",
            "16",
            "--workers",
            "1",
            "--no-progress",
            "--out",
            str(out_dir),
        ]
    )

    document = json.loads((out_dir / "piano.clustered.json").read_text())
    assert document["groups"] == 3
    assert {sample["depth"] for band in document["bands"] for sample in band["samples"]} == {16}


def test_cluster_asks_which_instrument_to_read_when_a_run_holds_several(clustered_run: Path, tmp_path: Path) -> None:
    """A run carrying more than one instrument names none of them by default, so the flag is required."""
    shutil.copytree(clustered_run / "0_subset" / "piano", clustered_run / "0_subset" / "other")
    shutil.copy(clustered_run / "0_subset" / "piano.notes.json", clustered_run / "0_subset" / "other.notes.json")

    with pytest.raises(ValueError, match="--instrument-id"):
        run(["cluster", str(clustered_run), "--stage", "subset", "--no-progress", "--out", str(tmp_path / "out")])
