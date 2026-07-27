"""Tests for the notebook's staged-command helpers."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

from notebooks.utils import runs

_INSTRUMENT = "Piano"


@pytest.fixture
def fields(tmp_path: Path) -> runs.Fields:
    """A control bundle pointed at a source dataset, with a run root nothing has written to yet."""
    source = runs.Dataset(tmp_path / "src" / "Piano.notes.json", tmp_path / "src" / "Piano")
    return runs.Fields(
        source=source,
        instrument_id=_INSTRUMENT,
        out_root=tmp_path / "run",
        budget_kb=96.0,
        fraction=0.1,
        tracker_format="it",
        strategy="ungrouped",
        interpolation="sinc",
        dedupe_key="pitch_velocity",
        candidates=3,
        rates=(22_050, 11_025),
        depths=(16, 8),
        loop=True,
        workers=2,
        seed=7,
        render=False,
    )


def _make(dataset: runs.Dataset) -> runs.Dataset:
    """Put both halves of ``dataset`` on disk, so a stage reading it finds them."""
    dataset.samples_dir.mkdir(parents=True, exist_ok=True)
    dataset.notes_json.parent.mkdir(parents=True, exist_ok=True)
    dataset.notes_json.write_text('{"notes": []}', encoding="utf-8")
    return dataset


def _flag(command: list[str], name: str) -> list[str]:
    """Every value ``name`` was given on the command line, in the order they appear."""
    return [command[index + 1] for index, token in enumerate(command) if token == name]


# --- the invocations the controls stand for ----------------------------------------------------------


def test_each_stage_invokes_its_own_subcommand(fields: runs.Fields) -> None:
    for command, subcommand in (
        (runs.subset_command(fields), "subset"),
        (runs.reduce_command(fields, fields.source), "reduce"),
        (runs.optimize_command(fields, fields.source), "optimize"),
    ):
        assert command[:4] == [sys.executable, "-m", "optisample", subcommand]


def test_the_ingest_flags_carry_every_control_the_stages_share(fields: runs.Fields) -> None:
    command = runs.reduce_command(fields, fields.source)

    assert command[4] == str(fields.source.notes_json)
    assert _flag(command, "--samples-dir") == [str(fields.source.samples_dir)]
    assert _flag(command, "--budget-kb") == ["96"]
    assert _flag(command, "--dedupe-key") == ["pitch_velocity"]
    assert _flag(command, "--candidates") == ["3"]
    assert _flag(command, "--seed") == ["7"]
    assert _flag(command, "--workers") == ["2"]


def test_a_repeatable_flag_is_given_once_per_value(fields: runs.Fields) -> None:
    command = runs.optimize_command(fields, fields.source)

    assert _flag(command, "--rate") == ["22050", "11025"]
    assert _flag(command, "--depth") == ["16", "8"]


def test_the_sweep_falls_back_to_the_config_when_no_value_is_chosen(fields: runs.Fields) -> None:
    """An empty control states no preference, so the flag stays off and the loaded config decides."""
    command = runs.reduce_command(replace(fields, rates=(), depths=()), fields.source)

    assert "--rate" not in command
    assert "--depth" not in command


def test_looping_and_rendering_are_asked_off_by_their_own_flags(fields: runs.Fields) -> None:
    assert "--no-loop" not in runs.optimize_command(fields, fields.source)
    assert "--no-loop" in runs.optimize_command(replace(fields, loop=False), fields.source)
    assert "--no-render" in runs.optimize_command(fields, fields.source)
    assert "--no-render" not in runs.optimize_command(replace(fields, render=True), fields.source)


def test_each_stage_writes_under_its_own_root(fields: runs.Fields) -> None:
    assert _flag(runs.subset_command(fields), "--out") == [str(fields.out_root / "subset")]
    assert _flag(runs.reduce_command(fields, fields.source), "--out") == [str(fields.out_root / "reduced")]
    assert _flag(runs.optimize_command(fields, fields.source), "--out") == [str(fields.out_root / "artifacts")]


# --- which dataset each stage reads --------------------------------------------------------------------


def test_reduction_reads_the_source_until_a_subset_exists(fields: runs.Fields) -> None:
    assert runs.reduction_source(fields) == fields.source
    _make(fields.subset)
    assert runs.reduction_source(fields) == fields.subset


def test_a_whole_fraction_leaves_the_source_worth_reading_directly(fields: runs.Fields) -> None:
    whole = replace(fields, fraction=1.0)
    _make(whole.subset)
    assert runs.reduction_source(whole) == whole.source


def test_allocation_prefers_the_reduced_dataset_once_it_is_there(fields: runs.Fields) -> None:
    """Allocating from the survivors reproduces the plan the source produces, having paid only the ingest."""
    assert runs.allocation_source(fields) == fields.source
    _make(fields.subset)
    assert runs.allocation_source(fields) == fields.subset
    _make(fields.reduced)
    assert runs.allocation_source(fields) == fields.reduced


def test_a_dataset_missing_half_of_itself_is_not_ready(fields: runs.Fields) -> None:
    fields.subset.notes_json.parent.mkdir(parents=True)
    fields.subset.notes_json.write_text("{}", encoding="utf-8")
    assert not fields.subset.ready  # its recordings directory is still absent


def test_discover_finds_the_dataset_a_stage_left_behind(fields: runs.Fields) -> None:
    assert runs.discover(fields.out_root / "subset") is None
    _make(fields.subset)
    assert runs.discover(fields.out_root / "subset") == fields.subset


# --- running one ------------------------------------------------------------------------------------


def test_a_stage_run_reports_what_it_printed_and_files_a_transcript(fields: runs.Fields) -> None:
    outcome = runs.run_stage("probe", [sys.executable, "-c", "print('written')"], fields)

    assert outcome.ok
    assert "written" in outcome.output
    assert outcome.log == fields.out_root / "logs" / "probe.log"
    assert outcome.log.read_text(encoding="utf-8") == outcome.output
    assert outcome.elapsed_s >= 0.0


def test_a_failing_stage_reports_its_exit_code_and_its_error_stream(fields: runs.Fields) -> None:
    outcome = runs.run_stage("probe", [sys.executable, "-c", "import sys; sys.exit(3)"], fields)

    assert (outcome.ok, outcome.returncode) == (False, 3)


def test_the_outcome_states_the_line_a_terminal_would_have_run(fields: runs.Fields) -> None:
    outcome = runs.execute([sys.executable, "-m", "optisample", "--help"], fields.out_root / "help.log")

    assert outcome.shell == "optisample --help"
