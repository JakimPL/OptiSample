from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts.context import DumpSettings
from optisample.artifacts.dump import dump_project
from optisample.artifacts.paths import pipeline_paths
from optisample.artifacts.pipeline import PipelineSettings, PipelineStage, run_pipeline
from optisample.config import OptiConfig
from optisample.io.audio import write_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import (
    IngestSettings,
    NoteRecord,
    dump_notes,
    load_notes,
)
from optisample.optimize.orchestrate.settings import OptimizeSettings

SR = 44_100
PITCHES = (60, 62, 64)
_INSTRUMENT = "piano"
_STRATEGY = "ungrouped"
_BUDGET_KB = 48.0
_SLICE = 0.67  # two of the three notes, so the slice is visibly what everything downstream reads
_SLICED_NOTES = 2


def _plan(directory: Path) -> str:
    """One instrument's plan as it landed on disk, which is what two runs are compared through."""
    return (directory / _STRATEGY / "plan.json").read_text(encoding="utf-8")


@pytest.fixture
def source(tmp_path: Path, piano_note: Callable[..., NDArray[np.float64]]) -> SourceDataset:
    """A minimal NoteExtractor dataset on disk: one WAV per note beside the manifest naming them."""
    samples_dir = tmp_path / "source" / _INSTRUMENT
    samples_dir.mkdir(parents=True)
    records = []
    for index, pitch in enumerate(PITCHES):
        write_wav(samples_dir / f"{index:04d}_p{pitch}_v100.wav", piano_note(pitch, 100, 0.6, seed=pitch), SR)
        records.append(NoteRecord(index=index, pitch=pitch, velocity=100, duration_s=0.5))

    notes_json = tmp_path / "source" / f"{_INSTRUMENT}.notes.json"
    dump_notes(records, notes_json)
    return SourceDataset(path=notes_json, samples_dir=samples_dir)


@pytest.fixture
def ingest(ingest_settings: Callable[..., IngestSettings]) -> IngestSettings:
    return ingest_settings(_INSTRUMENT, budget_kb=_BUDGET_KB)


@pytest.fixture
def allocation(tiny_settings: OptimizeSettings, config: OptiConfig) -> DumpSettings:
    """One strategy and the surrogate alone: the cheapest allocation a chain can end in."""
    return DumpSettings(
        optimize=tiny_settings,
        render=config.export.render,
        playback=config.export.playback,
        envelope=config.export.envelope,
        post_loop=config.export.instruments.post_loop,
        render_ground_truth=False,
        grouped=False,
    )


@pytest.fixture
def whole(
    config: OptiConfig, ingest: IngestSettings, tiny_settings: OptimizeSettings, allocation: DumpSettings
) -> PipelineSettings:
    """A chain reducing the source it was handed, which is what an already-sliced dataset asks for."""
    return PipelineSettings(
        ingest=ingest,
        reduce=tiny_settings,
        dump=allocation,
        instruments=allocation.instruments,
        intake=config.intake,
        fraction=None,
        skip=None,
    )


@pytest.fixture
def sliced(whole: PipelineSettings) -> PipelineSettings:
    """A chain slicing its source first, which is what a dataset worth iterating on smaller asks for."""
    return replace(whole, fraction=_SLICE)


def test_the_stage_directories_sort_into_the_order_the_stages_run(tmp_path: Path) -> None:
    """A numbered name is what makes a listing of the output root read as the route a dataset took."""
    paths = pipeline_paths(tmp_path)
    stages = (paths.subset_dir, paths.reduced_dir, paths.optimized_dir)
    assert [stage.name for stage in stages] == sorted(stage.name for stage in stages)
    assert all(stage.parent == tmp_path for stage in stages)


def test_a_chained_run_writes_each_stage_under_the_directory_that_names_it(
    tmp_path: Path, source: SourceDataset, sliced: PipelineSettings
) -> None:
    out = tmp_path / "chained"
    run = run_pipeline(source, out, sliced)
    paths = pipeline_paths(out)
    assert run.subset is not None and run.subset.dataset.source.path.parent == paths.subset_dir
    assert run.reduced is not None and run.reduced.paths.notes_json.parent == paths.reduced_dir
    assert run.optimized is not None and run.optimized.directory.parent == paths.optimized_dir
    assert (run.optimized.directory / _STRATEGY / "plan.json").is_file()


def test_a_run_naming_no_fraction_reduces_the_source_it_was_handed(
    tmp_path: Path, source: SourceDataset, whole: PipelineSettings
) -> None:
    """Leaving the fraction out is how a dataset already small enough goes through the same command."""
    out = tmp_path / "chained"
    run = run_pipeline(source, out, whole)
    assert run.subset is None
    assert not pipeline_paths(out).subset_dir.exists()
    assert run.reduced is not None and run.reduced.notes == len(PITCHES)
    assert run.optimized is not None and (run.optimized.directory / _STRATEGY / "plan.json").is_file()


def test_the_slice_a_run_takes_is_what_it_goes_on_to_reduce(
    tmp_path: Path, source: SourceDataset, sliced: PipelineSettings
) -> None:
    """Each stage reads the one before it, so a fraction narrows every stage downstream of it."""
    run = run_pipeline(source, tmp_path / "chained", sliced)
    assert run.subset is not None
    assert run.subset.dataset.kept_notes == _SLICED_NOTES
    assert run.reduced is not None and run.reduced.notes == _SLICED_NOTES


def test_the_allocation_reaches_the_plan_the_reduced_dataset_allocates_to(
    tmp_path: Path, source: SourceDataset, whole: PipelineSettings
) -> None:
    """The reduction exists to be read back, so a chain allocates from its dataset and that alone."""
    run = run_pipeline(source, tmp_path / "chained", whole)
    assert run.reduced is not None
    assert run.optimized is not None
    manifest = load_notes(run.reduced.paths.notes_json, run.reduced.paths.samples_dir, whole.ingest)
    (direct,) = dump_project(manifest, tmp_path / "direct", whole.dump)
    assert _plan(run.optimized.directory) == _plan(direct.directory)


def test_a_chained_run_states_the_wall_clock_the_whole_chain_took(
    tmp_path: Path, source: SourceDataset, whole: PipelineSettings
) -> None:
    """One run of several stages reports the time they add up to, which each stage's own is part of."""
    run = run_pipeline(source, tmp_path / "chained", whole)
    assert run.reduced is not None
    assert run.elapsed_s >= run.reduced.elapsed_s


def test_a_run_stopped_before_optimizing_writes_every_stage_but_the_allocation(
    tmp_path: Path, source: SourceDataset, whole: PipelineSettings
) -> None:
    """A chain stopped before optimizing keeps the datasets the allocation reads, and nothing past them."""
    run = run_pipeline(source, tmp_path / "chained", replace(whole, skip=PipelineStage.OPTIMIZE))
    paths = pipeline_paths(tmp_path / "chained")
    assert run.looped is not None
    assert run.reduced is not None
    assert run.optimized is None
    assert paths.reduced_dir.exists() and paths.looped_dir.exists()
    assert not paths.optimized_dir.exists()


def test_a_run_stopped_before_reducing_writes_only_the_loop_stage(
    tmp_path: Path, source: SourceDataset, whole: PipelineSettings
) -> None:
    """A chain stopped before reducing keeps the loops it settled and leaves the reduction unrun."""
    run = run_pipeline(source, tmp_path / "chained", replace(whole, skip=PipelineStage.REDUCE))
    paths = pipeline_paths(tmp_path / "chained")
    assert run.looped is not None
    assert run.reduced is None and run.optimized is None
    assert paths.looped_dir.exists()
    assert not paths.reduced_dir.exists() and not paths.optimized_dir.exists()


def test_a_run_stopped_before_looping_writes_only_the_slice(
    tmp_path: Path, source: SourceDataset, sliced: PipelineSettings
) -> None:
    """A chain stopped before looping keeps the slice and drops every stage past it."""
    run = run_pipeline(source, tmp_path / "chained", replace(sliced, skip=PipelineStage.LOOP))
    paths = pipeline_paths(tmp_path / "chained")
    assert run.subset is not None
    assert run.looped is None and run.reduced is None and run.optimized is None
    assert paths.subset_dir.exists()
    assert not paths.looped_dir.exists()
    assert not paths.reduced_dir.exists() and not paths.optimized_dir.exists()
