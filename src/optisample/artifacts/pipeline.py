from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from time import perf_counter

from optisample.artifacts.context import DumpResult, DumpSettings
from optisample.artifacts.dataset import SlicedDataset, SliceSettings, write_slice
from optisample.artifacts.dump import dump_project
from optisample.artifacts.instruments.dump import InstrumentSettings
from optisample.artifacts.looped import LoopedInstrumentArtifacts, loop_project
from optisample.artifacts.paths import pipeline_paths
from optisample.artifacts.reduced import ReducedInstrument, reduce_project
from optisample.config.subset import IntakeConfig
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import IngestSettings
from optisample.io.source import load_source
from optisample.optimize.orchestrate.settings import OptimizeSettings


@unique
class PipelineStage(StrEnum):
    """The stage a chained run is stopped at, named by the command that runs that stage alone.

    Holds the stages the chain always reaches -- looping, reduction and allocation -- in the order they
    run. Naming one stops the chain before it, so that stage and every stage after it are dropped.
    """

    LOOP = "loop"
    REDUCE = "reduce"
    OPTIMIZE = "optimize"


@dataclass(frozen=True)
class PipelineSettings:
    """What each stage of a chained run is carried out with.

    ``reduce`` and ``dump`` are stated apart so each stage is settled on its own terms: the velocity
    split and the sample cap reach the stage that allocates, while the reduction runs at the values that
    keep its dataset the one any allocation off it reads back.

    ``fraction`` is the share of the source a slice keeps. Naming none reduces the source as it stands,
    which is what a dataset small enough to run whole -- or already sliced -- asks for. ``intake`` is what
    that slice does to the material it draws on: the length a note sounds for to be kept, and the band it
    is written past, which together are where a chain takes its material in.

    ``instruments`` is what every stage carries its own recordings as, so each dataset the chain writes is
    playable in a tracker on the terms the run's own export states.

    ``skip`` is the stage a run ends at, dropping it and every stage after it. Naming none runs the whole
    chain.
    """

    ingest: IngestSettings
    reduce: OptimizeSettings
    dump: DumpSettings
    instruments: InstrumentSettings
    intake: IntakeConfig
    fraction: float | None
    skip: PipelineStage | None


@dataclass(frozen=True)
class PipelineRun:
    """What one chained run produced, stage by stage, and the wall-clock the chain took end to end.

    ``subset`` is the slice the run took of its source, carried by the runs that asked for one. Every
    stage past the point a run was stopped at is ``None``: a run stopped before looping leaves
    ``looped``, ``reduced`` and ``optimized`` empty, and the tree holds only the directories of the
    stages that ran.
    """

    subset: SlicedDataset | None
    looped: LoopedInstrumentArtifacts | None
    reduced: ReducedInstrument | None
    optimized: DumpResult | None
    elapsed_s: float


def _sliced(source: SourceDataset, out_dir: Path, settings: PipelineSettings) -> SlicedDataset | None:
    """The slice a run takes of its source ahead of everything else, on the runs that asked for one."""
    if settings.fraction is None:
        return None

    return write_slice(
        source,
        out_dir,
        SliceSettings(
            instrument_id=settings.ingest.instrument_id,
            fraction=settings.fraction,
            intake=settings.intake,
            instruments=settings.instruments,
        ),
    )


def _next_source(source: SourceDataset, subset: SlicedDataset | None) -> SourceDataset:
    """What the reduction reads: the slice a run took, or the source itself where it took none.

    A slice is written as a dataset of the same shape, so either one is read the very same way.
    """
    if subset is None:
        return source

    return subset.dataset.source


def _one_instrument[ResultT](results: Sequence[ResultT]) -> ResultT:
    """The single instrument a project-level stage answered for.

    An ingest joins one manifest to one samples directory and names the instrument that pair is
    (:func:`~optisample.io.note_extractor.load_notes`), so a chain carries exactly that one instrument
    from stage to stage.

    Raises:
        ValueError: if a stage answered for any other number of instruments.
    """
    (result,) = results
    return result


def _looped(source: SourceDataset, out_dir: Path, settings: PipelineSettings) -> LoopedInstrumentArtifacts:
    """The loops ``source``'s recordings are stored around, as the dataset the reduction reads back.

    The dataset written here holds the recordings as the ingest produced them -- onset aligned and at one
    rate -- which is what makes the frames each settled loop names index into the audio every later stage
    encodes from.
    """
    manifest = load_source(source, settings.ingest)
    return _one_instrument(loop_project(manifest, out_dir, settings.reduce, settings.instruments))


def _reduced(looped: LoopedInstrumentArtifacts, out_dir: Path, settings: PipelineSettings) -> ReducedInstrument:
    """What the pre-optimization stage reduces a looped dataset to, as the dataset the allocation reads back."""
    source = SourceDataset(path=looped.paths.notes_json, samples_dir=looped.paths.samples_dir)
    manifest = load_source(source, settings.ingest)
    return _one_instrument(reduce_project(manifest, out_dir, settings.reduce, settings.instruments))


def _optimized(reduced: ReducedInstrument, out_dir: Path, settings: PipelineSettings) -> DumpResult:
    """The artifacts allocated from a reduced dataset, one directory per strategy the run asked for."""
    source = SourceDataset(path=reduced.paths.notes_json, samples_dir=reduced.paths.samples_dir)
    return _one_instrument(dump_project(load_source(source, settings.ingest), out_dir, settings.dump))


def run_pipeline(source: SourceDataset, out_dir: Path, settings: PipelineSettings) -> PipelineRun:
    """Slice, reduce and allocate one dataset in a row, each stage writing its own directory under ``out_dir``.

    Every stage reads back the dataset the stage before it wrote, so the allocation reaches the sweep
    having paid the ingest over the surviving recordings alone, and the tree records the route it took:
    the slice under ``0_subset``, the loops settled on it under ``1_looped``, what that reduced to under
    ``2_reduced``, and the artifacts allocated from it under ``3_optimized``. A run naming no fraction
    settles loops on its source directly and begins at ``1_looped``, which is how an already-sliced dataset
    is carried through the same command.

    A run stopped at ``settings.skip`` ends before that stage and writes the stages before it alone, so
    the tree holds only the directories of the stages that ran and the result leaves every dropped stage
    ``None``.

    Every stage carries its own recordings as standalone instruments beside them, so each step of the
    route is playable in a tracker and what one stage did to the audio is audible against the stage before
    it.
    """
    started_at = perf_counter()
    paths = pipeline_paths(out_dir)
    subset = _sliced(source, paths.subset_dir, settings)
    if settings.skip is PipelineStage.LOOP:
        return PipelineRun(
            subset=subset,
            looped=None,
            reduced=None,
            optimized=None,
            elapsed_s=perf_counter() - started_at,
        )

    looped = _looped(_next_source(source, subset), paths.looped_dir, settings)
    if settings.skip is PipelineStage.REDUCE:
        return PipelineRun(
            subset=subset,
            looped=looped,
            reduced=None,
            optimized=None,
            elapsed_s=perf_counter() - started_at,
        )

    reduced = _reduced(looped, paths.reduced_dir, settings)
    if settings.skip is PipelineStage.OPTIMIZE:
        return PipelineRun(
            subset=subset,
            looped=looped,
            reduced=reduced,
            optimized=None,
            elapsed_s=perf_counter() - started_at,
        )

    optimized = _optimized(reduced, paths.optimized_dir, settings)
    return PipelineRun(
        subset=subset,
        looped=looped,
        reduced=reduced,
        optimized=optimized,
        elapsed_s=perf_counter() - started_at,
    )
