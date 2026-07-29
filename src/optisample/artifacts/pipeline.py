from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from optisample.artifacts.context import DumpResult, DumpSettings
from optisample.artifacts.dump import dump_project
from optisample.artifacts.paths import pipeline_paths
from optisample.artifacts.reduced import ReducedInstrument, reduce_project
from optisample.io.dataset import SourceDataset, SubsetDataset
from optisample.io.note_extractor import IngestSettings
from optisample.io.source import load_source, write_source_subset
from optisample.optimize.orchestrate.settings import OptimizeSettings


@dataclass(frozen=True)
class PipelineSettings:
    """What each stage of a chained run is carried out with.

    ``reduce`` and ``dump`` are stated apart so each stage is settled on its own terms: the velocity
    split and the sample cap reach the stage that allocates, while the reduction runs at the values that
    keep its dataset the one any allocation off it reads back.

    ``fraction`` is the share of the source a slice keeps. Naming none reduces the source as it stands,
    which is what a dataset small enough to run whole -- or already sliced -- asks for.
    """

    ingest: IngestSettings
    reduce: OptimizeSettings
    dump: DumpSettings
    fraction: float | None


@dataclass(frozen=True)
class PipelineRun:
    """What one chained run produced, stage by stage, and the wall-clock the chain took end to end.

    ``subset`` is the slice the run took of its source, carried by the runs that asked for one.
    """

    subset: SubsetDataset | None
    reduced: ReducedInstrument
    optimized: DumpResult
    elapsed_s: float


def _sliced(source: SourceDataset, out_dir: Path, settings: PipelineSettings) -> SubsetDataset | None:
    """The slice a run takes of its source ahead of everything else, on the runs that asked for one."""
    if settings.fraction is None:
        return None

    return write_source_subset(
        source,
        out_dir,
        instrument_id=settings.ingest.instrument_id,
        fraction=settings.fraction,
    )


def _next_source(source: SourceDataset, subset: SubsetDataset | None) -> SourceDataset:
    """What the reduction reads: the slice a run took, or the source itself where it took none.

    A slice is written as a dataset of the same shape, so either one is read the very same way.
    """
    if subset is None:
        return source

    return subset.source


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


def _reduced(source: SourceDataset, out_dir: Path, settings: PipelineSettings) -> ReducedInstrument:
    """What the pre-optimization stage reduces ``source`` to, as the dataset the allocation reads back."""
    manifest = load_source(source, settings.ingest)
    return _one_instrument(reduce_project(manifest, out_dir, settings.reduce))


def _optimized(reduced: ReducedInstrument, out_dir: Path, settings: PipelineSettings) -> DumpResult:
    """The artifacts allocated from a reduced dataset, one directory per strategy the run asked for."""
    source = SourceDataset(path=reduced.paths.notes_json, samples_dir=reduced.paths.samples_dir)
    return _one_instrument(dump_project(load_source(source, settings.ingest), out_dir, settings.dump))


def run_pipeline(source: SourceDataset, out_dir: Path, settings: PipelineSettings) -> PipelineRun:
    """Slice, reduce and allocate one dataset in a row, each stage writing its own directory under ``out_dir``.

    Every stage reads back the dataset the stage before it wrote, so the allocation reaches the sweep
    having paid the ingest over the surviving recordings alone, and the tree records the route it took:
    the slice under ``0_subset``, what that reduced to under ``1_reduced``, and the artifacts allocated
    from it under ``2_optimized``. A run naming no fraction reduces its source directly and begins at
    ``1_reduced``, which is how an already-sliced dataset is carried through the same command.
    """
    started_at = perf_counter()
    paths = pipeline_paths(out_dir)
    subset = _sliced(source, paths.subset_dir, settings)
    reduced = _reduced(_next_source(source, subset), paths.reduced_dir, settings)
    optimized = _optimized(reduced, paths.optimized_dir, settings)
    return PipelineRun(
        subset=subset,
        reduced=reduced,
        optimized=optimized,
        elapsed_s=perf_counter() - started_at,
    )
