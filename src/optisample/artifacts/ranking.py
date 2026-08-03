from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.artifacts.documents.ranking import pair_directory, ranking_document
from optisample.artifacts.paths import RankingPaths, ranking_paths
from optisample.artifacts.serialize import write_json, write_text
from optisample.calibrate.ranking import (
    ListeningPair,
    RankingSet,
    RankingSettings,
    Side,
    assemble_ranking,
    reference,
    rendered,
)
from optisample.io.audio import write_wav
from optisample.model import Manifest
from optisample.optimize.orchestrate import prepare_run
from optisample.optimize.orchestrate.audio import load_run_audio
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.tasks import EvalContext
from optisample.progress import ProgressSink

_REFERENCE_STEM: Final = "reference"
_WRITE_LABEL: Final = "Writing listening pairs"
_LABEL_COLUMNS: Final = ("directory", "closer", "note")
_VERDICTS: Final = ("a", "b", "tie")
_UNANSWERED: Final = ""  # what a pair's verdict holds while it waits for the listener

_README: Final = f"""# Listening set

One directory per question. Each holds `{_REFERENCE_STEM}.wav` -- the recording as it was played -- and
`{Side.A}.wav` and `{Side.B}.wav`, two encodings of it.

For each directory, answer one question: **which of {Side.A} and {Side.B} sounds closer to
`{_REFERENCE_STEM}.wav`?** Write `{_VERDICTS[0]}`, `{_VERDICTS[1]}` or `{_VERDICTS[2]}` in the `closer`
column of `labels.csv`, beside that directory's name. Answer `{_VERDICTS[2]}` freely: a pair you cannot
separate is a real reading, and it is one the metric is measured against like any other.

Which encoding took which side is settled by a seeded draw and is written in `pairs.json`, along with the
ranking the composite gives every pair. Reading it before you have finished tells you what the metric
already claims, which is the thing the labels are collected to check.
"""


@dataclass(frozen=True)
class ListeningSet:
    """One instrument's listening set as written: where it landed, and how much of it there is."""

    instrument_id: str
    paths: RankingPaths
    pairs: int
    priced: int
    elapsed_s: float


def _write_pair(pair: ListeningPair, directory: Path, context: EvalContext) -> None:
    """Write one question's three files: the recording, and each side's encoding of it."""
    directory.mkdir(parents=True, exist_ok=True)
    sample_rate = context.sample_rate
    write_wav(directory / f"{_REFERENCE_STEM}.wav", reference(pair.clip, context), sample_rate)
    write_wav(directory / f"{Side.A}.wav", rendered(pair.clip, pair.first, context), sample_rate)
    write_wav(directory / f"{Side.B}.wav", rendered(pair.clip, pair.second, context), sample_rate)


def labels_text(pairs: Sequence[ListeningPair]) -> str:
    """The answer sheet a listener fills in: one row per question, its verdict left open.

    Rows run in the order the pairs are met, so working down the directory listing and working down the
    sheet reach the same question at the same time.
    """
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_LABEL_COLUMNS)
    for index, pair in enumerate(pairs):
        writer.writerow((pair_directory(pair, index), _UNANSWERED, _UNANSWERED))

    return buffer.getvalue()


def write_ranking_set(
    ranking: RankingSet,
    paths: RankingPaths,
    *,
    instrument_id: str,
    seed: int,
    progress: ProgressSink,
) -> None:
    """Put ``ranking`` on disk: the audio of every question, the manifest decoding it, and the answer sheet."""
    paths.pairs_dir.mkdir(parents=True, exist_ok=True)
    tracked = progress.track(
        list(enumerate(ranking.pairs)),
        label=_WRITE_LABEL,
        total=len(ranking.pairs),
    )
    for index, pair in tracked:
        _write_pair(pair, paths.pairs_dir / pair_directory(pair, index), ranking.context)

    write_json(
        paths.manifest_json,
        ranking_document(
            ranking.pairs,
            instrument_id=instrument_id,
            sample_rate=ranking.context.sample_rate,
            seed=seed,
            priced_encodings=ranking.priced,
        ),
    )
    write_text(paths.labels_csv, labels_text(ranking.pairs))
    write_text(paths.readme, _README)


def dump_ranking(
    looped: LoopedInstrument,
    out_dir: Path | str,
    settings: OptimizeSettings,
    ranking: RankingSettings,
) -> ListeningSet:
    """Build one instrument's listening set from a prepared run and write it under ``out_dir``.

    The run is the one the allocation itself prepares, so the encodings a listener is asked about are
    priced by the objective exactly as the sweep prices them and a label speaks to the plan as it stands.
    """
    started_at = perf_counter()
    instrument = looped.loaded.instrument
    paths = ranking_paths(Path(out_dir), instrument.id)
    inputs = prepare_run(instrument, looped.recordings, settings)
    built = assemble_ranking(inputs, ranking, settings.progress)
    write_ranking_set(
        built,
        paths,
        instrument_id=instrument.id,
        seed=ranking.seed,
        progress=settings.progress,
    )
    return ListeningSet(
        instrument_id=instrument.id,
        paths=paths,
        pairs=len(built.pairs),
        priced=built.priced,
        elapsed_s=perf_counter() - started_at,
    )


def ranking_project(
    manifest: Manifest,
    out_dir: Path | str,
    settings: OptimizeSettings,
    ranking: RankingSettings,
) -> list[ListeningSet]:
    """Build a listening set for every instrument of a loaded manifest, each under its own directory."""
    out_dir = Path(out_dir)
    sets: list[ListeningSet] = []
    for instrument in manifest.instruments:
        looped = run_loops(load_run_audio(instrument, settings), settings)
        sets.append(dump_ranking(looped, out_dir, settings, ranking))

    return sets
