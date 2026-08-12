from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.artifacts.documents.ranking import (
    RankingSetDocument,
    pair_directory,
    ranking_document,
)
from optisample.artifacts.paths import RankingPaths, ranking_paths
from optisample.artifacts.serialize import read_json, write_json, write_text
from optisample.calibrate.ranking import (
    Fault,
    LabelSheet,
    ListeningPair,
    RankingSet,
    RankingSettings,
    Side,
    Verdict,
    assemble_ranking,
    blank_sheet,
    labels_text,
    read_labels,
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

_README: Final = f"""# Listening set

One directory per question. Each holds `{_REFERENCE_STEM}.wav` -- the recording as it was played -- and
`{Side.A}.wav` and `{Side.B}.wav`, two encodings of it.

For each directory, answer one question: **which of {Side.A} and {Side.B} sounds closer to
`{_REFERENCE_STEM}.wav`?** Write the verdict in `labels.csv`, beside that directory's name.

| verdict | means |
|---|---|
| `{Verdict.A_CLEARLY}` | {Side.A} is clearly closer |
| `{Verdict.A_SLIGHTLY}` | {Side.A} is slightly closer |
| `{Verdict.TIE}` | both stand the same distance from the recording |
| `{Verdict.IDENTICAL}` | the two sound alike, with nothing between them to hear |
| `{Verdict.B_SLIGHTLY}` | {Side.B} is slightly closer |
| `{Verdict.B_CLEARLY}` | {Side.B} is clearly closer |

Answer `{Verdict.TIE}` and `{Verdict.IDENTICAL}` freely -- a pair you place level is a real reading, and
the metric is measured against it like any other. The two are read apart: `{Verdict.TIE}` says each side
has its own fault and they cost the same, while `{Verdict.IDENTICAL}` says there was nothing to tell them
by. The second holds a metric to a floor, since a pair you met as one recording is one it is due to read
as one -- a different check from getting the order right, and the report states both.

The `fault` column is open on every row and names what the side you *rejected* does wrong:
`{Fault.HISS}` steady noise or grain, `{Fault.DULL}` lost top, `{Fault.FLUTTER}` a fast periodic wobble,
`{Fault.BEATING}` a slow interference, `{Fault.CLICK}` a discontinuity, `{Fault.STOP}` the note ending or
decaying wrongly, `{Fault.OTHER}` anything else. A verdict alone ranks the metric; the fault is what says
which change the ranking calls for, so it is worth a word where one comes to mind.

## Working through it

Fixed volume and one pair of headphones throughout. Level is deliberately left as each encoding produces
it, so a side that plays louder is telling you something real -- the gap is measured into `pairs.json` and
the reading is checked against it afterwards.

Take the set in blocks with breaks between them. A few questions are put more than once, blinded afresh
and far apart: answer each one as you hear it rather than reaching for what you said before, since how far
those agree is what says whether the whole sheet can be trusted.

Which encoding took which side is settled by a seeded draw and is written in `pairs.json`, along with the
ranking the composite gives every pair. Reading it before you have finished tells you what the metric
already claims, which is the thing the labels are collected to check.

## Reading it back

`optisample rank <this directory>` ranks the composite, each term inside it and the level diagnostics
against whatever the sheet holds, and writes the table to `report.json`. Run it part way through and it
reports on the questions answered so far, so the set is worth something from the first block onwards.
"""


@dataclass(frozen=True)
class PairClips:
    """The three recordings one question puts in front of a listener."""

    reference: Path
    first: Path
    second: Path


@dataclass(frozen=True)
class ListeningSet:
    """One instrument's listening set as written: where it landed, and how much of it there is."""

    instrument_id: str
    paths: RankingPaths
    questions: int
    repeats: int
    priced: int
    elapsed_s: float

    @property
    def pairs(self) -> int:
        """Every pair written, which is the listening the set costs."""
        return self.questions + self.repeats


def _write_pair(pair: ListeningPair, directory: Path, context: EvalContext) -> None:
    """Write one question's three files: the recording, and each side's encoding of it."""
    directory.mkdir(parents=True, exist_ok=True)
    sample_rate = context.sample_rate
    write_wav(directory / f"{_REFERENCE_STEM}.wav", reference(pair.clip, context), sample_rate)
    write_wav(directory / f"{Side.A}.wav", rendered(pair.clip, pair.first, context), sample_rate)
    write_wav(directory / f"{Side.B}.wav", rendered(pair.clip, pair.second, context), sample_rate)


def pair_clips(paths: RankingPaths, directory: str) -> PairClips:
    """Where one question's three recordings sit, so a listener is handed them without naming the files."""
    held = paths.pairs_dir / directory
    return PairClips(
        reference=held / f"{_REFERENCE_STEM}.wav",
        first=held / f"{Side.A}.wav",
        second=held / f"{Side.B}.wav",
    )


def read_ranking_set(paths: RankingPaths) -> RankingSetDocument:
    """The manifest decoding one written listening set: every pair, both sides, and the composite's call."""
    return read_json(paths.manifest_json, RankingSetDocument)


def read_label_sheet(paths: RankingPaths) -> LabelSheet:
    """The answer sheet as it stands beside one written set, whether part-filled or finished."""
    return read_labels(paths.labels_csv.read_text(encoding="utf-8"))


def write_label_sheet(sheet: LabelSheet, paths: RankingPaths) -> None:
    """Put ``sheet`` back beside the set it answers, which is what lets a session be picked up again."""
    write_text(paths.labels_csv, labels_text(sheet))


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
    write_text(
        paths.labels_csv,
        labels_text(blank_sheet(pair_directory(pair, index) for index, pair in enumerate(ranking.pairs))),
    )
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
        questions=built.questions,
        repeats=built.repeats,
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
