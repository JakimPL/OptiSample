from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.io.note_extractor import NOTES_SUFFIX

_AUDITIONS_DIR: Final = "auditions"
_REDUCTION_DIR: Final = "reduction"
_REDUCTION_JSON: Final = "reduction.json"
_RANKING_JSON: Final = "pairs.json"
_LABELS_CSV: Final = "labels.csv"
_LISTENING_README: Final = "README.md"
_RANKING_REPORT_JSON: Final = "report.json"
_LOOPS_DIR: Final = "loops"
_LOOPS_JSON: Final = "loops.json"
_MODULE_STEM: Final = "module"
_CONTAINER_EXTENSION: Final = ".bank"
SAMPLE_EXTENSION: Final = ".sample"
_SUBSET_STAGE: Final = "0_subset"
_LOOPED_STAGE: Final = "1_looped"
_REDUCED_STAGE: Final = "2_reduced"
_OPTIMIZED_STAGE: Final = "3_optimized"


@dataclass(frozen=True)
class PipelinePaths:
    """Where each stage of a chained run lands under one output root.

    The names are numbered in the order the stages run, so the tree reads as the route a dataset took:
    the slice taken of the source, the loops settled on its recordings, what the pre-optimization stage
    reduced that to, and the artifacts allocated from it. Each directory is the output root of the stage
    that writes it, so the same stage reached on its own through ``subset``, ``loop``, ``reduce`` or
    ``optimize`` fills it identically.
    """

    subset_dir: Path
    looped_dir: Path
    reduced_dir: Path
    optimized_dir: Path


def pipeline_paths(out_dir: Path) -> PipelinePaths:
    """The stage directories a chained run writes under ``out_dir``."""
    return PipelinePaths(
        subset_dir=out_dir / _SUBSET_STAGE,
        looped_dir=out_dir / _LOOPED_STAGE,
        reduced_dir=out_dir / _REDUCED_STAGE,
        optimized_dir=out_dir / _OPTIMIZED_STAGE,
    )


def calibrated_path(samples_dir: Path, stem: str) -> Path:
    """The calibrated container one recording is carried as, beside the WAV of the same stem.

    A ``.sample`` holds the recording split into the level it moves through and the carrier that level
    scales, together with the loops settled over it, so the pair sits with the audio it was measured from
    the way a bank's manifest sits with the waveforms it names. Every stage writing containers puts them
    here, so a dataset carried off from any of them reads the same way.
    """
    return samples_dir / f"{stem}{SAMPLE_EXTENSION}"


def instrument_files_dir(samples_dir: Path, extension: str) -> Path:
    """Where the instruments of one format land beside the recordings they were written from.

    A recording of a written dataset stands in that dataset's samples directory, and the instruments
    carrying it stand under a directory of their own named for the extension they are written with --
    ``ITI/`` and ``XI/`` -- so one recording's WAV and every instrument of it read as one set of files and
    a player looking for the format it loads finds all of them together.
    """
    return samples_dir / extension.removeprefix(".").upper()


@dataclass(frozen=True)
class LoopedPaths:
    """Where one instrument's looped dataset and its loop decisions land under the output root.

    ``notes_json`` and ``samples_dir`` are the sibling pair a later ingest resolves by default, so the
    output root is itself a NoteExtractor dataset -- the recordings as the stage analysed them, onset
    aligned and at one rate, which is what makes the frames a loop names index into them, each beside the
    container carrying it (:func:`calibrated_path`). What the stage decided sits apart under
    ``loops_json``, and ``auditions_dir`` holds each loop played out.
    """

    notes_json: Path
    samples_dir: Path
    loops_json: Path
    auditions_dir: Path


def looped_paths(out_dir: Path, instrument_id: str) -> LoopedPaths:
    """The tree one instrument's loop run writes under ``out_dir``."""
    loops_dir = out_dir / _LOOPS_DIR / instrument_id
    return LoopedPaths(
        notes_json=out_dir / f"{instrument_id}{NOTES_SUFFIX}",
        samples_dir=out_dir / instrument_id,
        loops_json=loops_dir / _LOOPS_JSON,
        auditions_dir=loops_dir / _AUDITIONS_DIR,
    )


@dataclass(frozen=True)
class ReducedPaths:
    """Where one instrument's reduced dataset and its inspection material land under the output root.

    ``notes_json`` and ``samples_dir`` are the sibling pair a later ingest resolves by default, so the
    output root is itself a NoteExtractor dataset. What the stage decided sits apart under
    ``reduction_json`` and ``auditions_dir``, leaving the dataset holding recordings alone.
    """

    notes_json: Path
    samples_dir: Path
    reduction_json: Path
    auditions_dir: Path


def reduced_paths(out_dir: Path, instrument_id: str) -> ReducedPaths:
    """The tree one instrument's reduce run writes under ``out_dir``."""
    reduction_dir = out_dir / _REDUCTION_DIR / instrument_id
    return ReducedPaths(
        notes_json=out_dir / f"{instrument_id}{NOTES_SUFFIX}",
        samples_dir=out_dir / instrument_id,
        reduction_json=reduction_dir / _REDUCTION_JSON,
        auditions_dir=reduction_dir / _AUDITIONS_DIR,
    )


@dataclass(frozen=True)
class RankingPaths:
    """Where one instrument's listening set lands: the audio a listener meets, and what decodes it.

    ``pairs_dir`` holds one directory per question and is the whole of what a listener browses.
    ``manifest_json`` states which encoding took which side, so the set is decoded where a listener is
    finished rather than while they are working, ``labels_csv`` is where their answers go, and
    ``report_json`` is what those answers rank the metric as.
    """

    pairs_dir: Path
    manifest_json: Path
    labels_csv: Path
    readme: Path
    report_json: Path


def listening_set_paths(instrument_dir: Path) -> RankingPaths:
    """The files one written listening set holds inside its own directory.

    Reading a set back needs the directory alone, since a listener is handed that and nothing above it,
    so the tree it was written into is settled here rather than asked for again.
    """
    return RankingPaths(
        pairs_dir=instrument_dir,
        manifest_json=instrument_dir / _RANKING_JSON,
        labels_csv=instrument_dir / _LABELS_CSV,
        readme=instrument_dir / _LISTENING_README,
        report_json=instrument_dir / _RANKING_REPORT_JSON,
    )


def ranking_paths(out_dir: Path, instrument_id: str) -> RankingPaths:
    """The tree one instrument's listening set is written as under ``out_dir``."""
    return listening_set_paths(out_dir / instrument_id)


@dataclass(frozen=True)
class PlanPaths:
    """Where one strategy's artifacts land under its own directory.

    The dumper writes through these and an inspector reads them back, so both sides describe the tree
    the same way and a reader stays right when the layout moves.
    """

    directory: Path

    @property
    def report(self) -> Path:
        """The human-readable report, opening with what the pre-optimization stage reduced."""
        return self.directory / "report.txt"

    @property
    def plan_json(self) -> Path:
        """The plan itself: budgets, the velocity map, and the encoding chosen for every kept item."""
        return self.directory / "plan.json"

    @property
    def reduction_json(self) -> Path:
        """What the pre-optimization stage left the allocation to work from."""
        return self.directory / _REDUCTION_JSON

    @property
    def metrics_json(self) -> Path:
        """Per-note surrogate fidelity, summing back to the plan's objective."""
        return self.directory / "metrics.json"

    @property
    def infeasible(self) -> Path:
        """The note a strategy leaves behind when the budget affords no allocation at all."""
        return self.directory / "INFEASIBLE.txt"

    @property
    def samples_dir(self) -> Path:
        """Every stored sample decoded back to a float WAV, bit-identical to what the module holds."""
        return self.directory / "samples"

    def container(self, name: str) -> Path:
        """The bank the whole plan plays through, holding its manifest and every instrument it names.

        A bank travels as one file, so it is named after the instrument it plays: a tree holding a bank
        per strategy states which plays what, and a bank carried off on its own keeps saying so.
        """
        return self.directory / f"{name}{_CONTAINER_EXTENSION}"

    def stored(self, entry: str) -> Path:
        """Where one entry of the bank lands when its instruments are spread over this directory as well.

        A bank names its entries against itself, and an entry spread here keeps that name, so the loose
        instruments read as the map the manifest states — which is what a tracker loading a single voice
        off the tree reaches.
        """
        return self.directory / entry

    @property
    def compare_dir(self) -> Path:
        """The per-pitch A/B pairs: the recording as scored, beside what the module produces for it."""
        return self.directory / "compare"

    def layer_dir(self, layer: str) -> Path:
        """One velocity layer's A/B pairs, filed under the band it answers for.

        A key played softly and loudly is reconstructed once per layer, so the band the pair belongs to
        is part of where it lands and each layer's notes are auditioned as a set.
        """
        return self.compare_dir / layer

    @property
    def render_dir(self) -> Path:
        """Where a ground-truth render of the whole module lands."""
        return self.directory / "render"

    @property
    def module_render(self) -> Path:
        """The whole module rendered through openmpt123."""
        return self.render_dir / f"{_MODULE_STEM}.wav"

    def sample_wav(self, label: str) -> Path:
        """One stored sample's decoded WAV, filed under the label its plan unit carries."""
        return self.samples_dir / f"{label}.wav"

    def reference_wav(self, layer: str, pitch: str) -> Path:
        """The stretch of the recording one pitch's reconstruction in ``layer`` was scored against."""
        return self.layer_dir(layer) / f"{pitch}_ref.wav"

    def rendered_wav(self, layer: str, pitch: str) -> Path:
        """What the module produces for one pitch's representative note in ``layer``."""
        return self.layer_dir(layer) / f"{pitch}_render.wav"


def plan_paths(out_dir: Path, strategy: str) -> PlanPaths:
    """The tree one strategy writes under an instrument's ``out_dir``."""
    return PlanPaths(directory=out_dir / strategy)
