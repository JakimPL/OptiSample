from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.io.note_extractor import NOTES_SUFFIX

_AUDITIONS_DIR: Final = "auditions"
_REDUCTION_DIR: Final = "reduction"
_REDUCTION_JSON: Final = "reduction.json"
_MODULE_STEM: Final = "module"
_CONTAINER_EXTENSION: Final = ".bank"
_SUBSET_STAGE: Final = "0_subset"
_REDUCED_STAGE: Final = "1_reduced"
_OPTIMIZED_STAGE: Final = "2_optimized"


@dataclass(frozen=True)
class PipelinePaths:
    """Where each stage of a chained run lands under one output root.

    The names are numbered in the order the stages run, so the tree reads as the route a dataset took:
    the slice taken of the source, what the pre-optimization stage reduced it to, and the artifacts
    allocated from that. Each directory is the output root of the stage that writes it, so the same
    stage reached on its own through ``subset``, ``reduce`` or ``optimize`` fills it identically.
    """

    subset_dir: Path
    reduced_dir: Path
    optimized_dir: Path


def pipeline_paths(out_dir: Path) -> PipelinePaths:
    """The stage directories a chained run writes under ``out_dir``."""
    return PipelinePaths(
        subset_dir=out_dir / _SUBSET_STAGE,
        reduced_dir=out_dir / _REDUCED_STAGE,
        optimized_dir=out_dir / _OPTIMIZED_STAGE,
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
