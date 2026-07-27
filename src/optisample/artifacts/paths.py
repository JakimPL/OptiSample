from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.io.note_extractor import NOTES_SUFFIX

_AUDITIONS_DIR: Final = "auditions"
_REDUCTION_DIR: Final = "reduction"
_REDUCTION_JSON: Final = "reduction.json"
_MODULE_STEM: Final = "module"


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
    def velocity_map_json(self) -> Path:
        """The velocity->volume map alone, as the pattern writes each note's dynamic through it."""
        return self.directory / "velocity_map.json"

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

    @property
    def compare_dir(self) -> Path:
        """The per-pitch A/B pairs: the recording as scored, beside what the module produces for it."""
        return self.directory / "compare"

    @property
    def render_dir(self) -> Path:
        """Where a ground-truth render of the whole module lands."""
        return self.directory / "render"

    @property
    def module_render(self) -> Path:
        """The whole module rendered through openmpt123."""
        return self.render_dir / f"{_MODULE_STEM}.wav"

    def module(self, extension: str) -> Path:
        """The written module, named by the format's own ``extension`` (``.it`` / ``.xm``)."""
        return self.directory / f"{_MODULE_STEM}{extension}"

    def sample_wav(self, label: str) -> Path:
        """One stored sample's decoded WAV, filed under the label its plan unit carries."""
        return self.samples_dir / f"{label}.wav"

    def reference_wav(self, pitch: str) -> Path:
        """The stretch of the recording one pitch's reconstruction was scored against."""
        return self.compare_dir / f"{pitch}_ref.wav"

    def rendered_wav(self, pitch: str) -> Path:
        """What the module produces for one pitch's representative note."""
        return self.compare_dir / f"{pitch}_render.wav"


def plan_paths(out_dir: Path, strategy: str) -> PlanPaths:
    """The tree one strategy writes under an instrument's ``out_dir``."""
    return PlanPaths(directory=out_dir / strategy)
