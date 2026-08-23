from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum, unique
from pathlib import Path
from shutil import rmtree
from typing import Any, Final

from notebooks.utils.runs import Outcome, execute
from optisample.io.note_extractor import NOTES_SUFFIX
from research.config_tree import variant
from research.materials import Material
from research.records import LoopReadings, MeasuredReadings, PlanReadings, ReductionReadings

_PACKAGE: Final = "optisample"
_LOG_SUFFIX: Final = ".log"
_FAILURE_LINES: Final = 12  # lines of a failed transcript kept, which is where the reason states itself
_PREPARE_BUDGET_KB: Final = 256.0  # the budget a preparing run is given, which its reduction never reads
_SHIPPED: Final = "shipped"  # the config copy a preparing run reads, which carries no override at all


@unique
class Source(StrEnum):
    """Which dataset a cell reads: the recordings as they were captured, or the survivors already reduced.

    Reading the reduced dataset reproduces the plan its source produces while reaching the sweep having
    paid only the ingest, so an allocation sweep prepares one reduction per material and spends its cells
    on the allocation alone.
    """

    SOURCE = "source"
    REDUCED = "reduced"


@unique
class Stage(StrEnum):
    """The stage one cell runs, which settles both the command and the artifact its readings come off."""

    LOOP = "loop"
    REDUCE = "reduce"
    OPTIMIZE = "optimize"


@dataclass(frozen=True)
class Cell:
    """One experiment: a stage run over one material under one setting, and what names it in the store.

    ``overrides`` are dotted config keys written into a copy of the shipped tree, and ``flags`` the CLI
    arguments a knob is reachable through without one. ``family`` groups the cells a question is answered
    from and ``label`` names this cell within it, so a record is readable without re-deriving the setting.
    """

    family: str
    label: str
    material: Material
    stage: Stage
    budget_kb: float
    overrides: Mapping[str, Any] = field(default_factory=dict)
    flags: Sequence[str] = field(default_factory=tuple)
    strategy: str = "grouped"
    source: Source = Source.SOURCE

    @property
    def slug(self) -> str:
        """The directory name this cell's run and config are filed under."""
        return f"{self.family}~{self.label}~{self.material.instrument_id}~{self.stage.value}".replace(" ", "_")


def failure(outcome: Outcome) -> str | None:
    """The tail of a failed run's transcript, which is where it says why it failed."""
    if outcome.ok:
        return None

    return "\n".join(outcome.output.splitlines()[-_FAILURE_LINES:])


@dataclass(frozen=True)
class Invocation:
    """What one staged command is run with beyond the dataset and the config it reads."""

    budget_kb: float
    workers: int
    seed: int


def _command(stage: Stage, material: Material, config: Path, out: Path, run: Invocation) -> list[str]:
    """The invocation one stage runs over ``material``, under the config copy the cell is measured at."""
    return [
        sys.executable,
        "-m",
        _PACKAGE,
        stage.value,
        str(material.notes_json),
        "--samples-dir",
        str(material.samples_dir),
        "--instrument-id",
        material.instrument_id,
        "--budget-kb",
        f"{run.budget_kb:g}",
        "--config",
        str(config),
        "--workers",
        str(run.workers),
        "--seed",
        str(run.seed),
        "--no-progress",
        "--out",
        str(out),
    ]


def _allocation_flags(cell: Cell) -> list[str]:
    """What an allocation asks for beyond the shared ingest: the strategy, and the ground truth left off."""
    return ["--strategy", cell.strategy, "--no-render"] if cell.stage is Stage.OPTIMIZE else []


def loops_json(out: Path, instrument_id: str) -> Path:
    """Where the loop stage files what every recording offered."""
    return out / "loops" / instrument_id / "loops.json"


def reduction_json(out: Path, instrument_id: str) -> Path:
    """Where the reduction files what it kept."""
    return out / "reduction" / instrument_id / "reduction.json"


def plan_json(out: Path, instrument_id: str, strategy: str) -> Path:
    """Where an allocation files the plan it settled on."""
    return out / instrument_id / strategy / "plan.json"


def metrics_json(out: Path, instrument_id: str, strategy: str) -> Path:
    """Where an allocation files the per-note metrics its written samples measure."""
    return out / instrument_id / strategy / "metrics.json"


def infeasible_txt(out: Path, instrument_id: str, strategy: str) -> Path:
    """Where an allocation states the reason it wrote no plan, which is an outcome rather than a failure."""
    return out / instrument_id / strategy / "INFEASIBLE.txt"


@dataclass(frozen=True)
class Readings:
    """Whichever readings the stage a cell ran leaves behind, the rest left unread."""

    loop: LoopReadings | None = None
    reduction: ReductionReadings | None = None
    plan: PlanReadings | None = None
    measured: MeasuredReadings | None = None
    infeasible: str | None = None


def _allocated(out: Path, instrument_id: str, strategy: str) -> Readings:
    """What an allocation left: the plan it wrote, or the reason the budget put every plan out of reach.

    A strategy with no allocation inside its budget states that and writes no plan, which is an answer the
    sweep records rather than an error, so the setting that caused it reads beside the ones that fitted.
    """
    plan = plan_json(out, instrument_id, strategy)
    if not plan.is_file():
        return Readings(infeasible=infeasible_txt(out, instrument_id, strategy).read_text(encoding="utf-8").strip())

    return Readings(
        plan=PlanReadings.read(plan),
        measured=MeasuredReadings.read(metrics_json(out, instrument_id, strategy)),
    )


def _readings(cell: Cell, out: Path) -> Readings:
    """The readings ``cell`` produced, taken off the artifacts its stage wrote."""
    instrument_id = cell.material.instrument_id
    match cell.stage:
        case Stage.LOOP:
            return Readings(loop=LoopReadings.read(loops_json(out, instrument_id)))
        case Stage.REDUCE:
            return Readings(reduction=ReductionReadings.read(reduction_json(out, instrument_id)))
        case Stage.OPTIMIZE:
            return _allocated(out, instrument_id, cell.strategy)


@dataclass(frozen=True)
class Bench:
    """Where a sweep works: the roots its configs, runs and transcripts land under, and how it runs them.

    ``keep`` holds each cell's artifacts on disk instead of clearing them once their readings are taken,
    which is what a cell worth listening to needs and what a long sweep cannot afford.
    """

    root: Path
    workers: int
    seed: int
    keep: bool = False

    @property
    def configs(self) -> Path:
        """Where the config copy each distinct setting is run under is written."""
        return self.root / "configs"

    @property
    def runs(self) -> Path:
        """Where each cell's stage output lands while its readings are taken off it."""
        return self.root / "runs"

    @property
    def logs(self) -> Path:
        """Where each cell's transcript is kept, which outlives the artifacts it describes."""
        return self.root / "logs"

    @property
    def stages(self) -> Path:
        """Where the reduction prepared once per material is kept, which allocation cells read back."""
        return self.root / "stages"

    def prepared(self, material: Material) -> Material:
        """``material`` as the reduction left it, run once at the shipped config and kept for every cell.

        Raises:
            RuntimeError: when the preparing run leaves no dataset, so every cell reading it would fail
                one at a time with the same reason.
        """
        out = self.stages / material.instrument_id
        reduced = replace(
            material,
            notes_json=out / f"{material.instrument_id}{NOTES_SUFFIX}",
            samples_dir=out / material.instrument_id,
        )
        if reduced.notes_json.is_file() and reduced.samples_dir.is_dir():
            return reduced

        outcome = execute(
            _command(
                Stage.REDUCE,
                material,
                variant(self.configs / _SHIPPED, {}),
                out,
                Invocation(budget_kb=_PREPARE_BUDGET_KB, workers=self.workers, seed=self.seed),
            ),
            self.logs / f"prepare~{material.instrument_id}{_LOG_SUFFIX}".replace(" ", "_"),
        )
        if not outcome.ok:
            raise RuntimeError(f"preparing {material.instrument_id} failed: {failure(outcome)}")

        return reduced

    def _material(self, cell: Cell) -> Material:
        """The dataset ``cell`` reads, preparing the reduction first where it asks for one."""
        if cell.source is Source.REDUCED:
            return self.prepared(cell.material)

        return cell.material

    def run(self, cell: Cell) -> tuple[Outcome, Readings | None]:
        """Run one cell and read what it produced, clearing its artifacts unless they are being kept."""
        config = variant(self.configs / cell.slug, cell.overrides)
        out = self.runs / cell.slug
        command = _command(
            cell.stage,
            self._material(cell),
            config,
            out,
            Invocation(budget_kb=cell.budget_kb, workers=self.workers, seed=self.seed),
        )
        outcome = execute([*command, *_allocation_flags(cell), *cell.flags], self.logs / f"{cell.slug}{_LOG_SUFFIX}")
        readings = _readings(cell, out) if outcome.ok else None
        if not self.keep:
            rmtree(out, ignore_errors=True)

        return outcome, readings
