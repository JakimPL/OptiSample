from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

from optisample.io.note_extractor import NOTES_SUFFIX

_PACKAGE: Final = "optisample"
_SUBSET_ROOT: Final = "subset"
_REDUCED_ROOT: Final = "reduced"
_ARTIFACTS_ROOT: Final = "artifacts"
_LOGS_ROOT: Final = "logs"
_LOG_SUFFIX: Final = ".log"
_WHOLE: Final = 1.0


@dataclass(frozen=True)
class Dataset:
    """A NoteExtractor dataset: the manifest naming every note, and the recordings it joins to."""

    notes_json: Path
    samples_dir: Path

    @classmethod
    def rooted(cls, root: Path, instrument_id: str) -> Dataset:
        """The dataset a stage writes under ``root``: ``<id>.notes.json`` beside its ``<id>/`` recordings."""
        return cls(notes_json=root / f"{instrument_id}{NOTES_SUFFIX}", samples_dir=root / instrument_id)

    @property
    def ready(self) -> bool:
        """Whether both halves of the dataset are on disk, so a stage may read it."""
        return self.notes_json.is_file() and self.samples_dir.is_dir()


@dataclass(frozen=True)
class Fields:
    """Every CLI field the notebook exposes as a control, in the shape the commands read them.

    One bundle stands behind all three stages, so a control moved once reaches whichever of them runs
    next and the reduced dataset an allocation picks up was narrowed under the same knobs.
    """

    source: Dataset
    instrument_id: str
    out_root: Path
    budget_kb: float
    fraction: float
    tracker_format: str
    strategy: str
    interpolation: str
    dedupe_key: str
    candidates: int
    rates: tuple[int, ...]
    depths: tuple[int, ...]
    loop: bool
    workers: int
    seed: int
    render: bool

    @property
    def subset_root(self) -> Path:
        """The directory ``subset`` writes its dataset into."""
        return self.out_root / _SUBSET_ROOT

    @property
    def reduced_root(self) -> Path:
        """The directory ``reduce`` writes its dataset, its reduction report and its auditions into."""
        return self.out_root / _REDUCED_ROOT

    @property
    def subset(self) -> Dataset:
        """Where ``subset`` writes the share of the source spanning its pitch and velocity ranges."""
        return Dataset.rooted(self.subset_root, self.instrument_id)

    @property
    def reduced(self) -> Dataset:
        """Where ``reduce`` writes the survivors an allocation picks up from."""
        return Dataset.rooted(self.reduced_root, self.instrument_id)

    @property
    def artifacts(self) -> Path:
        """The instrument directory ``optimize`` writes each strategy's plan under."""
        return self.out_root / _ARTIFACTS_ROOT / self.instrument_id

    @property
    def takes_whole_source(self) -> bool:
        """Whether the subset stage would keep every note, leaving the source worth reading directly."""
        return self.fraction >= _WHOLE


def discover(root: Path) -> Dataset | None:
    """The first complete NoteExtractor dataset sitting under ``root``, in name order.

    A stage leaves its output as ``<id>.notes.json`` beside ``<id>/``, so looking for that pair is how
    the notebook opens on what a previous run already produced.
    """
    for notes_json in sorted(root.glob(f"*{NOTES_SUFFIX}")):
        dataset = Dataset(notes_json=notes_json, samples_dir=notes_json.parent / notes_json.name[: -len(NOTES_SUFFIX)])
        if dataset.ready:
            return dataset

    return None


def _ingest_flags(fields: Fields, dataset: Dataset) -> list[str]:
    """The flags ``optimize`` and ``reduce`` share, as the notebook's controls stand."""
    flags = [
        str(dataset.notes_json),
        "--samples-dir",
        str(dataset.samples_dir),
        "--instrument-id",
        fields.instrument_id,
        "--budget-kb",
        f"{fields.budget_kb:g}",
        "--format",
        fields.tracker_format,
        "--interpolation",
        fields.interpolation,
        "--dedupe-key",
        fields.dedupe_key,
        "--candidates",
        str(fields.candidates),
        "--seed",
        str(fields.seed),
        "--workers",
        str(fields.workers),
    ]
    for rate in fields.rates:
        flags += ["--rate", str(rate)]

    for depth in fields.depths:
        flags += ["--depth", str(depth)]

    if not fields.loop:
        flags.append("--no-loop")

    return flags


def subset_command(fields: Fields) -> list[str]:
    """The ``optisample subset`` invocation carving the notebook's share out of the source dataset."""
    return [
        sys.executable,
        "-m",
        _PACKAGE,
        "subset",
        str(fields.source.notes_json),
        "--samples-dir",
        str(fields.source.samples_dir),
        "--instrument-id",
        fields.instrument_id,
        "--fraction",
        f"{fields.fraction:g}",
        "--out",
        str(fields.subset_root),
    ]


def reduce_command(fields: Fields, dataset: Dataset) -> list[str]:
    """The ``optisample reduce`` invocation running the pre-optimization stage over ``dataset``."""
    return [
        sys.executable,
        "-m",
        _PACKAGE,
        "reduce",
        *_ingest_flags(fields, dataset),
        "--out",
        str(fields.reduced_root),
    ]


def optimize_command(fields: Fields, dataset: Dataset) -> list[str]:
    """The ``optisample optimize`` invocation allocating the budget across ``dataset``."""
    command = [
        sys.executable,
        "-m",
        _PACKAGE,
        "optimize",
        *_ingest_flags(fields, dataset),
        "--out",
        str(fields.out_root / _ARTIFACTS_ROOT),
        "--strategy",
        fields.strategy,
    ]
    if not fields.render:
        command.append("--no-render")

    return command


@dataclass(frozen=True)
class Outcome:
    """What one staged command did: how it ended, how long it took, and everything it printed."""

    command: tuple[str, ...]
    returncode: int
    output: str
    elapsed_s: float
    log: Path

    @property
    def ok(self) -> bool:
        """Whether the stage completed, so the artifacts it writes are there to read."""
        return self.returncode == 0

    @property
    def shell(self) -> str:
        """The same invocation as a line to paste into a terminal, with ``optisample`` as its program."""
        return shlex.join(["optisample", *self.command[3:]])


def execute(command: Sequence[str], log: Path) -> Outcome:
    """Run one staged command to completion, keeping everything it printed in ``log`` and in the result.

    Stages report to whichever stream fits what they say -- results to stdout, progress to stderr -- so
    both are captured into the one transcript that reads the way a terminal would show it.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    completed = subprocess.run(list(command), capture_output=True, text=True, check=False)
    output = completed.stdout + completed.stderr
    log.write_text(output, encoding="utf-8")
    return Outcome(
        command=tuple(command),
        returncode=completed.returncode,
        output=output,
        elapsed_s=perf_counter() - started_at,
        log=log,
    )


def run_stage(stage: str, command: Sequence[str], fields: Fields) -> Outcome:
    """Run ``command`` and file its transcript under the run root as ``logs/<stage>.log``."""
    return execute(command, fields.out_root / _LOGS_ROOT / f"{stage}{_LOG_SUFFIX}")


def allocation_source(fields: Fields) -> Dataset:
    """The dataset an allocation reads: the reduced one once it exists, else what reduction would read.

    Allocating from a reduced dataset reproduces the plan its source produces while reaching the sweep
    having paid only the ingest, so it is the one to prefer whenever a reduce run has already left it.
    """
    if fields.reduced.ready:
        return fields.reduced

    return reduction_source(fields)


def reduction_source(fields: Fields) -> Dataset:
    """The dataset reduction reads: the subset once it exists, else the source as it was given."""
    if fields.takes_whole_source or not fields.subset.ready:
        return fields.source

    return fields.subset
