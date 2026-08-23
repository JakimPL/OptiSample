from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from notebooks.utils.runs import Outcome
from research.records import LoopReadings, MeasuredReadings, PlanReadings, ReductionReadings
from research.runner import Cell, Readings, failure

_RECORDS: Final = "records.jsonl"
_HEAD: Final = "HEAD"


def commit() -> str:
    """The revision the working tree stands at, which every record is stamped with.

    A knob's reading is only comparable with another taken on the same code, and this tree moves under a
    sweep, so the revision travels with the reading rather than with the sweep as a whole.
    """
    return subprocess.run(["git", "rev-parse", _HEAD], capture_output=True, text=True, check=True).stdout.strip()


class CellRecord(BaseModel):
    """One cell as the store holds it: what was run, on what, under which setting, and what came back."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sweep: str
    commit: str
    family: str
    label: str
    instrument_id: str
    material_class: str
    notes: int
    tier: str
    stage: str
    strategy: str
    budget_kb: float
    overrides: dict[str, Any]
    flags: list[str]
    elapsed_s: float
    ok: bool
    failure: str | None
    loop: LoopReadings | None
    reduction: ReductionReadings | None
    plan: PlanReadings | None
    measured: MeasuredReadings | None
    infeasible: str | None

    @classmethod
    def of(cls, sweep: str, cell: Cell, outcome: Outcome, readings: Readings | None) -> CellRecord:
        """One record from the cell that produced it and whatever its stage left to read."""
        taken = readings or Readings()
        return cls(
            sweep=sweep,
            commit=commit(),
            family=cell.family,
            label=cell.label,
            instrument_id=cell.material.instrument_id,
            material_class=cell.material.material_class,
            notes=cell.material.notes,
            tier=cell.material.tier.value,
            stage=cell.stage.value,
            strategy=cell.strategy,
            budget_kb=cell.budget_kb,
            overrides=dict(cell.overrides),
            flags=list(cell.flags),
            elapsed_s=outcome.elapsed_s,
            ok=outcome.ok,
            failure=failure(outcome),
            loop=taken.loop,
            reduction=taken.reduction,
            plan=taken.plan,
            measured=taken.measured,
            infeasible=taken.infeasible,
        )


class Store:
    """The sweep's own results, appended one JSON record per line so a long run is readable while it runs."""

    def __init__(self, root: Path) -> None:
        self.path = root / _RECORDS
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CellRecord) -> None:
        """Add one record to the store."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")

    def records(self) -> Iterator[CellRecord]:
        """Every record the store holds, in the order they were run."""
        if not self.path.is_file():
            return

        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield CellRecord.model_validate_json(line)

    def done(self) -> set[tuple[str, str, str, str]]:
        """The cells already answered, so a sweep resumed after an interruption re-runs none of them."""
        return {(record.sweep, record.family, record.label, record.instrument_id) for record in self.records()}


def key(sweep: str, cell: Cell) -> tuple[str, str, str, str]:
    """How a cell is recognised in :meth:`Store.done`."""
    return (sweep, cell.family, cell.label, cell.material.instrument_id)


def rows(store: Store, family: str) -> list[Mapping[str, Any]]:
    """Every successful record of ``family``, flattened into the rows a table is read off."""
    flattened: list[Mapping[str, Any]] = []
    for record in store.records():
        if record.family != family or not record.ok:
            continue

        row: dict[str, Any] = {
            "label": record.label,
            "instrument": record.instrument_id,
            "class": record.material_class,
            "notes": record.notes,
            "elapsed_s": round(record.elapsed_s, 1),
        }
        for name, readings in (
            ("loop", record.loop),
            ("reduction", record.reduction),
            ("plan", record.plan),
            ("measured", record.measured),
        ):
            if readings is not None:
                row |= {f"{name}.{field}": value for field, value in readings.model_dump().items()}

        flattened.append(row)

    return flattened


def dump(store: Store, path: Path) -> None:
    """Write every record the store holds as one JSON array, which is what a report reads."""
    path.write_text(
        json.dumps([json.loads(record.model_dump_json()) for record in store.records()], indent=1),
        encoding="utf-8",
    )
