from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from research.store import CellRecord, Store

_MISSING: Final = "."
_WIDTH: Final = 9  # characters one reading is printed in, which holds a share, a second and an objective alike
_NAME: Final = 26
_CLASS: Final = 17

LOOP_FIELDS: Final[tuple[str, ...]] = (
    "loop.looped_share",
    "loop.offers_per_recording",
    "loop.median_start_s",
    "loop.median_cheapest_s",
    "loop.median_stored_share",
    "loop.median_span_ratio",
    "loop.median_seam_step",
    "loop.median_spectral_distance_db",
)

PLAN_FIELDS: Final[tuple[str, ...]] = (
    "plan.objective",
    "measured.objective",
    "measured.written_gap",
    "plan.samples",
    "plan.looped_share",
    "plan.stored_seconds",
    "plan.fill",
    "measured.segmental_snr_db",
)


def _reading(record: CellRecord, dotted: str) -> float | None:
    """The value ``dotted`` names on ``record``, or nothing where the cell's stage left no such reading."""
    group, field = dotted.split(".", 1)
    readings = getattr(record, group)
    if readings is None:
        return None

    value = getattr(readings, field)
    return None if value is None else float(value)


@dataclass(frozen=True)
class Pivot:
    """One family's readings of one field, laid out as materials down and settings across."""

    family: str
    field: str
    labels: tuple[str, ...]
    classes: Mapping[str, str]
    values: Mapping[tuple[str, str], float]

    def render(self) -> str:
        """The table as it is read: a header of settings, then one row per material."""
        header = f"{'instrument':{_NAME}} {'class':{_CLASS}} " + " ".join(f"{label:>{_WIDTH}}" for label in self.labels)
        lines = [f"=== {self.family}  ::  {self.field}", header]
        for name in sorted(self.classes, key=lambda item: (self.classes[item], item)):
            cells = [
                f"{self.values[name, label]:{_WIDTH}.3f}" if (name, label) in self.values else f"{_MISSING:>{_WIDTH}}"
                for label in self.labels
            ]
            lines.append(f"{name:{_NAME}} {self.classes[name]:{_CLASS}} " + " ".join(cells))

        return "\n".join(lines)


def _sorted_labels(labels: set[str]) -> tuple[str, ...]:
    """The settings in the order a reader compares them: by value where they are numbers, by name otherwise."""
    try:
        return tuple(sorted(labels, key=float))
    except ValueError:
        return tuple(sorted(labels))


def pivot(records: Sequence[CellRecord], family: str, field: str) -> Pivot:
    """Every successful reading of ``field`` the cells of ``family`` produced."""
    wanted = [record for record in records if record.family == family and record.ok]
    values = {
        (record.instrument_id, record.label): reading
        for record in wanted
        if (reading := _reading(record, field)) is not None
    }
    return Pivot(
        family=family,
        field=field,
        labels=_sorted_labels({label for _, label in values}),
        classes={
            record.instrument_id: record.material_class for record in wanted if _reading(record, field) is not None
        },
        values=values,
    )


def reasons(records: Sequence[CellRecord], family: str) -> str:
    """What each cell of ``family`` turned down, which is where a gate's effect states itself in words."""
    lines = [f"--- {family}  ::  lacking / rejected-by-gate / infeasible"]
    for record in records:
        if record.family != family or record.loop is None:
            continue

        if record.loop.lacking or record.loop.rejected_by_gate:
            lines.append(
                f"    {record.instrument_id:{_NAME}} {record.label:>8}"
                f"  lacking={record.loop.lacking}  gates={record.loop.rejected_by_gate}"
            )

    for record in records:
        if record.family == family and record.infeasible is not None:
            lines.append(f"    {record.instrument_id:{_NAME}} {record.label:>8}  infeasible: {record.infeasible}")

    return "\n".join(lines)


def _fields(records: Sequence[CellRecord], family: str) -> tuple[str, ...]:
    """The readings worth tabulating for ``family``, read off the stage its cells ran."""
    stages = {record.stage for record in records if record.family == family}
    return LOOP_FIELDS if stages == {"loop"} else PLAN_FIELDS


def report(store: Store, families: Sequence[str], fields: Sequence[str]) -> str:
    """Every table the named families answer, in the order they were asked for."""
    records = list(store.records())
    wanted = families or sorted({record.family for record in records})
    blocks: list[str] = []
    for family in wanted:
        for field in fields or _fields(records, family):
            blocks.append(pivot(records, family, field).render())

        blocks.append(reasons(records, family))

    return "\n\n".join(blocks)


def main(argv: Sequence[str] | None = None) -> None:
    """Print one table per field per family, off the records a sweep left at ``root``."""
    parser = argparse.ArgumentParser(prog="research.report", description="Tabulate a sweep's readings.")
    parser.add_argument("root", type=Path, help="The sweep root holding records.jsonl")
    parser.add_argument("--family", action="append", default=[], help="Families to report, repeatable")
    parser.add_argument("--field", action="append", default=[], help="Dotted readings to tabulate, repeatable")
    args = parser.parse_args(argv)
    print(report(Store(args.root), args.family, args.field))


if __name__ == "__main__":
    main()
