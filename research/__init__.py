from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Final

from research.runner import Bench, Cell, Source
from research.store import CellRecord, Store, key

_HEADLINE: Final = 92  # characters a cell's line is held to, so a long sweep reads as a column


def _headline(record: CellRecord) -> str:
    """What one finished cell says on screen: how it went, and the reading its family is read by."""
    mark = "ok  " if record.ok else "FAIL"
    reading = ""
    if record.infeasible is not None:
        reading = "infeasible"
    elif record.plan is not None:
        reading = f"objective {record.plan.objective:9.4f}  samples {record.plan.samples:3}"
    elif record.loop is not None:
        reading = f"looped {record.loop.looped_share:6.1%}  offers {record.loop.offers:4}"
    elif record.reduction is not None:
        reading = f"kept {record.reduction.kept_recordings:4}  covering {record.reduction.covering_share:6.1%}"

    line = f"{mark} {record.label:22} {record.instrument_id:26} {reading}"
    return f"{line[:_HEADLINE]}  [{record.elapsed_s:6.1f}s]"


def sweep(name: str, cells: Sequence[Cell], bench: Bench, store: Store, *, lanes: int) -> None:
    """Run every cell of ``name`` the store has no answer for, ``lanes`` of them at a time.

    Cells already recorded are skipped, so a sweep interrupted part-way resumes where it stopped and a
    grid widened afterwards costs only the cells it added. Each stage caps its own fan-out at the items
    it holds, so a small instrument leaves cores idle however many workers it is given; running several
    cells beside each other is what fills the machine instead.

    Every reduction a cell reads is prepared first and in order, so the lanes share one copy of each
    rather than racing to write it.
    """
    answered = store.done()
    pending = [cell for cell in cells if key(name, cell) not in answered]
    print(f"{name}: {len(pending)} cells to run, {len(cells) - len(pending)} already answered", flush=True)
    for material in {
        cell.material.instrument_id: cell.material for cell in pending if cell.source is Source.REDUCED
    }.values():
        bench.prepared(material)

    guard = Lock()
    done = 0

    def answer(cell: Cell) -> None:
        nonlocal done
        outcome, readings = bench.run(cell)
        record = CellRecord.of(name, cell, outcome, readings)
        with guard:
            done += 1
            store.append(record)
            print(f"[{done:3}/{len(pending):3}] {_headline(record)}", flush=True)

    with ThreadPoolExecutor(max_workers=lanes) as pool:
        list(pool.map(answer, pending))
