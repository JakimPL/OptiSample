from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from research import sweep
from research.loops import cells as loop_cells
from research.loops import one_shot, prefix_worth
from research.materials import CATALOGUE, Corpus
from research.runner import Bench, Cell
from research.store import Store

Grid = Callable[[Corpus], tuple[Cell, ...]]

SWEEPS: Final[Mapping[str, Grid]] = {"loops": loop_cells, "one_shot": one_shot, "prefix_worth": prefix_worth}

_LANES: Final = 4  # cells run beside each other, which is what fills a machine a small stage leaves idle
_WORKERS: Final = 5  # processes one cell fans out over
_SEED: Final = 137


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research", description="Run one tuning sweep and store its readings.")
    parser.add_argument("sweep", choices=sorted(SWEEPS), help="Which grid to run")
    parser.add_argument("--root", type=Path, required=True, help="Where configs, runs, logs and records land")
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE, help="Corpus catalogue to read materials from")
    parser.add_argument("--lanes", type=int, default=_LANES, help="Cells run beside each other")
    parser.add_argument("--workers", type=int, default=_WORKERS, help="Processes one cell fans out over")
    parser.add_argument("--seed", type=int, default=_SEED, help="Seed every cell is run under")
    parser.add_argument("--keep", action="store_true", help="Hold each cell's artifacts instead of clearing them")
    parser.add_argument("--only", action="append", default=[], help="Families to run, repeatable (default: all)")
    return parser


def _wanted(cells: Sequence[Cell], families: Sequence[str]) -> tuple[Cell, ...]:
    """The cells a run asks for, which is every one of them where it named no family."""
    if not families:
        return tuple(cells)

    return tuple(cell for cell in cells if cell.family in set(families))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the named sweep over the catalogued corpus, appending each cell's readings as it finishes."""
    args = _parser().parse_args(argv)
    corpus = Corpus.load(args.catalogue)
    bench = Bench(root=args.root, workers=args.workers, seed=args.seed, keep=args.keep)
    sweep(
        args.sweep,
        _wanted(SWEEPS[args.sweep](corpus), args.only),
        bench,
        Store(args.root),
        lanes=args.lanes,
    )


if __name__ == "__main__":
    main()
