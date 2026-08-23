# Research harness

The sweeps behind [`docs/tuning.md`](../docs/tuning.md). Every figure that document states comes from a
record this harness wrote, so a claim can be re-checked rather than believed.

## What it does

A **cell** is one stage run over one material under one setting. Most knobs have no CLI flag, and
`--config` reads a whole directory with no merge or fallback, so a cell copies the shipped tree, sets one
key, and runs against the copy. The copy is loaded before the run starts, which is what makes a mistyped
key or an out-of-bounds value fail immediately instead of after the minutes a run costs.

Runs decompose by stage, which is what keeps a grid affordable:

| Sweeping | Stage the cell runs | What it reads |
|---|---|---|
| loop knobs | `loop` | `loops.json` — offers, rejections and the gate each fell outside |
| reduction knobs | `reduce` | `reduction.json` — survivors, lengths, the rung each pitch is stored at |
| allocation knobs | `optimize` | `plan.json` and `metrics.json` |

An allocation cell reads a reduction prepared once per material at the shipped config, so its minutes go
on the allocation rather than on repeating the ingest.

## Running one

```python
from pathlib import Path

from research import sweep
from research.materials import Corpus
from research.runner import Bench, Cell, Source, Stage
from research.store import Store

corpus = Corpus.load()
by_id = {material.instrument_id: material for material in corpus.sources()}
root = Path("...")                       # a scratch root, not the repo
bench = Bench(root=root, workers=6, seed=137)
store = Store(root)

sweep(
    "length",
    [
        Cell(
            family="length",
            label=f"max_length_s={seconds}",
            material=by_id["Ensemble - Strings C"],
            stage=Stage.OPTIMIZE,
            budget_kb=128,
            source=Source.REDUCED,
            overrides={"reduce.trim.max_length_s": seconds},
        )
        for seconds in (1.0, 2.0, 3.0, 5.0)
    ],
    bench,
    store,
    lanes=4,
)
```

Results append to `<root>/records.jsonl`, one JSON object per cell, each stamped with the revision it was
run at — this tree moves, and a knob's reading is only comparable with another taken on the same code. A
sweep re-run skips the cells already answered, so an interrupted grid resumes and a widened one costs
only what it added.

## Sizing a sweep

Each stage caps its own fan-out at the items it holds, so a small instrument leaves cores idle however
many workers it is given: `workers=6` reaches within a few percent of `workers=16` on a 17-note
instrument, and the throughput comes from `lanes` instead. Allocation is the expensive stage, and its
cost is the composite: about 90 ms per scored class per encoding on a one-second clip at 48 kHz,
multiplied by the zones, the representatives each offers and the loops each recording carries.

`corpus.yaml` names where the material sits and the acoustic class each instrument stands for, which is
what a claim is tagged with so a finding states the material it holds across.
