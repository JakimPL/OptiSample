# Notebooks

Interactive [marimo](https://marimo.io) notebooks for inspecting the pipeline by ear and eye.
They are a **thin** layer: all reusable logic lives in `notebooks/utils/` (type-checked, linted, and
tested like the rest of the code), so the notebook files themselves stay small and are validated by
running them, not by static analysis.

## Run

From the repo root:

```bash
uv run marimo edit notebooks/pipeline.py    # the CLI as controls, its output as an explorer
uv run marimo edit notebooks/explore.py     # samples and the metric layer
uv run marimo run  notebooks/pipeline.py    # read-only app
```

## `pipeline.py` — pipeline console

Every CLI flag as a control and every stage's output as an explorer. A run shells out to
`optisample <command>` exactly as a terminal would, and the invocation behind each button is printed
above it, so anything found here is reproducible from a shell.

**Controls** — the source dataset and run root; subset fraction; budget; tracker format; strategy;
interpolation; dedupe key; content floor; rate ladder and bit depth; looping; ground-truth render;
workers; dither seed. One bundle stands behind every stage, so a control moved once reaches whichever
stage runs next.

**Stages** — `subset` carves the share of a dataset that still spans its pitch and velocity ranges,
`loop` settles the loop each recording is stored around, `reduce` writes the survivors an allocation
picks up from, and `optimize` allocates the budget. Each stage prefers what the one before it left:
looping reads the subset once it exists, reduction reads the looped dataset, and allocation reads the
reduced one, so the sweep is reached having paid only the ingest. Every transcript is filed under
`<run root>/logs/`.

**Looping explorer** — the loop each recording keeps with the seam step and timbre distance it measured,
the recordings stored over the span they play instead, and every cheaper candidate the ladder climbed
past beside the gate it fell outside. Then the **auditions**: each recording beside its loop played out
— wrapped several times under the fitted decline, which is how a seam step or a level pulse is heard.

**Reduction explorer** — each axis read as `before -> after`, every kept recording measured against
the material its pitch asks of it, the format each pitch is stored at, and the survivors written. Then
the **auditions**: the recording a pitch was judged against, beside every encoding the sweep runs
rendered to audio — the stored format made audible before the sweep is paid for.

**Allocation explorer** — the byte accounting, the encoding each kept item spends its bytes on (one
table shape for both strategies), and where every item landed on the rate-distortion plane. Then the
per-note fidelity summing back to the plan's objective, and per pitch the **A/B pair** the objective
actually scored: the recording and the module's reconstruction, side by side with spectrograms and
the sub-scores behind the number.

The run root defaults to the repository root, so `subset/`, `looped/`, `reduced/` and `artifacts/` from
a terminal run are picked up as they are. The explorers read whatever is on disk, so a long run started
in a shell can be inspected here without re-running it.

## `explore.py` — sample & metric inspector

- **Per instrument:** budget roll-up (full-length 16-/8-bit storage vs the byte budget) and the
  material (notes the song actually plays).
- **Per sample:** audio playback, waveform, dynamic-range-floored spectrogram, and a descriptor +
  footprint table.
- **Degradation lab:** apply a preview degradation (bit-depth, lowpass, gain, resample) and hear the
  original vs. the result while the composite fidelity, per-metric breakdown, and diagnostics update.
- **Smoke-test panel:** the fixed P1 degradation set with a stacked bar decomposing the composite
  (bar height ≈ fidelity) — the interactive form of the P1 smoke test.

On first run it generates a synthetic demo project under `notebooks/_demo/` (git-ignored) — one
`.notes.json` + samples directory per preset — and loads them into a combined manifest. Point the
*demo directory* field at your own directory of `.notes.json` files to inspect real recordings.
