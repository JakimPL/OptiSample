# Notebooks

Interactive [marimo](https://marimo.io) notebooks for inspecting the pipeline by ear and eye.
They are a **thin** layer: all reusable logic lives in `notebooks/utils/` (type-checked, linted, and
tested like the rest of the code), so the notebook files themselves stay small and are validated by
running them, not by static analysis.

## Run

From the repo root:

```bash
uv run marimo edit notebooks/explore.py     # interactive
uv run marimo run  notebooks/explore.py     # read-only app
```

On first run, `explore.py` generates a synthetic demo project under `notebooks/_demo/`
(git-ignored) and loads its manifest. Point the *manifest.yaml* field at your own project to
inspect real recordings instead.

## `explore.py` — sample & metric inspector

- **Per instrument:** budget roll-up (full-length 16-/8-bit storage vs the byte budget) and the
  material (notes the song actually plays).
- **Per sample:** audio playback, waveform, dynamic-range-floored spectrogram, and a descriptor +
  footprint table.
- **Degradation lab:** apply a preview degradation (bit-depth, lowpass, gain, resample) and hear the
  original vs. the result while the composite fidelity, per-metric breakdown, and diagnostics update.
- **Smoke-test panel:** the fixed P1 degradation set with a stacked bar decomposing the composite
  (bar height ≈ fidelity) — the interactive form of the P1 smoke test.

Rendering the *optimized* IT instrument is deferred to a later phase; this notebook covers the
inputs and the measurement layer only.
