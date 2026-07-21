# Architecture & Ownership

Optisample turns an instrumental's recorded samples into an Impulse Tracker (`.IT`) module under a hard byte
budget. This document says which part of the package owns what, so shared logic has one home and new code lands
in the right place.

## Package map (`src/optisample/`)

| Module / subpackage | Owns | Depends on |
|---|---|---|
| `music.py` | Pitch primitives: note names, `semitone_ratio`, MIDI/tuning constants. A leaf shared by everyone. | nothing in-package |
| `model.py` | Manifest DTOs: `Manifest`, `ProjectSpec`, `InstrumentSpec`, `SourceSample`, `NoteEvent`. | pydantic |
| `config/` | Pydantic **schema** for every tunable parameter; values live in `src/opticonfig/*.yaml`. Loaded once via `load_config`. | pydantic |
| `dsp/` | Signal primitives (`spectral`, `resample`, `quantize`, `loop`, `timebase`) and the surrogate codec — the `surrogate/` subpackage (`params`, `sample`, `encode`, `render`) exposing `encode`/`render`, `StoredSample`, `EncodingParams`, `MAX_VOLUME`. | `config`, `music`, `metrics.size` |
| `metrics/` | Fidelity measurement (`composite`, `spectral`, `timbre`, `diagnostics`, `preprocess`) and the byte-`size` model. | `config`, `dsp` primitives |
| `optimize/` | The allocation pipeline (see below). | `dsp`, `metrics`, `model`, `io`, `config`, `music` |
| `io/` | The file/format boundary: `audio` (WAV), `manifest`, `it_format` (declarative IT record layout), the `it_writer/` subpackage (IT binary serializer — `constants`, `samples`, `instruments`, `patterns`, `module`), `it_read` (round-trip), `render` (openmpt123 wrapper). | `config`, `metrics.size`, `music`, `dsp.surrogate` (`MAX_VOLUME`) |
| `synth/` | Synthetic demo-audio generation: the `archetypes` module (pure archetype synthesis) and `generate` (render a preset grid to WAVs + a `manifest.yaml`). | `config`, `music`, `io`, `model` |
| `calibrate.py` | Surrogate-vs-openmpt calibration diagnostics. | `optimize`, `io`, `metrics` |
| `artifacts/` | Inspection-artifact dumper (module + report + plan + per-note A/B WAVs + metrics). | `optimize`, `io`, `metrics` |
| `cli.py`, `__main__.py` | Entry point: load config once, build settings, thread them down. | everything |

### Inside `optimize/`

- `operating_points.py` — per-sample rate-distortion sweep + lower convex hull (`SourceClip`, `OperatingPoint`).
- `tasks.py` — pitch-task construction and the **single** reconstruction scorer (the objective).
- `velocity_map.py` — the loudness-matched velocity→volume map.
- `knapsack.py` — the MCKP byte-budget DP (exact + Lagrangian).
- `grouping/` — pitch-zone grouping as a subpackage: `cost_model` (zone-option enumeration), `solve` (exact partition + allocation DP), and the `__init__` orchestration + `GroupedInstrumentPlan`.
- `orchestrate/` — the ungrouped end-to-end pipeline as a subpackage: `cost_model` (per-pitch rate-distortion sweep → knapsack items), `solve` (MCKP allocation + per-pitch plans), and the `__init__` orchestration plus the shared `OptimizeSettings`/`prepare_run`/`load_instrument_audio` that pitch-zone grouping reuses.
- `export.py` — plan → `ITModule` bridge.
- `plans/` — plan value objects (`budget`, `ungrouped`, `grouped`), grouped as a subpackage rather than one bag-of-classes module.
- `report.py` — human-readable text reports for both strategies.

## Rules

1. **Shared primitives have one home.** Pitch/music math lives in `music.py`; the IT record layout in
   `io/it_format.py`; byte-size math in `metrics/size.py`; the IT volume ceiling (`MAX_VOLUME`, 0x40) sits with
   the surrogate codec that renders against it and is imported from there. Import from the owner — do not re-derive.
2. **Config is schema, not values.** No tunable value defaults live in Python; YAML in `src/opticonfig/` is the
   single source. Config loads once at the entry point and threads down as explicit arguments.
3. **One responsibility per module.** Keep data shapes, algorithms, and reporting/serialization separated. When
   a group of related types would otherwise become a bag of classes, make it a small subpackage.
4. **No pass-through re-exports.** `__init__` exposes only its own subtree's public API; do not add re-export
   shims so other modules can import through them. Import shared helpers directly from their owner.
5. **Behaviour is frozen unless a change is requested.** Refactors keep the optimizer output bit-identical; the
   demo end-to-end numbers are the regression contract (see `docs/guidelines.md` and the plan's verification).
6. **External tools sit behind a typed, probed boundary.** `openmpt123` is wrapped in `io/render.py` with a
   runtime `openmpt123_available()` probe; callers degrade gracefully instead of branching on the environment.
