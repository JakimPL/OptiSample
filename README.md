# OptiSample

OptiSample turns an instrument's recorded sample grid into a byte-budgeted tracker module — Impulse
Tracker (`.it`) or FastTracker 2 (`.xm`): it stores one sample per key, collapses the velocity axis to
one representative per pitch, and reproduces dynamics through a velocity→volume map.

## Requirements

[`trackmod`](trackmod) is a tracker format library that ships as a git submodule, so a fresh clone fetches it before installing:

```bash
git submodule update --init
uv sync
```

Ground-truth renders use `openmpt123` (optional — the pipeline falls back to a numpy surrogate when it
is absent):

```bash
sudo apt install -y openmpt123
```

## Input: NoteExtractor output

OptiSample consumes [NoteExtractor](../NoteExtractor) output directly as its native input:

- a **samples directory** of per-note WAVs named `{index}_p{pitch}_v{velocity}_...wav`, and
- a **`.notes.json`** manifest listing each note's `pitch`, `velocity`, `cc_averages`, and a nested
  `render` block (`index`, `start_seconds`, `release_end_seconds`).

Each note becomes both a recorded sample and a played event; the join key is `render.index` matched
against each WAV filename's leading index token. Reducing the grid to one recording per key is the
optimizer's job: it keeps the shortest recording that still covers the notes that key has to play,
under the identity `reduce.yaml` names (`pitch`, `pitch_velocity`, or `pitch_velocity_cc`).

## Reduction: shrinking the problem before it is solved

A whole pre-optimization stage (`src/opticonfig/reduce.yaml`) runs before any byte is allocated, and
every report and artifact tree states what it left behind:

- **Deduplication** keeps one recording per identity — the shortest that still covers its key's longest
  note under the transposition headroom, ranked from the WAV headers alone.
- **Event merging** collapses the notes at a pitch into the classes that reconstruct identically
  (same reference recording, same mapped volume, same scored length), widened onto a geometric duration
  grid by `duration_bucket_ratio`.
- **The bandwidth pre-pass** measures the band each recording occupies and the interval it is played
  at, then keeps the `candidates` rate/depth vertices priced nearest what the budget affords that key.
- **Pitch-zone grouping** bounds its own search with `max_zone_semitones` and reuses a scored
  `(representative, encoding, key)` reconstruction across every zone containing it.

Each strategy's `report.txt` opens with a `Reduction (pre-optimization)` block reading `before -> after`
per axis, and `reduction.json` records every kept recording (with the material it covers) and every
pitch's shortlist. A recording too short for its notes is named there rather than quietly truncated.

## Usage

Optimize a `.notes.json` into an inspectable artifact tree:

```bash
optisample optimize path/to/Piano.notes.json \
    --samples-dir path/to/Samples/Piano \
    --budget-kb 96 \
    --out artifacts
```

`--samples-dir` defaults to the notes file's sibling `<name>/` directory, and `--instrument-id`
defaults to the `.notes.json` base name. `--pre-roll-ms` / `--post-roll-ms` mirror the trimmer's
padding (the pre-roll is trimmed as each sample's lead-in so frame 0 lands on the note onset). Other
flags: `--format {it,xm}`, `--strategy {both,grouped,ungrouped}`, `--no-render`, `--rate`/`--depth`
(repeatable sweep values), `--no-loop`, `--interpolation`, `--seed`. `--dedupe-key` and `--candidates`
override the two reduction knobs worth varying per run (see below).

`--format` overrides `src/opticonfig/tracker.yaml`, which also sets the compliance level the module is
graded against and each format's own settings. Impulse Tracker numbers the whole ten-octave keyboard
(MIDI 12–131); FastTracker 2 stops at MIDI 107, and a higher key is reported before anything is
written.

Generate a synthetic demo dataset (one `.notes.json` + samples directory per preset) to exercise the
pipeline without real recordings:

```bash
optisample synth demo_out --sample-rate 22050
optisample optimize demo_out/piano.notes.json --budget-kb 96 --no-render
```
