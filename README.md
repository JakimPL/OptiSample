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

### Reducing on its own

`optisample reduce` runs that whole stage and stops, writing what it decided as a dataset an allocation
run reads back later:

```bash
optisample reduce path/to/Piano.notes.json --budget-kb 96 --out reduced
optisample optimize reduced/Piano.notes.json --budget-kb 96      # picks up from there
```

It takes every ingest and reduction flag `optimize` takes, and writes:

```
reduced/
  Piano.notes.json                 # one entry per played note, pointing at the survivor serving it
  Piano/0000_p060_C4_v100.wav      # one WAV per surviving recording, onset-aligned at the analysis rate
  reduction/Piano/
    reduction.json                 # the whole summary, plus the dedupe key it was produced under
    auditions/p060_C4/             # reference.wav beside every shortlisted encoding, rendered
```

The root is itself a NoteExtractor dataset, so the reduced tree is the resume point for a long run: the
survivors are trimmed to what the material asks of them (the demo's 40 recordings become 5 files, 2.6 MB
down to 340 KB) and the allocation reaches the sweep having paid only the ingest. Allocating from a
reduced dataset reproduces the plan allocating from its source produces, so the round trip costs nothing
in fidelity. The auditions make the shortlist audible before the sweep is paid for.

A dataset reproduces its survivors exactly under the key it was reduced with; reducing it again under a
coarser key projects several identities onto one survivor, which then reports the identity of the first
note that reaches it.

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

Each long stage draws a labelled progress bar on stderr, carrying the count it will reach and an ETA, so
a large instrument states how long it needs while it runs:

```
Probing recordings................ 100%     40/40     [00:00<00:00]
Narrowing stored grids............ 100%      5/5      [00:00<00:00]
Sweeping encodings................ 100%     13/13     [00:00<00:00]
Scoring pitch zones............... 100%     12/12     [00:07<00:00]
```

`--seed` fixes the dither so a run reproduces byte for byte; every other flag above is shared with
`optisample reduce` (see below).

Bars are drawn when stderr is a terminal, so redirected output stays clean; `--no-progress` silences them
at a terminal too. Results keep to stdout either way.

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
