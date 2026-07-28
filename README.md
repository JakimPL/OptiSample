# OptiSample

OptiSample turns an instrument's recorded sample grid into a byte-budgeted tracker module — Impulse
Tracker (`.it`) or FastTracker 2 (`.xm`): it stores a recording per key per velocity band, writes one
instrument for each band so a note's dynamic names the one it plays through, and reproduces the
dynamics within a band through a velocity→volume map. How many bands a budget is worth spending on is
the optimizer's decision, taken against the same objective everything else is (see
[Velocity layers](#velocity-layers-trading-samples-for-dynamics)).

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

### Taking a slice first

A thousand-note instrument is worth iterating on at a tenth of its size before it is run whole, and
`optisample subset` writes that tenth as a dataset of the same shape:

```bash
optisample subset path/to/Piano.notes.json --fraction 0.1 --out subset
optisample reduce subset/Piano.notes.json --budget-kb 512      # picks up from there
```

Notes are grouped by pitch, each pitch is allotted a share, and the notes it contributes are the ones
its velocities spread evenly over — so a slice covers the keyboard the source plays and, within each
pitch, the dynamics it was played across. Every pitch is represented before any pitch takes a second
note; what is left over then goes round the pitches busiest first. Each kept note is written exactly
as the source states it, so the slice measures the same material at the same lengths, and the output
root is itself a NoteExtractor dataset.

Because a pitch's picks start from its quietest and loudest recordings, a slice holds one recording
per identity and leaves deduplication nothing to collapse. That is the honest reading of a slice's
`recordings: n listed -> n kept` line: the dedupe axis is exercised by the whole dataset, not by a
tenth of it.

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

## Gain staging: storing hot and getting the level back

An 8-bit grid has 256 steps, so how much of it a recording occupies decides how much of the recording
survives. Every sample is therefore stored hot — normalized to `headroom_db` (`src/opticonfig/quantize.yaml`)
under full scale, far enough down that the dither has somewhere to go — and the level it was lifted from
is given back on playback. Normalization reads the span the sample actually stores, after the trim or the
loop, so a peak in a discarded tail leaves the stored sample exactly where it asked to be.

Where that level comes back depends on the format. Impulse Tracker keeps a 0–64 multiplier per sample, and
that is where the balance is restored — so a naturally quiet key sounds quiet again while still spending
its whole depth on its own recording. The multiplier carries only the part the pattern does not: the
velocity→volume map already states the dynamic of the velocity each sample was recorded at, so that share
is divided out and what remains is the balance between the recordings themselves. FastTracker 2 has no such
field, so an XM run normalizes every clip against the instrument's loudest peak instead and the balance
rides in the PCM. That reference is fixed before the sweep, because how hot a sample is stored is part of
what the objective measures.

Ahead of the quantizer sits an optional soft-knee compressor (`src/opticonfig/dynamics.yaml`), reading its
threshold against each clip's own peak so it shapes crest factor rather than level. It is a **swept axis**,
not a preprocessing step: the grid offers each 8-bit encoding both compressed and plain, prices them the
same way, and the objective picks. A 16-bit encoding is enumerated uncompressed, since its noise floor
already sits below anything compression could protect. The `comp` column in `report.txt` and the `_c` in an
audition filename say which way each sample went.

## Velocity layers: trading samples for dynamics

A piano's timbre changes further with how hard a key is struck than its level does, so reconstructing a
pianissimo note by scaling a fortissimo recording down is the largest distortion left once the encoding
is settled. A **velocity layer** is the move that answers it: the velocity axis is cut into bands, each
band stores its own recording of the keys played in it, and a note sounds through the band its dynamic
falls in.

The tracker formats make that an instrument-level split — an IT or XM keymap is keyed by note alone — so
each band is written as one instrument and the pattern names the instrument beside the note. The price is
paid honestly: every layer adds an instrument record to the budget before a frame of audio is stored, so a
three-layer plan starts from fewer sample bytes than a one-layer plan of the same size.

Which split to buy is the optimizer's decision, taken against the objective the whole pipeline shares.
`src/opticonfig/layers.yaml` sets the terms:

```yaml
max_layers: 3    # velocity bands stored per key == instruments written; 1 keeps one recording per key
nodes: 6         # elementary velocity cells bands are assembled from; the cost dial ((nodes+1)/2)
min_gain: 0.02   # relative objective gain an extra layer must buy to be preferred to fewer
```

The cells are cut on the material's own playing time, so the axis is resolved finely where the instrument
spends it, and each band keeps the recording nearest the loudest dynamic *it* covers. Every split into at
most `max_layers` runs of cells is solved in full and the best objective wins, provided it beats the
plainer plan by `min_gain` per extra layer. A material playing each key at a single dynamic earns no split
and pays for none. `--max-layers` overrides the cap for one run.

The grouped `report.txt` states the split band by band — the dynamics each band answers for, the keys its
samples reach, and the bytes and distortion they carry — and both strategies record the same under
`plan.json`'s `layers`, with every zone naming the layer it belongs to. The A/B renders are filed per band
under `compare/<band>/`, so a key stored three times is heard three ways.

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
override the two reduction knobs worth varying per run (see below), and `--max-layers` the velocity
bands a key may store.

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

The stages that fan out — narrowing each pitch's stored grid, rendering each demo note — share their work
across processes. `--workers N` sets how many; `0` (the shipped `src/opticonfig/runtime.yaml` value) asks
for one per core, and each stage caps that at the items it holds, so a five-pitch stage starts five
workers. `--workers 1` carries the whole run in the calling process, which is what `--profile` reads end
to end and what a machine you are also working on stays responsive under. A run states the same reduction
and the same plan whichever count it was given: every pitch is narrowed from its own clip against a fixed
dither seed, and each demo note's phases are drawn in note order before any of them are rendered.

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

The `dynamics` preset plays each of its three keys across four velocities, which is the material
velocity layering pays on — the grouped strategy stores a recording per band and writes one instrument
for each, so the module plays a soft note through the sample recorded softly:

```bash
optisample optimize demo_out/dynamics.notes.json --budget-kb 192 --strategy grouped
```

## Driving it from a notebook

[`notebooks/pipeline.py`](notebooks/pipeline.py) is the same CLI with its flags as controls and its
output as an explorer: run `subset`, `reduce` and `optimize` from buttons, then read the reduction
axes, listen to every shortlisted encoding, and compare each pitch's reconstruction against the
recording the objective scored it on. It shells out to `optisample` and prints the invocation behind
every button, so what it finds is reproducible from a terminal, and its explorers read whatever is
already on disk — a long run started in a shell can be inspected without re-running it.

```bash
uv run marimo edit notebooks/pipeline.py
```

See [`notebooks/README.md`](notebooks/README.md) for what each panel shows.
