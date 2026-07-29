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
- **Trimming** bounds every kept recording at `max_length_s` and cuts it where its decay falls under
  `tail_floor`, so a note recorded into thirty seconds of room tone is stored and scored over the
  stretch that sounds. A recording whose peak never reaches `silence_floor` never sounded at all: it is
  left out of the dataset, and any pitch that strips of every recording loses its notes with it, both
  counted in the run's own output and in `reduction.json`.
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

## The objective: what a note's distortion is worth

Every stage above competes for the same number, so it is worth stating exactly what that number counts.
A reconstruction is scored against the recording it came from with both sides loudness-matched, which
makes the fidelity a **relative** reading: how wrong the note sounds measured against itself. Left there,
a pianissimo note's error would count for as much as a fortissimo note's, though one of them is 30 dB
further down in the mix.

So each class's distortion is scaled by the energy of the stretch of recording it is scored over, raised
to `energy_exponent` (`src/opticonfig/optimize.yaml`), and by the playing time the material spends on it:

```yaml
energy_exponent: 0.5   # 0.0 prices every note alike | 0.5 amplitude | 1.0 energy | 0.3 perceived loudness
```

Full scale weighs 1.0, so the exponent alone fixes how steeply a quiet note counts for less — at the
shipped `0.5` a note 30 dB down carries 0.032 of the weight the loudest note does, and at `1.0` it carries
0.001. The two extremes are both defensible and they buy different plans, which is why it is a knob: `1.0`
is the honest total-error-energy reading and will spend almost nothing on the quiet end; `0.3` follows the
loudness an ear reports and keeps it in the running. `0.0` reproduces the time-only weighting.

Because the value changes what the objective means, both `report.txt` (`energy^0.5-weighted`, beside the
objective) and `plan.json` (`energy_exponent`) record it, so two runs are only compared when they were
measured the same way.

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

The grouped `report.txt` states the written instruments one row each — the dynamics and keys each answers
for, and the bytes and distortion they carry — and both strategies record the same under `plan.json`'s
`instruments`, with every zone naming the layer it belongs to. The A/B renders are filed per band under
`compare/<band>/`, so a key stored three times is heard three ways.

## The keyboard a written instrument answers

An instrument is recorded over the keys its material plays, which on a real slice is a stretch in the
middle of the keyboard. Every other key is answered too, from the recording nearest it: the lowest
recording reaches down toward the bottom key, the highest up toward the top, and each stretch between two
of them is served by whichever side is closer, with an equal distance handed to the lower one — the
direction `transposition_headroom_semitones` already reserved bandwidth for. A tracker states a
transposition as the note it sounds, so a recording reaches five octaves below its own key and just under
five above; a key past that goes to the next recording that can name it. The keys the material played keep
their own recording exactly as the plan allocated it, and a filled key sounds its sample at the same
interval its own keys do, so one sample stays tuned once however many keys reach it.

This is what makes a written instrument playable outside the material it was measured on. It also settles
a format asymmetry: Impulse Tracker sounds silence at a key naming no sample, while FastTracker 2, whose
keymap numbers every key to a real slot, would otherwise sound the instrument's first sample there at that
sample's own tuning. Both now answer with the nearest recording. The report and `plan.json` state the
coverage:

```
Keyboard:      120 of 120 keys answered  (5 played, 115 filled from the nearest recording)
```

## Instrument slots: what a format numbers inside one instrument

Since a keymap is keyed by note alone, every axis a plan splits becomes a choice of instrument. Velocity
bands are the first: one instrument each. The format's own sample count is the second. FastTracker 2 gives
an instrument sixteen samples of its own, where Impulse Tracker lets one reach the whole sample table, so
a band storing more zones than sixteen is **cut into slots**: its samples in key order, in runs of at most
sixteen, each run written as an instrument owning a contiguous stretch of the keyboard. Everything the
allocation paid for stays stored, and the pattern names the slot each note's dynamic and pitch resolve to.
A format numbering samples freely writes one instrument per band, which is the shape an IT run has always
had.

The instrument records are reserved before the solve, at the most instruments those keys could fill — a
sample per key, cut into runs — because the zones a band comes out as are known only afterwards. A plan
that groups keys into zones then comes out as fewer instruments, and the report says what the reserve left
free:

```
Instruments:     4 reserved  ->  2 written  (0.5 KiB of the reserve left free)
```

A written instrument names the axes the plan split, inside the 22 bytes FastTracker 2 keeps for a name:
`Piano` while one instrument holds everything, `Piano v000-v051` once dynamics split it, and
`Piano v000-v051 F1-B4` once keys do as well. Each slot answers the whole keyboard from the samples it
owns, so every one of them is playable on its own.

## Instrument files: one voice per file

`module.it` is a piece of music — the plan playing its own material, which is how a run is auditioned. An
instrument is what a producer of sampled instruments ships, so every instrument the module numbers is also
written on its own under `instruments/`:

```
artifacts/Piano/grouped/
├── module.it
├── bank.json
├── velocity_map.json
└── instruments/
    ├── p029-p063_v000-v051.iti
    └── p029-p063_v052-v127.iti
```

Each file carries the instrument header, the samples its keymap reaches and their waveforms, with those
samples renumbered into a table of its own — which is what lets a tracker or a player load one voice and
leave its own song alone. The name states both axes the plan split, the run of keys and then the velocity
band, so the directory sorts by keyboard position and reads as the map of which file plays what. Impulse Tracker writes `.iti` and FastTracker 2
`.xi`, at the same compliance level the module was graded against; the format is the run's, so
`--format xm` writes `.xi` beside `module.xm`.

## The bank manifest: the whole plan as one voice

A file per band ships the voices; `bank.json` says which one a note reaches. It states every band the
allocation chose, in order, each naming its instrument file and the velocity map the plan measured:

```json
{
  "version": 1,
  "name": "Piano",
  "layers": [
    {
      "source": { "file": "instruments/p029-p063_v000-v051.iti" },
      "select": { "velocity": { "low": 0, "high": 51 } },
      "velocity_map": "velocity_map.json"
    },
    {
      "source": { "file": "instruments/p029-p063_v052-v127.iti" },
      "select": { "velocity": { "low": 52, "high": 127 } },
      "velocity_map": "velocity_map.json"
    }
  ]
}
```

Every path is read against the directory the manifest sits in, so the strategy directory is the unit that
travels: copy it anywhere and it still plays. Each band states its own velocities rather than the last one
catching whatever fell through, so the manifest says out loud what the allocation decided. This is what
[MIDIToTracker](https://github.com/JakimPL/MIDIToTracker) reads:

```bash
midi2tracker song.mid --bank artifacts/Piano/grouped/bank.json --out song.it
```

A note picks its layer by the dynamic it was struck at, and that instrument's own keymap then picks the
sample — the same two steps `module.it` takes internally.

Where a format has room for a band only in several instruments, the keys tell those apart as well as the
dynamics, and the manifest states that too. Each of those files answers every key it was filled over, so
the keymaps alone can no longer say which one owns a key and a `pitch` band says it instead:

```json
"layers": [
  {
    "source": { "file": "instruments/p029-p060_v000-v127.xi" },
    "select": { "velocity": { "low": 0, "high": 127 }, "pitch": { "low": 0, "high": 60 } },
    "velocity_map": "velocity_map.json"
  },
  {
    "source": { "file": "instruments/p061-p101_v000-v127.xi" },
    "select": { "velocity": { "low": 0, "high": 127 }, "pitch": { "low": 61, "high": 127 } },
    "velocity_map": "velocity_map.json"
  }
]
```

The key bands tile the whole keyboard rather than stopping at the runs each file stores, so a note below
or above the recorded stretch reaches the instrument whose keymap was filled towards it — the same note a
single-instrument bank sounds. A band written as one instrument therefore owns the whole keyboard, which
is why it states its velocities alone: every plan is reachable through one manifest, whichever axes its
instruments were split along.

## The sample cap: asking for fewer, wider zones

A budget in bytes and a count of samples are different asks. `max_samples` is the second one:

```yaml
# src/opticonfig/optimize.yaml
max_samples: 0 # stored samples a grouped plan may keep, met by storing wider zones; 0 keeps what the format numbers
```

Pitch-zone grouping meets it by **charging every stored sample** a reserve on top of the bytes it
occupies. A charge makes a partition of many narrow zones expensive and one of few wide zones cheap, so
the same exact DP that spends the byte budget also answers the count. The run walks once for free: a plan
already inside its cap is the plan the budget alone chose, at no extra cost. Where the cap bites, the
search starts from the charge that prices every wider partition out of the budget and narrows back toward
the smallest charge that still holds, since a smaller charge leaves more of the budget for the samples
themselves. The report states what it settled on:

```
Samples:         8 stored of 8 allowed  (2371 B charged per sample, objective 24.8130 against 20.9520 uncapped)
```

This is a Lagrangian relaxation of a count constraint, and it behaves like one: it answers with the best
plan a charge induces, which the byte-vs-distortion frontier may leave a hair off the best plan of exactly
N samples. The exact alternative — a zone-count axis in the DP — multiplies a table already ~1.5 GB at
512 KiB by the cap. `--max-samples` overrides the cap for one run, and the cap a run works to is always
held inside what the target format numbers. The ungrouped strategy keeps a recording per key it plays, so
the cap is the grouped strategy's to meet.

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
override the two reduction knobs worth varying per run (see below), `--max-layers` the velocity bands a
key may store and `--max-samples` how many samples a grouped plan may keep.

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
