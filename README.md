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
under the identity `reduce/dedupe.yaml` names (`pitch`, `pitch_velocity`, or `pitch_velocity_cc`).

### Input: a directory of recordings

Every command also takes a directory of WAVs in place of a manifest, so a set of samples with no
performance behind it runs as it stands:

```bash
optisample optimize path/to/Samples/Piano --budget-kb 512 --out artifacts
```

The filenames carry what the manifest would have said. A `p<number>` token states the key and a
`v<number>` token the dynamic, which is how every dataset this project writes names them
(`0000_p029_v018_cc0-0_cc1-0.wav`, `0000_p029_F1_v018.wav`, `0000_p60_v100.wav`); `cc<number>-<average>`
tokens carry controller averages. A note name is read where no `p<number>` token appears, so a sample
library named `C4.wav`, `Bb2.wav`, `Grand_F#3_take.wav` resolves too, in either capitalisation and in
either spelling of a black key.

A directory states which keys were recorded and how hard, and says nothing of a song, so **each take is
read as one note held for as long as its recording sounds**. Usage weight then follows take length and
the recorded grid itself is what the allocation is optimized over. Where the filenames name no velocity
at all, every take shares one dynamic and the whole keyboard routes to it — one band is the most such a
set says about itself.

### Taking a slice first

A thousand-note instrument is worth iterating on at a tenth of its size before it is run whole, and
`optisample subset` writes that tenth as a dataset of the same shape:

```bash
optisample subset path/to/Piano.notes.json --fraction 0.1 --out subset
optisample reduce subset/Piano.notes.json --budget-kb 512      # picks up from there
```

A slice is written in the shape it was taken from: slicing a manifest dataset gives a manifest dataset,
and slicing a directory of recordings gives a directory of recordings, so the stage after it reads the
slice the way it would have read the whole.

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

A whole pre-optimization stage (`src/opticonfig/reduce/`) runs before any byte is allocated, and
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
  at, and settles the format the sample is stored at: the lowest rung of `sweep.rates` reaching that band,
  at the configured depth. The stored format follows from the recording alone, so the allocation spends its
  bytes on zone width, sample count and whether a sample stores its loop or the span it plays, and every
  sample it does keep carries the band its recording asked for.
- **Pitch-zone grouping** bounds its own search with `max_zone_semitones` and reuses a scored
  `(representative, encoding, key)` reconstruction across every zone containing it.

Each strategy's `report.txt` opens with a `Reduction (pre-optimization)` block reading `before -> after`
per axis, and `reduction.json` records every kept recording (with the material it covers) and the format
every pitch is stored at, beside the band it was settled from. A recording too short for its notes is
named there rather than quietly truncated.

Reduction is also where a pack's quality is decided, since it settles how long each stored sample may be
and which encodings the allocation ever gets to choose from. [`docs/tuning.md`](docs/tuning.md) is the
guide to that trade: which number picks the stored rate, what 8-bit costs, and which knob to reach for
when a run comes back lo-fi.

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
    auditions/p060_C4/             # reference.wav beside every encoding the sweep runs, rendered
```

The root is itself a NoteExtractor dataset, so the reduced tree is the resume point for a long run: the
survivors are trimmed to what the material asks of them (the demo's 40 recordings become 5 files, 2.6 MB
down to 340 KB) and the allocation reaches the sweep having paid only the ingest. Allocating from a
reduced dataset reproduces the plan allocating from its source produces, so the round trip costs nothing
in fidelity. The auditions make the stored format audible before the sweep is paid for.

A dataset reproduces its survivors exactly under the key it was reduced with; reducing it again under a
coarser key projects several identities onto one survivor, which then reports the identity of the first
note that reaches it.

### Running the stages in a row

`optisample pipeline` is the chain `subset`, `loop`, `reduce` and `optimize` make, under one output root:

```bash
optisample pipeline path/to/Piano.notes.json \
    --samples-dir path/to/Samples/Piano \
    --fraction 0.3 --budget-kb 512 --strategy grouped --max-layers 3 --no-render \
    --out artifacts
```

```
artifacts/
  0_subset/      # the slice taken of the source, as a dataset of the same shape
  1_looped/      # the loop each recording is stored around, plus its loops.json and auditions
  2_reduced/     # what the pre-optimization stage reduced that to, plus its reduction.json and auditions
  3_optimized/   # the artifacts allocated from it, one directory per instrument and strategy
```

Each stage reads back the dataset the one before it wrote, so the allocation reaches the sweep having
paid the ingest over the surviving recordings alone. Leaving `--fraction` out settles loops over the source
itself and the run begins at `1_looped`, which is how a dataset already sliced — or small enough to run
whole — goes through the same command. Every stage directory is what that stage writes when it is reached
on its own, so a chained run and the four commands typed out reach the same artifacts.

It takes every ingest, looping, reduction and allocation flag those commands take. `--max-layers` and
`--max-samples` reach the allocating stage, while the reduction between them runs at the configured
split and cap — which is what keeps its dataset the one any allocation off it reads back.

## The objective: what a note's distortion is worth

Every stage above competes for the same number, so it is worth stating exactly what that number counts.
A reconstruction is scored against the recording it came from with both sides loudness-matched, which
makes the fidelity a **relative** reading: how wrong the note sounds measured against itself. Left there,
a pianissimo note's error would count for as much as a fortissimo note's, though one of them is 30 dB
further down in the mix.

So each class's distortion is scaled by the energy of the stretch of recording it is scored over, raised
to `energy_exponent` (`src/opticonfig/optimize/budget.yaml`), and by the playing time the material spends on it:

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
survives. Every sample is therefore stored hot — normalized to `headroom_db` (`src/opticonfig/codec/quantize.yaml`)
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

Ahead of the quantizer sits an optional soft-knee compressor (`src/opticonfig/codec/dynamics.yaml`), reading its
threshold against each clip's own peak so it shapes crest factor rather than level. It is a **swept axis**,
not a preprocessing step: the grid offers each 8-bit encoding both compressed and plain, prices them the
same way, and the objective picks. A 16-bit encoding is enumerated uncompressed, since its noise floor
already sits below anything compression could protect. The `comp` column in `report.txt` and the `_c` in an
audition filename say which way each sample went.

## Looping: a stage that settles what a sample repeats

A looped sample is stored as the attack plus one loop region, and playback wraps that region for as long
as the note is held — which is what lets a note held for a minute cost a second of PCM.

Where a note turns periodic, how long a loop holds its timbre, and whether the wrap is clean are properties
of the recording rather than of the budget, so **a stage of its own settles them** (`src/opticonfig/loop/`),
ahead of the reduction and at the rate the analysis runs at. Each recording is offered the loops its
geometry allows — `placements` starts through the sustain, each at the lengths `length_multiples` asks for
and each clearing the `min_loop_s` floor and the window a spectrum is read over — ordered cheapest first,
meaning earliest and shortest. Every one clearing all three quality gates is **offered**, and every
candidate is long enough for all three to have measured it:

```yaml
max_seam_step: 4.0              # wrap step, in units of the frame-to-frame motion the waveform makes there
max_level_drift_db: 12.0        # fall across the region that holding it at one level has to flatten
max_spectral_distance_db: 12.0  # log-spectral distance between the loop and the stretch it stands in for
```

The gates say which loops a recording supports at all; **a budget says which of them is worth its bytes.**
Tighten the gates and the cheap loops drop out of the offer, and a recording with nothing clearing them is
stored over the span its material plays instead. Every candidate turned down is kept beside the ones
offered, naming the gate it fell outside, so `loops.json` states the whole case. On 60 real Piano
recordings the ladder offers a median of **4** loops that all clear the gates, spanning **4.66×** in stored
bytes between the cheapest and the dearest — a rate axis the allocation now gets to price rather than a
single choice made before it. Settling once, at the analysis rate, is also what lets the seam be measured
on the resolution a 440 Hz note actually has — 109 frames per cycle at 48 kHz against 18 at 8 kHz.

### Making the wrap inaudible

Two things are done to the region before it is stored, both at the resolution the split bought.

**The end is matched to the start.** The period is searched around the pitch the key already names, within a
semitone either side of it — a band that holds the tuning a set was recorded at while excluding two rounds of
the note, which a sweep of the whole pitched range reads instead on **19 of 120** real Piano recordings
wherever the even harmonics run strong. A period read off a finite window then lands between frames, and a
loop spans a hundred of them, so a whole count of rounded periods drifts part-way through a cycle by the time
it wraps. The peak of the autocorrelation is read between frames by the parabola through its top three, and
the end is then chosen within half a period of the whole count by maximising how alike the approach to the
end and the approach to the start measure — so the wrap lands on the phase the material left.

**The region is held at one level, and the seam is blended.** A struck note's region falls across itself, so
a player wrapping it steps the level back up once per round — a 0.5 s loop pulses at 2 Hz. Dividing the
region by the level its own material holds, pinned at the loop start, holds it at one amplitude; the decline
it gave up is handed to the fitted decay below. That level is read frame by frame as a local mean square
under a weighting spanning two periods of the note's own pitch, held inside the band `loop/envelope.yaml`
bounds, so it follows a region of any length and states the level the material actually holds where the loop
begins. The seam is then blended over a share of the loop
(`fade_share: 0.125`, floored in seconds and bounded by the material ahead of the start) with each side
weighted by how alike the two measure: material that repeats exactly is left untouched, material whose
partials have drifted apart keeps its level across the blend. Across 54 real Piano loops the level step at
the wrap falls from a median **|2.65| dB to |0.21| dB**, and reading each note over its own period is worth
**2.6×** of that against one weighting for every note alike.

The allocation then prices the trimmed span **plus one encoding per loop offered**, so how much of a note
is stored is chosen against the budget the same way its rate is, while the loop search itself is paid once
per recording. `EncodingParams.loop_index` names which offer an encoding holds, and the rate-distortion
hull drops the lengths that buy nothing.

### Settling loops on their own

`optisample loop` runs that stage and stops, writing the dataset the reduction picks up from:

```bash
optisample loop path/to/Piano.notes.json --budget-kb 96 --out looped
optisample reduce looped/Piano.notes.json --budget-kb 96      # picks up from there
```

```
looped/
  Piano.notes.json                 # one entry per played note, pointing at the recording serving it
  Piano/0000_p060_C4_v100.wav      # one WAV per recording, exactly as the stage analysed it
  loops/Piano/
    loops.json                     # the loops each recording offers, and the candidates turned down
    auditions/p060_C4/             # recording.wav beside looped0.wav ... one per offer
```

The WAVs are the ingest's own output — onset-aligned, at the run's one rate — so the frames a loop names
index straight into them, and every stage after this one only shortens them. The auditions are what makes
the stage judgeable by ear on its own: each one wraps the loop several times and puts the fitted decline
over it, which turns a seam step or a level pulse into a rhythm a listener hears.

### A held note that declines

Wrapping a region holds one level forever, which suits an organ and lies about a piano. So the stage fits a
**linear decay** beside the loop it settles (`src/optisample/dsp/decay.py`): the level the region is held
at, the levels the recording falls to from there read in short windows, and a least-squares line through
them saying where the material ends up. It is fitted over the whole recording rather than the stretch the
candidates were searched over, so every level the material states reaches the ramp. `render` plays a held
note down it, so a struck note stored as attack plus loop declines the way its recording did. Material that
holds its level to the end states no decline and carries no ramp.

The ramp begins at the loop start, which is where the stored material stops following the recording's own
envelope: the attack sounds as it was recorded, and from there the ramp restores exactly the decline
levelling erased. Its seconds run on the played timeline, which is the clock a tracker's volume envelope
runs on, so every key sounding the sample declines over the same stretch of time.

**The written module carries that decline as a volume envelope.** `optimize/export/build.py` gives every
instrument the curve its own samples state ([`export/envelope.py`](src/optisample/optimize/export/envelope.py)):
full volume through the attack, still full where the loop begins, the level the recording fell to by the
end of its ramp, and silence a release later. The third breakpoint is the sustain point, so a held note
stays where the recording left it and a released one goes on to the fourth and dies.

A format gives the envelope to the *instrument* rather than the sample, so every sample a written slot
holds shares one curve — the median decline among them. What that costs the sample it suits worst is
reported per instrument as `envelope_drift_db` in `plan.json`, which is the reading that says whether the
keys sharing an envelope are better written as several instruments.

Ticks are what a format counts envelope time in, so a curve is only right at the tempo it was fitted at. A
written module carries its own clock, so it is safe on its own; a `.bank` entry is an instrument lifted out
of that module and carries none, which is why the manifest states the `tempo` its envelopes were fitted
against.

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
`src/opticonfig/optimize/layers.yaml` sets the terms:

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

An instrument is what a producer of sampled instruments ships, so every instrument the plan is written as
lands as a file of its own — inside the bank the run writes, and spread out beside it under `instruments/`:

```
artifacts/Piano/grouped/
├── Piano.bank                        # the manifest and both instruments, as one file
├── instruments/
│   ├── p029-p063_v000-v051.iti
│   └── p029-p063_v052-v127.iti
└── render/module.wav                 # the plan playing its own material, rendered
```

Each file carries the instrument header, the samples its keymap reaches and their waveforms, with those
samples renumbered into a table of its own — which is what lets a tracker or a player load one voice and
leave its own song alone. The name states both axes the plan split, the run of keys and then the velocity
band, so the directory sorts by keyboard position and reads as the map of which file plays what. Impulse
Tracker writes `.iti` and FastTracker 2 `.xi`, at the same compliance level the module was graded against;
the format is the run's, so `--format xm` fills the bank with `.xi`.

The piece a run is auditioned as is a module of the plan playing the instrument's own material. It is
built in memory, rendered to `render/module.wav` and scored note by note under `compare/`, so what a run
leaves on disk to listen to is audio and what it leaves to play is the bank.

## The bank: the whole plan as one file

A file per band ships the voices; the bank says which one a note reaches, and carries them all with it.
`<name>.bank` is a zip holding the manifest and every instrument that manifest names:

```
Piano.bank
├── bank.json
└── instruments/
    ├── p029-p063_v000-v051.iti
    └── p029-p063_v052-v127.iti
```

The manifest states every band the allocation chose, in order, each naming the entry it is stored as and
carrying the velocity map the plan measured:

```json
{
  "version": 2,
  "name": "Piano",
  "layers": [
    {
      "source": { "file": "instruments/p029-p063_v000-v051.iti" },
      "select": { "velocity": { "low": 0, "high": 51 } },
      "velocity_map": { "reference_volume": 57, "anchors": [ ... ], "volumes": [ ... ] }
    },
    {
      "source": { "file": "instruments/p029-p063_v052-v127.iti" },
      "select": { "velocity": { "low": 52, "high": 127 } },
      "velocity_map": { "reference_volume": 57, "anchors": [ ... ], "volumes": [ ... ] }
    }
  ]
}
```

The map is written out in full where it is abbreviated above: `volumes` is the whole 0..127 lookup table a
note's velocity indexes, and `anchors` one record per measured dynamic (`velocity`, `loudness_lufs`,
`volume`) stating the loudness the fit was made through. It travels inside the bank because each stored
sample's gain is derived from it (see [Gain staging](#gain-staging-storing-hot-and-getting-the-level-back)),
which makes the waveforms and the map one calibrated unit — so a bank travelling as a single file keeps a
layer playing the dynamics its own recordings were written for however it is copied, renamed or handed on.
Each band states its own velocities rather than the last one catching whatever fell through, so the
manifest says out loud what the allocation decided. This is what
[MIDIToTracker](https://github.com/JakimPL/MIDIToTracker) reads:

```bash
midi2tracker song.mid song.it --bank artifacts/Piano/grouped/Piano.bank
```

A note picks its layer by the dynamic it was struck at, and that instrument's own keymap then picks the
sample — the same two steps the audition module takes internally.

Where a format has room for a band only in several instruments, the keys tell those apart as well as the
dynamics, and the manifest states that too. Each of those files answers every key it was filled over, so
the keymaps alone can no longer say which one owns a key and a `pitch` band says it instead:

```json
"layers": [
  {
    "source": { "file": "instruments/p029-p060_v000-v127.xi" },
    "select": { "velocity": { "low": 0, "high": 127 }, "pitch": { "low": 0, "high": 60 } },
    "velocity_map": { "reference_volume": 57, "anchors": [ ... ], "volumes": [ ... ] }
  },
  {
    "source": { "file": "instruments/p061-p101_v000-v127.xi" },
    "select": { "velocity": { "low": 0, "high": 127 }, "pitch": { "low": 61, "high": 127 } },
    "velocity_map": { "reference_volume": 57, "anchors": [ ... ], "volumes": [ ... ] }
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
# src/opticonfig/optimize/budget.yaml
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

Optimize a `.notes.json` — or a directory of recordings — into an inspectable artifact tree:

```bash
optisample optimize path/to/Piano.notes.json \
    --samples-dir path/to/Samples/Piano \
    --budget-kb 96 \
    --out artifacts
```

`--samples-dir` names where a manifest's recordings live, defaulting to the notes file's sibling
`<name>/` directory; a directory source holds its own. `--instrument-id` defaults to the `.notes.json`
base name, or to the directory's own name. `--pre-roll-ms` / `--post-roll-ms` state the padding a
**directory** of recordings holds around each note; a `.notes.json` records its own rolls in
`settings.rolls` and is read by those. Either way the pre-roll comes off the front so frame 0 lands on
the note onset, and the padding past the release comes off the end so a stored sample ends where the
note does — `--keep-tail` stores it through that padding instead. Other flags: `--format {it,xm}`, `--strategy {both,grouped,ungrouped}`, `--no-render`, `--rate`
(repeatable, the ladder a stored rate is chosen from), `--depth`, `--no-loop`, `--interpolation`,
`--seed`. `--dedupe-key` and `--content-floor-db` override the two reduction knobs worth varying per run
(see below), `--max-layers` the velocity bands a key may store and `--max-samples` how many samples a
grouped plan may keep.

Each long stage draws a labelled progress bar on stderr, carrying the count it will reach and an ETA, so
a large instrument states how long it needs while it runs:

```
Probing recordings................ 100%     40/40     [00:00<00:00]
Narrowing stored grids............ 100%      5/5      [00:00<00:00]
Sweeping encodings................ 100%     13/13     [00:00<00:00]
Scoring pitch zones............... 100%     12/12     [00:07<00:00]
```

`--seed` fixes the dither so a run reproduces byte for byte; every other flag above is shared with
`optisample loop` and `optisample reduce` (see above), and `optisample pipeline` takes all of them at once
to run the slice, the looping, the reduction and the allocation in a row under one output root.

Bars are drawn when stderr is a terminal, so redirected output stays clean; `--no-progress` silences them
at a terminal too. Results keep to stdout either way.

The stages that fan out — narrowing each pitch's stored grid, rendering each demo note — share their work
across processes. `--workers N` sets how many; `0` (the shipped `src/opticonfig/runtime.yaml` value) asks
for one per core, and each stage caps that at the items it holds, so a five-pitch stage starts five
workers. `--workers 1` carries the whole run in the calling process, which is what `--profile` reads end
to end and what a machine you are also working on stays responsive under. A run states the same reduction
and the same plan whichever count it was given: every pitch is narrowed from its own clip against a fixed
dither seed, and each demo note's phases are drawn in note order before any of them are rendered.

`--format` overrides `src/opticonfig/export/tracker.yaml`, which also sets the compliance level the module is
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
axes, listen to every swept encoding, and compare each pitch's reconstruction against the
recording the objective scored it on. It shells out to `optisample` and prints the invocation behind
every button, so what it finds is reproducible from a terminal, and its explorers read whatever is
already on disk — a long run started in a shell can be inspected without re-running it.

```bash
uv run marimo edit notebooks/pipeline.py
```

See [`notebooks/README.md`](notebooks/README.md) for what each panel shows.
