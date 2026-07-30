# Tuning: buying quality back

The optimizer answers one question — *given this many bytes, which encodings lose the least?* — and it
answers it well. What it does **not** decide is how many samples the bytes are spread over, how long each
one is allowed to be, or how deep the quantizer may go. Those are the inputs, and they are what a pack
that comes back lo-fi is really complaining about.

This document is the map: what each stage decides, which number it reads, and which knob moves it. The
measurements come from one real run — 1038 piano takes sliced to 30 %, reduced to 307 survivors, allocated
at `--budget-kb 512 --strategy grouped --max-layers 3`; the predictive tables are arithmetic on top of it.

## 1. Read the run you already have

Three files say everything. Start with `report.txt`:

```
Budget:      512.0 KiB module  ->    510.7 KiB samples
Used:        510.4 KiB samples  ( 99.9% of budget, 0.3 KiB free)
Keyboard:      120 of 120 keys answered  (61 played, 59 filled from the nearest recording)
Samples:        24 stored of 24 allowed  (each priced at the bytes it stores)
Stored grid: 14.2-32.0 kHz useful  ->  16.0-37.8 kHz stored, 4.0 swept per key
```

Then the zone table under it, which is the whole plan in one page:

```
layer         keys   rep  rep.vel  rate(Hz)  depth  comp  size(KiB)  distortion  options
    0   41-53 (12)    45       47      6000     16     -      117.3      1.4218        7
    0   68-80 (13)    76       40      6000     16     -      112.2      1.9082        5
    ...
    0      101 (1)   101       41      6000      8     -        1.6      0.0028        3
```

Two of the twenty-four samples hold **45 % of the budget**. That is the finding, and everything below is
why it happens and which knob undoes it.

For anything the table leaves out, `plan.json` carries the same zones with `frames`, `trim_s`, `loop`,
`decay` and `stored_bytes`, and `reduction.json` carries every survivor's length beside the length its
notes asked for, plus the format every key is stored at and the band it was settled from.

## 2. Who decides what: the format is settled before the budget is spent

The stored format — rate, depth, compression — is settled by the **reduction**, from the recording's own
content ([`stored_format`](../src/optisample/optimize/reduce/bandwidth.py)). The band a clip occupies is
measured, the rate that carries it at the transpose it plays at is read off that, and the sample is stored
at the ladder's **lowest rung reaching** it. The budget is not consulted.

What the allocation spends bytes on is therefore everything else:

| Settled before the budget | The allocation trades |
|---|---|
| stored rate, depth, compression (the reduction) | zone width, sample count |
| how long a recording may run, §5 (the reduction) | the settled loop, or the trimmed span |
| which loop a recording repeats, §8a (the loop stage) | how many velocity bands a key stores |

The consequence is worth stating plainly. **A budget too small for the formats its recordings ask for is
infeasible, and the run says so** rather than quietly storing everything duller:

```
ungrouped: infeasible (budget 97553 B too small; cheapest allocation needs 106540 B)
```

Every allocation stage raises that
([`BudgetInfeasibleError`](../src/optisample/optimize/dp.py)) and the artifact dump records it as the
reason a strategy wrote no plan, so the other strategy's plan still lands beside it. The two answers are
to raise `--budget-kb` or to lower `--content-floor-db` (§3), which is the knob that decides how much band
every stored sample carries.

The mirror case is a budget with room the allocation cannot spend: with the format settled and the lengths
set by the material, an ungrouped plan has few upgrades left to buy. The shipped demo at 96 KiB stores
70 644 B and stops — the remaining 26 KiB buys nothing, because no key has anything costlier worth storing.

## 3. The stored rate: why 6 kHz, and why pitch has little to do with it

The rate ladder is the configured list, read as far as a clip reaches
([`sweep_rates`](../src/optisample/optimize/operating_points.py)): every listed rate below the clip's own,
plus the clip's own rate so storing it as recorded is always a candidate. From `optimize/sweep.yaml`'s
`[8000, 11025, 16000, 22050, 28800, 37800, 44100]`, a 48 kHz recording is offered

```
48000  44100  37800  28800  22050  16000  11025  8000
```

and a 22.05 kHz one is offered `22050 16000 11025 8000` — a listed rate above the recording would
resample it upward, spending bytes on a band it never held.

Two things then narrow it.

**The bandwidth bound** is the pitch-aware half, and it is the one intuition expects. Each recording's own
spectral edge is measured (`content_floor_db` below its loudest band), the playback ceiling is scaled down
by the widest upward transpose the sample plays at, and Nyquist doubles the survivor into a rate
([`_audible_rate_hz`](../src/optisample/optimize/reduce/bandwidth.py)).

`content_floor_db` is now the number that **decides** the rate, so it is the single most consequential
knob in the reduction. Read on real piano takes at 48 kHz, the rung it lands each key on:

| `content_floor_db` | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|---|
| `45` | 8.0 | 8.0 | 11.0 | 8.0 | 16.0 | 16.0 |
| `60` (shipped) | 8.0 | 8.0 | 16.0 | 11.0 | 16.0 | 37.8 |
| `70` | 11.0 | 11.0 | 37.8 | 37.8 | 37.8 | 37.8 |
| `80` | 37.8 | 37.8 | 37.8 | 37.8 | 37.8 | 37.8 |

The jump between 60 and 70 is the tell: ten decibels moving a key from 16 kHz to 37.8 kHz means the
spectrum is nearly flat across that stretch, which is a **noise floor**, not content. Eighty decibels reads
the room, and lands every key on the ceiling (`ceiling_hz: 16000`, doubled by Nyquist, rounded up to the
32 kHz rung). Sixty reads the content above that noise, which is why it is shipped.

Measured against the shipped demo at 96 KiB, the floor is also what decides whether a plan exists at all:

| `content_floor_db` | demo piano, ungrouped | demo piano, grouped |
|---|---|---|
| `45` | 1.4375 / 63 640 B | 1.1650 / 46 652 B |
| `60` (shipped) | **0.0979 / 70 644 B** | **0.0985 / 70 644 B** |
| `70` | 36.2265 / 91 066 B | 0.9603 / 83 182 B |
| `80` | infeasible | 0.9603 / 92 862 B |

Two readings worth carrying away. A floor too **low** stores less band than the budget could afford (45
scores fifteen times worse than 60 while spending fewer bytes). A floor too **high** forces the allocation
to buy its bytes back the only way it still can — by storing loops of notes too short to loop well, which
is what the 36.2 reading is, and past that by not fitting at all.

**The ladder's floor is a real bound.** Below its cheapest rung a clip has nowhere to go, however little
band it occupies: at `--content-floor-db 45` the demo's C3 asks for 1.4 kHz while the cheapest rung is
8000. Rounding **up** to the nearest rung is deliberate — it never drops content the material still plays —
so a ladder wants rungs where its material lands. `32000` is on the shipped ladder for exactly that
reason: it is where a 16 kHz ceiling lands, and storing there rather than at 37800 saves 18 % of every
such sample.

**Levers.** `--content-floor-db` per run; `optimize/sweep.yaml: rates` is the ladder itself and `--rate`
(repeatable) replaces it for one run.

## 4. Bit depth: what 8 bits costs, and how to refuse it

A signed 8-bit grid has a step of `2^-7`. Samples are normalized to `headroom_db: 0.5` under full scale,
so one LSB sits at

```
20 * log10( (1/128) / 10^(-0.5/20) )  =  -41.6 dB   below the stored peak
```

That is a hard floor of dither noise, constant for the whole length of the sample. A piano decays through
it in one to three seconds — after which the sample is *only* noise, and it stays that way until the
sample ends. That is the hiss you can hear behind the decay. At 16 bits the same floor sits at −89.8 dB
and never surfaces.

Compression goes with the depth: `compress` applies only at depths of 8 or fewer
([`compresses`](../src/optisample/optimize/operating_points.py)) precisely because it trades crest factor
for headroom against a shallow grid — which also lifts the decayed tail toward the noise floor.

**Depth is a decision, not a search.** Every sample in a run is stored at one depth, and it is settable
from both sides:

```bash
optisample optimize Piano.notes.json --budget-kb 512 --depth 16     # one run at 16 bits
```

```yaml
# src/opticonfig/optimize/sweep.yaml
depth: 16 # bits every stored sample keeps
```

Expect fewer samples, or shorter ones, for the same budget — a 16-bit sample buys half the frames an
8-bit one does, and the allocation can no longer answer a tight budget by going shallow.

## 5. Stored length: the biggest lever by far

Length enters twice, and both places are worth knowing.

**At reduce time**, each key's survivor must be long enough to serve its longest note, transposed
([`required_duration_s`](../src/optisample/optimize/reduce/dedupe.py)):

```
required = max( min( longest_note_s * 2^(transposition_headroom_semitones/12), max_length_s ),
                attack_skip_s + min_loop_s + tail_skip_s )
```

With the shipped `transposition_headroom_semitones: 12`, the longest note at a key **doubles**, and
`max_length_s` then caps the result. In the measured run **305 of 307 survivors were shorter than what was
asked of them**, and eight were written at the full 10 s cap.

**At encode time**, the stored span is cut to what the zone needs
([`_zone_trim`](../src/optisample/optimize/grouping/cost_model.py)): the longest note of any key the zone
covers, scaled by how much faster that key plays the sample. A zone whose keys hold long notes stores a
long sample; a zone whose keys hold blips stores 0.13 s.

**A handful of held notes can set the length for the whole plan.** The measured material has a median note
of **0.674 s** — and five notes, all at low velocity, held for **99 to 106 seconds** (pedalled, at pitches
62, 69, 71, 76 and 88). Those five are why three zones asked for 68, 86 and 108 seconds of stored audio,
were capped at ten, and then stored every second of it at the format their band asked for. Measured on the two
samples they produced, the peak envelope falls 30 dB below its own peak within 2.3–4.0 s and 50 dB below it
by 7.9–8.8 s: the budget bought roughly six seconds of near-inaudible decay at 16-bit, twice.

**Levers, strongest first:**

| Knob | File | What it does |
|---|---|---|
| `reduce.trim.max_length_s` | `reduce/trim.yaml` | Hard ceiling on every kept recording, now 5.0. Lowering it further caps the whales and frees their bytes for everyone else. |
| `reduce.dedupe.transposition_headroom_semitones` | `reduce/dedupe.yaml` | 12 doubles every requirement. Set it to the widest zone you actually allow (see `max_zone_semitones`); 7 costs 1.5x instead of 2x. |
| `reduce.trim.tail_floor` | `reduce/trim.yaml` | Where a decay stops counting as content — `1e-4` is −80 dB, which keeps far more tail than a tracker ever plays audibly. `1e-3` (−60 dB) is defensible. |

`max_length_s` is read at **every** ingest, including one reading an already-reduced dataset, so you can
re-allocate off `2_reduced/` with a lower cap and see the effect without re-reducing.

## 6. Sample count, zone width, layers

These three decide how the budget is *divided*, which is where the allocation does its spending (§2).

- **`--max-samples N`** (`optimize.yaml: max_samples`) charges a reserve per stored sample, so a
  many-narrow-zones partition is priced out and a few-wide-zones one wins. Fewer samples means *wider*
  zones and *bigger* byte targets each — it buys per-sample quality with repitching error.
- **`reduce.grouping.max_zone_semitones`** caps how far one recording may be stretched. The shipped 12 is
  an octave; a piano stretched an octave is audibly wrong long before the metric says so. Lower it to 5–7
  and the plan is forced into more, narrower zones.
- **`--max-layers`** (`layers.yaml: max_layers`) splits each key across velocity bands, one written
  instrument per band. Every extra layer multiplies the samples the same budget has to cover. The measured
  run took 2 of the 3 allowed.

The three interact: lowering `max_zone_semitones` while holding `--max-samples` fixed can make a plan
infeasible, and the run says so rather than guessing.

## 7. What the objective is actually scoring

Two settings decide what "loss" means, and both are worth a look when the plan spends bytes somewhere your
ear says it should not.

- **`metrics.preprocess.dynamic_range_db: 80.0`** floors the log-spectral comparison 80 dB under the
  reference's own peak. Every time-frequency bin above that floor counts equally, and a ten-second decay
  has ten times the bins of a one-second note — most of them in material 40–60 dB down. That is a large
  part of why storing the whole decay looks worth it to the solver. Lowering it to 45–55 dB makes the
  metric ignore what a tracker mix would bury anyway.
- **`optimize.energy_exponent: 0.5`** scales each note's distortion by its own energy to that power. `0.0`
  prices every note alike, `0.5` follows amplitude, `1.0` follows energy, `0.3` tracks perceived loudness.
  Raising it concentrates the budget on the loud notes.

Loudness normalization (`target_lufs: -23.0`) is applied to reference and candidate separately before
comparison, so a candidate that truncates the decay gets *boosted* before it is measured — which makes a
missing tail cost more than its loudness suggests.

## 8. Clicks and tails

Measured on the run's own stored samples, last 5 ms against the sample's peak:

```
  -2.8 dB  zone10  16-bit  loop  0.55 s
  -3.9 dB  zone00   8-bit  loop  0.55 s
  -4.2 dB  zone18   8-bit    -   0.13 s
  -8.4 dB  zone20   8-bit    -   0.51 s
 -12.1 dB  zone07  16-bit    -   0.21 s
```

`trim_recording` cuts where the decay falls under `tail_floor`; the encoder's `_loop_or_trim` cuts at
`trim_s` — the length the *material* asks for, which is routinely shorter than the recording's own decay.
The result was a step from −4 dB straight to zero, and a tracker plays that step as a click. The source
takes contribute too: these end at a median of −33 dB below their own peak, because the extractor cuts at
note-off rather than at silence.

**`quantize.yaml: release_fade_s` closes that step.** A stored span that plays to its end is ramped to
silence over its last stretch (10 ms as shipped), so the last frame is zero. A looped span keeps its wrap
point, already made continuous by `crossfade_loop`. On the demo, forcing trimmed samples with `--no-loop`,
the last frame moves from −8.3…−21.4 dB below peak to digital silence.

**The ground truth closes on the same ramp.** The ramp is policy, not a choice the allocation makes, so
charging it as distortion measured the policy instead of the codec: the demo's `--no-loop` objective read
0.0234 without the ramp and 0.1050 with it, and since only non-looped candidates paid it, the fade nudged
the plan toward looping — which is the tail in item 2 below. `Event.scored_reference` now puts the same
ramp over the recording, on the output frames the candidate's ramp actually covers: a stored sample records
how far its ramp reaches (`StoredSample.release_frames`), repitching carries that stretch onto the output
timeline with the material, and the ground truth is closed over the frames it reaches. A class whose note
ends before the ramp begins is measured against the recording as it stands, which is what its candidate is
— cut at the note's length, short of the ramp. A looped sample wraps at its seam and closes on nothing, so
its ground truth is the recording throughout, and the two cases are now scored alike.

The stored *length* stays charged. Past the end of the stored material the ground truth is the recording,
so a note held longer than its sample is still measured against every frame it is missing — only the ramp
itself stops being counted as error. That distinction is why the ramp is placed by frame rather than
applied to the last 10 ms of every scored window: the reach is `min(note end, stored end)`, and at a stored
rate below the analysis rate the repitched span lands a few frames short of the note, which a rule keyed on
the note's end alone would read as "no ramp here" for most of the ladder.

Three things read as "a tail after the piano decays":

1. **8-bit dither noise** (§4) running at −41.6 dB for the rest of the sample's length.
2. **A loop on decaying material.** The run above was taken under a sustain gate that let short recordings
   of low struck notes through — over 1.3–1.8 s a bass note has fallen only 5.7–5.8 dB — and stored them
   as the shortest loop allowed, ringing at that level until the next note cut them. Struck material is
   loopable by design now: the decline is carried beside the PCM as a
   [`LinearDecay`](../src/optisample/dsp/decay.py), fitted from the level the loop holds and the levels
   the recording falls to past it, and [`render`](../src/optisample/dsp/surrogate/render.py) plays a held
   note down it. So the objective scores a piano note as attack + loop + decline, which is what makes a
   long note cheap.

   **The written module does not carry that envelope yet.** `optimize/export/build.py` writes each
   `Instrument` with no `volume_envelope` and no `fadeout`, so an exported looped note holds the loop's
   level while its score says it declines: the module sounds worse than the number. While that gap is
   open, `--no-loop` is what closes it — and it costs bytes, since the trimmed sample stores every second
   it sounds for.
3. **A neighbouring note inside the take.** One 10.16 s source take decays to −52 dB and then re-attacks to
   −19 dB, 0.29 s before it ends — the following event in the performance, captured inside this note's take.
   the 10 s cap in force at the time cut right at that onset, so the stored sample ended *on* the
   transient. Where takes overlap like this, a length cap is also a content decision: the 5 s cap now
   shipped keeps the note and leaves the neighbour out.

**Levers.** Which loop a sample repeats is settled by the loop stage (§8a), so the grid offers the settled
loop beside the trimmed sample and a loop is bought only where the objective prefers it to storing the
recording as played; `--no-loop` leaves the stage off and pins the grid to the trimmed sample alone.
`loop.min_loop_s` is a floor every candidate clears, which is what keeps a loop from shrinking to the
stutter the run above stored. What a loop settles at is then brought down by the fitted decay, so the lever
for "this rings on" is the export gap above rather than a looping threshold.

### 8a. Which loop a sample repeats

The loop stage (`src/opticonfig/loop/`) settles that per recording, before any byte is allocated and at the
rate the analysis runs at. Three groups tune it:

- **`loop/geometry.yaml`** lays out what is on offer. `placements` spreads the starts through the sustain
  and `length_multiples` offers each start at several lengths, so a note that changes as it rings can be
  looped where it has settled. `min_loop_s` floors the length, `min_periods` keeps a loop from beating at
  its own rate, and `min_hz` / `max_hz` / `min_correlation` bound what counts as periodic at all.
- **`loop/quality.yaml`** holds the two gates, which are the aggressiveness dial. `max_seam_step: 4.0`
  bounds the step at the wrap, measured in units of the frame-to-frame motion the waveform makes there, so
  the reading means the same on a loud attack and a quiet decay. `max_spectral_distance_db: 12.0` bounds how
  far the loop's timbre sits from the stretch it stands in for. Candidates are climbed cheapest first —
  earliest and shortest — and the first clearing both is kept, so tightening a gate buys a longer, better
  loop and loosening one buys bytes.
- **`loop/seam.yaml`** sets the crossfade the wrap is blended over (`crossfade_s: 0.01`).

`loops.json` states the loop each recording keeps and every candidate climbed past with the gate it fell
outside, so a retune reads off the last run rather than guessing. `1_looped/loops/<id>/auditions/` holds
each loop played out against its recording, which is the by-ear reading of the same decision.

For *where* the cut lands, `max_length_s` is no help — it is a ceiling, and the clicking samples sit far
under it. The knob that reaches `trim_s` is `reduce.events.duration_bucket_ratio`, which rounds every
scored note length *up* to a geometric grid edge; `2.0` gives each sample up to twice the decay before the
cut, and costs up to twice the bytes for it. `quantize.release_fade_s` handles the step the cut leaves.

## 9. Recipes

**"The whole pack is lo-fi."** The stored band is what the reduction read, so start there: raise
`--content-floor-db` (§3) and see what the rungs become. If the budget then will not fit, buy the room the
usual way — cap the length further (`max_length_s: 3.0`), refuse 8-bit (`--depth 16`), then narrow the
zones (`max_zone_semitones: 7`). Re-allocate off `2_reduced/` — both are read at every ingest, so no
re-reduce is needed.

```bash
cp -r src/opticonfig myconfig           # --config takes a directory laid out like the bundled one
$EDITOR myconfig/reduce/trim.yaml       # max_length_s: 3.0
optisample optimize artifacts/2_reduced/Piano.notes.json \
    --budget-kb 512 --strategy grouped --max-layers 3 --depth 16 \
    --config myconfig --no-render --out artifacts/tuned
```

**"One sample eats the budget."** It is a wide zone storing a long recording (§5) at the format its band
asked for (§2). Cap `max_length_s`, or lower `max_zone_semitones` so no zone covers thirteen keys.

**"There is hiss behind the decay."** 8-bit noise floor. `--depth 16`.

**"Samples click."** `quantize.release_fade_s` closes each stored span on silence (§8); raise it if 10 ms
is too abrupt for your material. `duration_bucket_ratio: 2.0` buys each sample more decay before the cut
in the first place, and `--no-loop` removes the looped variant of the problem.

**"It sounds fine but there are only 24 samples."** Raise `--max-samples`, or set it to `0` to keep as many
as the format numbers. Each extra sample is charged a reserve, so the run states what it settled on.

## 10. Knob index

| Knob | Where | Reach for it when |
|---|---|---|
| `--budget-kb` | CLI | Everything is too thin and the target has room. |
| `reduce.trim.max_length_s` | `reduce/trim.yaml` | A few samples dominate the plan. |
| `--depth 16` | CLI (`optimize/sweep.yaml: depth`) | Hiss behind the decay; 8-bit is unacceptable. |
| `--rate` (repeatable) | CLI (`optimize/sweep.yaml: rates`) | Replacing the ladder for one run. |
| `sweep.rates` | `optimize/sweep.yaml` | Reshaping the ladder itself — its floor is what long samples land on. |
| `quantize.release_fade_s` | `codec/quantize.yaml` | Samples click at their end, or 10 ms of ramp is audible. |
| `--content-floor-db` | CLI (`reduce/bandwidth.yaml`) | The stored band is duller than the budget can afford, or a budget will not fit. |
| `--max-samples` | CLI (`optimize/budget.yaml`) | Trading sample count against per-sample quality. |
| `reduce.grouping.max_zone_semitones` | `reduce/grouping.yaml` | Repitching artefacts across a zone. |
| `--max-layers` | CLI (`optimize/layers.yaml`) | Dynamics matter more (or less) than fidelity per note. |
| `dedupe.transposition_headroom_semitones` | `reduce/dedupe.yaml` | Survivors are longer than the music needs. |
| `reduce.events.duration_bucket_ratio` | `reduce/events.yaml` | Samples are cut mid-decay and click. |
| `metrics.preprocess.dynamic_range_db` | `analysis/metrics.yaml` | The solver pays for material you cannot hear. |
| `optimize.budget.energy_exponent` | `optimize/budget.yaml` | Quiet notes are getting a budget share out of proportion. |
| `--no-loop` | CLI (leaves the loop stage off) | Looped decays ring on. |
| `loop.quality.max_seam_step` | `loop/quality.yaml` | A wrap ticks or clicks. |
| `loop.quality.max_spectral_distance_db` | `loop/quality.yaml` | A held note keeps a timbre the recording moves away from. |
| `loop.geometry.min_loop_s` | `loop/geometry.yaml` | Loops are short enough to buzz at their own rate. |
| `loop.geometry.placements`, `loop.geometry.length_multiples` | `loop/geometry.yaml` | The loop sits where the note has not settled yet. |
| `loop.seam.crossfade_s` | `loop/seam.yaml` | The wrap is continuous but audible as a texture change. |

`--config` takes a **directory** laid out the way the bundled one is -- a stage per directory, a group per
file -- so copy the whole `src/opticonfig/` tree and edit the copy.
