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
Stored grid:    18 encodings  ->    3.0 shortlisted per key  (14.2-32.0 kHz useful)
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

For anything the table leaves out, `plan.json` carries the same zones with `frames`, `trim_s`, `loop` and
`stored_bytes`, and `reduction.json` carries every survivor's length beside the length its notes asked
for, plus each key's shortlist.

## 2. The per-key share: the number the whole run turns on

Before a single encoding is scored, the run computes one figure
([`per_key_bytes`](../src/optisample/optimize/plans/budget.py)) — the sample budget split evenly over
every key that has to be answered:

```
one layer, 61 played keys    523537 // 61   =  8582 B per key
the 2-layer split it chose   522979 // 106  =  4933 B per key
a zone's target              per-key share  *  the keys the zone covers
```

That share is what the shortlist is ranked around. For each clip the bandwidth pre-pass prices the whole
`(rate, depth, compress)` grid, takes the rate-distortion hull, and keeps the `candidates` (3) vertices
whose **byte cost is nearest the share** ([`_shortlist`](../src/optisample/optimize/reduce/bandwidth.py)).
Encodings far from that price are dropped before the expensive sweep ever sees them. The per-pitch
shortlists in `reduction.json` are priced at the unlayered share; each candidate zone is priced at its own
split's share times its width.

The 112 KiB sample follows from that, with one twist. Its zone covers 13 keys, so it shortlisted around
`13 x 4933 = 64 129 B` — but its representative has to hold 9.57 s of audio, and at the rate ladder's
cheapest rung that already costs `9.57 s x 6000 Hz x 8 bit = 57 420 B`. **There is no cheap encoding of a
long clip.** All three shortlisted vertices were long and expensive, and the allocation bought the 16-bit
one (114 816 B of PCM plus 84 B of sample record) because thirteen keys' worth of weight justified
doubling the depth. Length set the floor price; weight paid for the rest.

**Lever.** Cap the length (§5) — that is what moves the floor. Raising `--budget-kb` or narrowing the
zones (§6) helps at the margin; neither makes a ten-second clip affordable.

## 3. The stored rate: why 6 kHz, and why pitch has little to do with it

The rate ladder is the configured list, read as far as a clip reaches
([`sweep_rates`](../src/optisample/optimize/operating_points.py)): every listed rate below the clip's own,
plus the clip's own rate so storing it as recorded is always a candidate. From `sweep.yaml`'s
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

At the shipped `content_floor_db: 80` it almost never binds. In the measured run `useful_rate_hz` came out
at **32.0 kHz for 57 of the 61 keys** — the `ceiling_hz: 16000` cap, doubled. Eighty decibels under the
loudest band is the recording's own noise floor, so the measure reads the noise rather than the content.

`--content-floor-db` sets it per run, and tightening it is what makes the bound track pitch. On the synth
demo the useful rate becomes monotone in pitch:

| | C3 | G3 | C4 | G4 | C5 |
|---|---|---|---|---|---|
| `80` (shipped) | 11.4 kHz | 21.8 kHz | 21.8 kHz | 21.8 kHz | 21.8 kHz |
| `45` | 1.4 kHz | 2.2 kHz | 5.4 kHz | 7.8 kHz | 10.6 kHz |

Forty-five decibels is aggressive enough to sound dull; the region worth auditioning is roughly 50–60.

**The budget share** is what actually picks the rate, and it is a function of *length*:

```
bytes = length_s * rate * depth / 8 + 84      ->      rate ≈ 8 * byte_target / (length_s * depth)
```

At the run's per-key share of 8582 B (the one the per-pitch shortlists were priced at):

| stored length | 8-bit | 16-bit | rung the shortlist lands on |
|---|---|---|---|
| 0.2 s | 43 kHz | 21 kHz | 44100 / 22050 |
| 0.5 s | 17 kHz | 8.6 kHz | 16000 / 8000 |
| 1.0 s | 8.6 kHz | 4.3 kHz | 8000 / 8000 |
| 3.0 s | 2.9 kHz | 1.4 kHz | 8000 (the ladder's floor) |
| 10.0 s | 0.9 kHz | 0.4 kHz | 8000 |

The run measured before the list landed followed that table — D#5, whose notes are short, shortlisted the
top three rungs; A2, whose survivor ran the full 10 s, shortlisted the floor alone. **Long samples get low
rates. Pitch enters only through the bandwidth bound, and only once that bound is tightened.**

(The `+ 84` is the per-sample record every stored sample costs on top of its PCM, which is why a plan of
many tiny zones pays a real fixed price for each of them.)

**The ladder's floor is a real bound.** Below it a clip has nowhere cheaper to go, however little band it
occupies. On the synth demo, `--content-floor-db 45` puts C3's useful rate at 1.4 kHz while the cheapest
rung is 8000 — five times more than the content asks for. A ladder aimed at a tight budget wants a cheap
end as much as a high one.

**Levers.** `sweep.yaml: rates` is the ladder itself; `--rate 22050 --rate 16000` replaces it for one run
(repeatable). Shortening the samples (§5) is what moves the byte cost, though — a rate floor with unchanged
lengths just means fewer samples fit.

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

Compression compounds it: `compress` is swept only at depths of 8 or fewer
([`compression_at`](../src/optisample/optimize/operating_points.py)) precisely because it trades crest
factor for headroom against a shallow grid — which also lifts the decayed tail toward the noise floor.

**The knob you asked for already exists.** Bit depth is a swept axis, and the sweep is settable from both
sides:

```bash
optisample optimize Piano.notes.json --budget-kb 512 --depth 16     # 16-bit only, one run
```

```yaml
# src/opticonfig/sweep.yaml
depths: [16] # swept bit depths
```

`--depth` is repeatable and replaces the configured list, so `--depth 16` leaves 8-bit unreachable for
that run. Compression goes with it: at depth 16 the sweep enumerates uncompressed alone.

Expect fewer samples, or shorter ones, for the same budget — the byte target has not changed, so a 16-bit
sample buys half the frames an 8-bit one does.

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
were capped at ten, and then bought the most expensive encodings on the shortlist. Measured on the two
samples they produced, the peak envelope falls 30 dB below its own peak within 2.3–4.0 s and 50 dB below it
by 7.9–8.8 s: the budget bought roughly six seconds of near-inaudible decay at 16-bit, twice.

**Levers, strongest first:**

| Knob | File | What it does |
|---|---|---|
| `reduce.trim.max_length_s` | `reduce.yaml` | Hard ceiling on every kept recording, now 5.0. Lowering it further caps the whales and frees their bytes for everyone else. |
| `reduce.dedupe.transposition_headroom_semitones` | `reduce.yaml` | 12 doubles every requirement. Set it to the widest zone you actually allow (see `max_zone_semitones`); 7 costs 1.5x instead of 2x. |
| `reduce.trim.tail_floor` | `reduce.yaml` | Where a decay stops counting as content — `1e-4` is −80 dB, which keeps far more tail than a tracker ever plays audibly. `1e-3` (−60 dB) is defensible. |

`max_length_s` is read at **every** ingest, including one reading an already-reduced dataset, so you can
re-allocate off `1_reduced/` with a lower cap and see the effect without re-reducing.

## 6. Sample count, zone width, layers

These three decide how the budget is *divided*, which sets every byte target in §2.

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

It costs measured fidelity, because the reference the objective scores against is the recording truncated
at the note's length rather than faded: the demo's `--no-loop` objective moves 0.0234 → 0.1050. Since only
non-looped candidates pay it, the fade also nudges the plan toward looping — which is the tail in item 2
below. Scoring the reference through the same ramp would settle both; it does not do that yet.

Three things read as "a tail after the piano decays":

1. **8-bit dither noise** (§4) running at −41.6 dB for the rest of the sample's length.
2. **A loop on decaying material.** The gate ([`_is_sustained`](../src/optisample/dsp/loop.py)) compares
   the last third of the analysed region to the first and accepts at `sustain_decay_ratio: 0.5`. A *short
   recording* of a low struck note passes it: over 1.3–1.8 s a bass note has fallen only 5.7–5.8 dB, which
   is a ratio of 0.513–0.516 — barely over the gate. Both loops were then placed at the shortest length
   allowed (`min_loop_s: 0.5`), and the exported instrument carries no volume envelope, so the note rings
   until the next one cuts it.

   The third looped zone is more interesting: on its own, that recording fails the gate at **0.351**
   (−9.1 dB). It only loops because `compress: true` was chosen for it — compression is applied *before*
   loop detection ([`_stored_span`](../src/optisample/dsp/surrogate/encode.py)), and lifting the decayed
   tail raises the ratio to **0.643**. So the encoding choice made for the quantizer's benefit is what made
   a struck note look sustained. `--no-loop`, or dropping 8-bit (which is the only depth compression is
   swept at), both remove that path.
3. **A neighbouring note inside the take.** One 10.16 s source take decays to −52 dB and then re-attacks to
   −19 dB, 0.29 s before it ends — the following event in the performance, captured inside this note's take.
   the 10 s cap in force at the time cut right at that onset, so the stored sample ended *on* the
   transient. Where takes overlap like this, a length cap is also a content decision: the 5 s cap now
   shipped keeps the note and leaves the neighbour out.

**Levers.** `--no-loop` stores full-length samples and takes looping off the table entirely.
`loop.sustain_decay_ratio: 0.7` keeps looping available for genuinely sustained material while declining
all three loops this run took, whose ratios are 0.513, 0.516 and 0.643. Raising `loop.min_loop_s` refuses
loops too short to be anything but a stutter.

For *where* the cut lands, `max_length_s` is no help — it is a ceiling, and the clicking samples sit far
under it. The knob that reaches `trim_s` is `reduce.events.duration_bucket_ratio`, which rounds every
scored note length *up* to a geometric grid edge; `2.0` gives each sample up to twice the decay before the
cut, and costs up to twice the bytes for it. `quantize.release_fade_s` handles the step the cut leaves.

## 9. Recipes

**"The whole pack is lo-fi."** The budget is spread too thin. In order: cap the length further
(`max_length_s: 3.0`), refuse 8-bit (`--depth 16`), then narrow the zones (`max_zone_semitones: 7`).
Re-allocate off `1_reduced/` — the cap is read at every ingest, so no re-reduce is needed.

```bash
cp -r src/opticonfig myconfig            # --config takes a directory and reads every file in it
$EDITOR myconfig/reduce.yaml             # max_length_s: 3.0
optisample optimize artifacts/1_reduced/Piano.notes.json \
    --budget-kb 512 --strategy grouped --max-layers 3 --depth 16 \
    --config myconfig --no-render --out artifacts/tuned
```

**"One sample eats the budget."** It is a wide zone spending its keys' pooled share (§2) on a long
recording (§5). Cap `max_length_s`, or lower `max_zone_semitones` so no zone pools thirteen keys.

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
| `reduce.trim.max_length_s` | `reduce.yaml` | A few samples dominate the plan. |
| `--depth 16` | CLI (`sweep.yaml: depths`) | Hiss behind the decay; 8-bit is unacceptable. |
| `--rate` (repeatable) | CLI (`sweep.yaml: rates`) | Replacing the ladder for one run. |
| `sweep.rates` | `sweep.yaml` | Reshaping the ladder itself — its floor is what long samples land on. |
| `quantize.release_fade_s` | `quantize.yaml` | Samples click at their end, or 10 ms of ramp is audible. |
| `--content-floor-db` | CLI (`reduce.yaml: bandwidth`) | You want the stored rate to track pitch. |
| `--max-samples` | CLI (`optimize.yaml`) | Trading sample count against per-sample quality. |
| `reduce.grouping.max_zone_semitones` | `reduce.yaml` | Repitching artefacts across a zone. |
| `--max-layers` | CLI (`layers.yaml`) | Dynamics matter more (or less) than fidelity per note. |
| `dedupe.transposition_headroom_semitones` | `reduce.yaml` | Survivors are longer than the music needs. |
| `reduce.events.duration_bucket_ratio` | `reduce.yaml` | Samples are cut mid-decay and click. |
| `--candidates` | CLI (`reduce.yaml: bandwidth`) | The shortlist is missing encodings you want considered. |
| `metrics.preprocess.dynamic_range_db` | `metrics.yaml` | The solver pays for material you cannot hear. |
| `optimize.energy_exponent` | `optimize.yaml` | Quiet notes are getting a budget share out of proportion. |
| `--no-loop` | CLI (`sweep.yaml: loops`) | Looped decays ring on. |
| `loop.sustain_decay_ratio` | `loop.yaml` | Struck notes are being judged loopable. |

`--config` takes a **directory** and reads every YAML file in it, so copy the whole `src/opticonfig/` tree
and edit the copy.
