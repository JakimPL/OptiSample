# The clustered carrier instrument, then carriers through the whole pipeline

## Context

This file is the living handover. It supersedes the previous version, which was written on a different
machine against a dataset that was not present there; the corrections it needed are listed below.

The premise is unchanged: **a stored sample should hold timbre, not level.** Today every sample carries its own decline in
the PCM, so the attack sits at full scale and the tail falls into the quantizer's floor — exactly where
8-bit storage does its damage. Dividing the level out (`decompose`, exact, already written into every
`.sample` container) lifts the whole note toward full scale and hands the decline to the per-instrument
volume envelope, which is what a tracker has an envelope for.

**The objective, as stated this session:** 16→8 bits halves the bytes and so doubles what a budget buys.
The carrier is the mechanism that makes that conversion least damaging. It is worth pursuing on that ground
even where an aggregate metric reads flat — so the work is *how well can 8 bits be made to work, and for
which clips*, not *should we try 8 bits*.

**What changed this session.** Before the large plan is touched, a new first phase is added: a **short
route from a clustered cut straight to a playable instrument**, taking a given number of groups per
velocity band and building the samples out of the group representatives, stored as **looped carriers**.
It exists to give quick, audible results ahead of the long work.

It is not a detour. Measured on a real container this session, the carrier saves **9.6 dB of crest factor
(≈1.6 bits)** on a long decaying note, and encoding a carrier through the existing `encode` chain already
works — the one piece with no home today is **replaying the level curve at playback**, because
`StoredSample.gain` only undoes peak normalization. That missing piece is precisely Phase D's mechanism.
Building it here, in a route with no byte budget and no allocation DP, **validates the carrier idea cheaply
and early** and leaves Phase D a proven component to adopt. And because the route goes through
`prepare_loop`, Phase C's crossfade fix reaches it with no further work.

Phase A shipped and is committed. This plan covers **Phase I (new)** and then **B–F**.

---

## Corrections to the handover — verified this session

Everything in the handover's **Map** was checked; it is largely accurate. These are the deltas that change
what gets executed.

### Repository state

| Handover says | Actual |
|---|---|
| Phase A is "complete and **uncommitted**" | **Committed** as `019162f` (21 files). Tree clean; only untracked file is `plan.md`. |
| `notebooks/utils/{clusters,scatter}.py` are "not part of it" | They went in with `019162f`. Moot. |
| "24 cores… use `-n 4`" | `nproc` = **8**. `-n 4` still right, for a different reason. |

**The environment is now green** — the user initialized `trackmod`, installed `openmpt123`, and provided
`./artifacts`. `uv run` works. On disk:

- `artifacts/0_subset/Piano` — **415 WAVs**, 61 pitches (MIDI 29–101), irregular velocity coverage (a real
  sparse capture, not a grid). This stands in for the entire dataset.
- `artifacts/1_looped` — **401 WAVs and 401 `.sample` containers** at 48 kHz. Offers per recording:
  `{4: 94, 5: 80, 2: 72, 3: 62, 1: 53, 0: 40}` — **40 recordings offer no loop at all**. Rejections across
  the whole corpus are tiny: 12 on timbre, 1 on seam.
- `artifacts/3_optimized/Piano/grouped` — a complete 512 KiB run: **24 stored samples over 61 keys and 2
  velocity layers**, objective **15.925959** (`plan_objective` 15.926074), every zone at **depth 16, comp
  `-`**. This is what Phase I is judged against by ear.

### Technical corrections that change the work

- **`scratchpad/retracted_6cii_seam_loss_meter.patch` is gone, and the meter was never tracked code.**
  `git log -S seam_loss_db --all` finds nothing; `66e7c2f` *Retracted: the seam loss meter* touches only
  `README.md` and `docs/tuning.md`; `git fsck --lost-found` yields nothing but a stale `plan.md`. **Phase C
  rebuilds it from spec.** The spec survives in the `66e7c2f` README diff: per-band level at the wrap,
  bands past 1 dB and past 3 dB out of 768, deepest notch (it read 170 → 23, 75 → 4, −8.6 → −4.5 dB).
- **Going to 8 bits silently switches the compressor on.** `compresses(sweep, depth)`
  (`optimize/operating_points.py:83-90`) is `sweep.compress and depth <= _COMPRESSIBLE_DEPTH`, with
  `_COMPRESSIBLE_DEPTH = 8`. `sweep.yaml` pins `depth: 16`, so **the compressor is dead code in every
  default run** — confirmed by `comp -` across all 24 zones. Every 8-bit measurement is taken both ways.
- **`stored_encodings` carries an index-ordering contract a depth axis breaks**
  (`optimize/reduce/bandwidth.py:178-180`: "the trimmed span leads however many loops there are, so it sits
  at the same index for every clip"). Phase D must redefine it deliberately.
- **The envelope floor is soft**: `dsp/envelope.py:111-112` adds it *in quadrature*, not as a clamp.
- **The ordering inversion is worse than stated.** `build_song` (`optimize/export/build.py:118-131`) runs
  `plan_samples` (encode + quantize + gains) at `:119`, then fits envelopes from `planned.stored` at `:126`.
  Beyond that, `sample_gains` (`optimize/export/samples.py:97-116`) needs *every* encoded sample to state
  each gain against the loudest, and `PlayedVoices` consumes `planned.gains`. Inverting order touches
  `plan_samples`, not only `build_song`.
- **`level_loop` runs before the crossfade and the asymmetry is structural** (`dsp/loop.py:474-481`): the
  fade *target* is levelled, the fade *source* `[start-fade, start)` is not. Confirms candidate #4.
  **Measured on a carrier it moves 0.13 dB** — effectively but not exactly a no-op, because
  `local_level(s/L) ≠ local_level(s)/L` when `L` varies inside the kernel.
- **`_apply_loop` does not call `loop_at_rate`** — that is in `_looped_span` (`dsp/surrogate/encode.py:71`).
  `encode()` receives the **raw** recording, so the stored crossfade is an *independent* fade at the stored
  rate, not a second fade over faded audio. Candidate #2 is about **placement**: zero-crossing and
  whole-period snapping at the analysis rate is rescaled with ±0.5-frame rounding before the seam is recomputed.
- **`_stored_span`'s release fade covers only the unlooped branches** (`dsp/surrogate/encode.py:76`).
- **The `.sample` container and `loops.json` are write-only.** `read_sample` / `sample_decomposition` /
  `sample_loops` have zero call sites in `src/`. Worse, **stage 2 re-derives everything**: `reduce_project`
  (`artifacts/reduced.py:234-256`) re-decodes the WAVs and re-settles every loop, ignoring both. Phase I
  becomes the first production reader.
- **Phase E details:** `reserved_slots` (`optimize/layers/slots.py:173-181`) reserves over *key counts per
  band*; `pack_slots` with `per_instrument=0` raises an unguarded `ValueError`; `solve_grouping`'s `reserve`
  shapes the partition but is not billed into reported `total_bytes`; XM under `extended` compliance reaches
  255 samples/instrument, not 16 (canonical is 16, and that cap binds this route).

---

## Phase I — the clustered carrier instrument. **Done, uncommitted.**

`optisample cluster <run-root> --groups N --layers L --depth 8` writes a run's recordings as playable
carrier instruments. The reusable half lives in **`src/optisample/carrier/`** (`source`, `shape`, `store`,
`instrument`, `audition`) and knows nothing of clustering or budgets — **Phase D adopts it directly**. The
stage driving it is `cluster/instruments.py` — it lives there, not under `artifacts/`, because
`cluster/` already depends on `artifacts/` and putting it the other way round closed a real import cycle.
The manifest it writes is `artifacts/documents/clustered.py`.

**Three rules were extracted rather than duplicated**, each to the module that owns the concern:
`played_gain`/`sounding_gain` → `optimize/export/envelope.py` (out of `artifacts/instruments/normalize.py`);
`balanced_gains` and `sample_label` → `io/tracker/target.py` (out of `optimize/export/samples.py`).
`cluster/stages.py` gained `stage_material`, and `artifacts/paths.SAMPLE_EXTENSION` went public — this is
the **first production reader of the `.sample` container**. `cli.py` crossed 1000 lines, so the reporting
block moved to `src/optisample/console.py`.

**Measured on the real Piano set** (`artifacts/1_looped`, 401 recordings, `--groups 8 --layers 2 --depth 8`):
two instruments, 16 samples, **205.7 KiB total** — `v000-v034` 8 samples/146.1 KiB/**dispersion 17.69 dB**,
`v035-v127` 8 samples/59.6 KiB/**dispersion 10.04 dB**. 15 of 16 stored around a loop; one had no offer and
stored the span it plays.

**The carrier claim, verified on real containers.** Waveform × envelope × step reconstructs the recording at
**21–34 dB SNR** over the stored span, and dropping 16 → 8 bits costs only **1–2 dB** while halving the
bytes. At 8 bits the quantizer is no longer the bottleneck — the envelope's fit is (dispersion ≫ the depth
penalty), which is exactly what Phases E and F are about.

**Two honest limits.** Absolute output level is uncalibrated — `balanced_gains` states the set relative to
its loudest sample, so the balance *between* samples is exact while the whole instrument sits at an
arbitrary master level (the same property the optimizer's `sample_gains` has). And one format is written
per run (`--format`), not both.

**Known environment issues, neither caused by this work:** `openmpt123` aborts under `LANG=en_US.UTF-8`
because that locale is not generated (`locale::facet::_S_create_c_locale name not valid`) — run tests with
`LC_ALL=C`, or `sudo locale-gen en_US.UTF-8`. And `describe_corpus` over 401 recordings at `--workers 4`
exhausts memory on this 23 GB machine; `--workers 1` completes. Worth a look before Phase F leans on it.

Proposed message: **_Added: the instrument a cut sample space is written as, carrier and curve apart_**

---

## Phase I — what it does (design)

**One command:** a source, a group count and a layer count in; a playable, auditioned instrument out, whose
samples are looped carriers and whose decline is carried by a shared volume envelope.

No byte budget, no rate-distortion sweep, no allocation DP.

### Decisions taken (this session)

- **Bands first, N groups per band.** `--layers L --groups N` cuts velocity into L bands, then cuts each
  band's space into N groups → **L instruments of up to N samples each**. A tracker keymap has no velocity
  axis (`trackmod` has none anywhere), so one instrument per band is forced by the format; the velocity
  mapping rides in `bank.json`, which already carries exactly that (`documents/bank.py:_selector`).
- **Store the most faithful offer** — the loop with the lowest `spectral_distance`. Note the container
  stores offers **cheapest-span-first**, so this is *not* index 0; it must be sorted. **The 40 recordings
  with no offers store the trimmed span** (`_asked_loop` already falls back).
- **A new CLI subcommand**, following the existing `_run_X` / settings-dataclass / `_print_X` idiom. The
  clustering subsystem has no CLI at all today — it is reachable only through `notebooks/cluster.py`.

### Ownership — where the new code goes

Two pieces, deliberately separated so Phase D inherits the reusable half:

1. **`src/optisample/carrier/`** (new subpackage) — *a set of recordings written as carrier samples under
   one shared envelope*. This is the general mechanism: fit the shape, write it, divide each carrier by the
   gain it applies, state the remainder in the format's level fields, route the keys. It knows nothing about
   clustering or budgets. **Phase D adopts this directly.**
2. **`src/optisample/cluster/instruments.py`** — the stage that drives it: read the cut, pick
   representatives, call `carrier/`, write files and auditions.

Output root `artifacts/clustered/` rather than a numbered stage, because this route **branches off
`1_looped`** rather than following `3_optimized`.

### What already exists and must be reused

| Need | Existing machinery |
|---|---|
| Velocity bands cut by playing time | `optimize/layers/bands.py` — `velocity_cells(material, nodes)`, `VelocityBand`, `VelocityLayers` |
| A stage's recordings, decoded as the optimizer decodes them | `cluster/stages.py` — `stage_recordings`, `StageCorpus`, `StageSettings` |
| Describe once, re-weigh cheaply | `cluster/corpus.py` — `describe_corpus`, `DescribedCorpus.space(config)` |
| The cut and the representative | `cluster/partition.py` — `partition(coordinates, groups=…, config=…)`; `cluster/representative.py` — `grouping(...)`, which resolves medoid / weighted-medoid / nearest-centroid to a **real member** and enforces `min_duration_s` |
| Carrier, level and the settled loops off disk | `artifacts/documents/sample.py` — `read_sample`, `sample_decomposition`, `sample_loops` |
| The wrap | `dsp/loop.py` `prepare_loop` (via `encode`), so Phase C lands here free |
| Encoding one recording standalone | `dsp/surrogate/encode.py` — `encode(signal, rate, EncodingParams(...), EncodeContext(root_pitch, config.encode, settled, rng))`. **Verified standalone this session**; it imports nothing from `optimize` |
| One envelope over many members | `dsp/trajectory.py` — `fit_shared_trajectory`, `TrajectoryMember`, `SharedTrajectory.dispersion_db` |
| Writing that curve on the format's grid | `optimize/export/envelope.py` — `volume_envelope`, `envelope_grid`, `shape_nodes`, `QUIETEST_STEP` |
| Carrier ÷ written envelope, and the level split | `artifacts/instruments/normalize.py` — `played_gain`, `stored_level`, `StoredLevel` |
| N samples in one instrument, keys routed | `optimize/export/samples.py:_unit_assignments`/`_slot_keymaps` pattern + `optimize/export/coverage.py:covered_routing`, which **splits the keyboard at the midpoints automatically** from each sample's root key — no explicit key ranges needed |
| Inter-sample balance | `optimize/export/samples.py` — `sample_gains`, `_makeup` (IT); `EncodeConfig.peak_reference` (XM) |
| Writing `.iti`/`.xi` and the bank | `io/tracker/target.py:instrument_file`; `artifacts/container.py` — `bank_contents`, `write_container`; `documents/bank.py:bank_document` |

**Nothing here needs inventing** except the one piece named in the Context: replaying the level curve.

### The route

1. **Read the stage** (default `1_looped`) with `stage_recordings`; read the manifest's material for the
   velocity cut.
2. **Cut velocity into L bands** with `velocity_cells(material, L)`.
3. **Describe the corpus once** with `describe_corpus`, then per band subset the descriptors, build the
   space and `partition(..., groups=N)` → `grouping(...)` → one representative per group. Describing is the
   expensive half; doing it once and subsetting is what keeps the route quick.
4. **Per representative**, `read_sample` its container → carrier, level, offers. Choose the offer with the
   lowest `spectral_distance`; store trimmed where there are none.
5. **Fit one shared trajectory per band** from the representatives' own `level` curves (read through
   `level_readings` on the window `reading_window_s` gives), weighted by playing time, and write it with
   `volume_envelope`. **Report `dispersion_db`** — that number says whether the group count was well chosen,
   and it is the first real evidence for Phases E and F.
6. **Store the carrier**: divide by `played_gain` of the written envelope floored at
   `QUIETEST_STEP / MAX_VOLUME`, then `encode` at the chosen rate and depth with the chosen loop.
7. **Balance and route**: `sample_gains` for IT, `peak_reference` for XM; keymap by merging each sample's
   root-key assignment and passing it through `covered_routing`.
8. **Write** one instrument per band plus `bank.json`, and auditions so it can be heard at once.

### Surface

```
optisample cluster artifacts/1_looped/Piano.notes.json \
    --groups 8 --layers 2 --depth 8 --out artifacts/clustered
```

`--groups` overrides `cluster.partition.groups`, `--layers` the velocity split, `--stage` which stage to
read, `--depth` so 8-bit is audible against 16-bit immediately. Config additions belong under
`opticonfig/cluster/`.

### Constraints to respect

- **XM canonical caps 16 samples per instrument**; IT allows 255. `--groups` above 16 either splits the
  band `pack_slots`-style or writes IT only — state which, do not fail silently.
- **`StoredLevel.gap_db`** must be reported per representative, so a recording the format had no room to
  sound at its captured amplitude is stated rather than passing silently (`dump.py` already calls this
  `understated`).

---

## Phase B — how much of 8 bits the carrier buys back. **Done, measured.**

Probes at `scratchpad/{carrier_worth,transposition}.py`, read-only, over **399 of the 401 `.sample`
containers** at 22050 Hz, each cell scored end to end — what a player actually puts out, against the
recording over the span the sample stores.

| kept as | median segSNR | median composite (lower is better) |
|---|---|---|
| recording @16 | 30.33 dB | 0.0995 |
| recording @8 | 27.01 dB | 2.8754 |
| recording @8 compressed | 16.06 dB | 3.4973 |
| **carrier @16** | **33.22 dB** | **0.0449** |
| **carrier @8** | **32.99 dB** | **2.1520** |
| carrier @8 compressed | 19.12 dB | 2.9596 |

**The headline answers yes.** `carrier@8` reads **+1.14 dB median above `recording@16`** — 71.2 % of clips
beat it outright, **93.5 % come within 1 dB** and 98.7 % within 3 dB. Half the bytes for the same fidelity
is real, on this material.

**The last bit costs a carrier almost nothing: median +0.09 dB** (p90 +0.56) against **3.32 dB** for the
recording. That is the whole thesis in one number — flattening the waveform is what makes the eighth bit
cheap, because the quantizer stops fighting a 40 dB decline.

**The compressor must be decoupled from depth before D offers 8 bits.** `compresses(sweep, 8)` is `True`
today, and switching it on costs the recording **11 dB** and the carrier **13.9 dB** of segmental SNR. An
8-bit run would silently take that hit. This is now a requirement of Phase D, not an open question.

**The two metrics disagree, and the disagreement is Phase 3's.** Both rank the carrier better than the
recording *at matched depth* (0.0449 < 0.0995 at 16 bits, 2.15 < 2.88 at 8). They part on whether
`carrier@8` matches `recording@16`: segmental SNR says yes, the composite says no (2.15 against 0.0995).
The composite is level-invariant and cannot see what the envelope does, so it is the weaker witness here —
but this is exactly the ranking the listening set exists to settle.

**The envelope reaches most of the decline.** The material travels a median **18.3 dB** and the written
curve reaches **16.1 dB** of it, so about 2 dB stays in the waveform — which is the residual the carrier
carries by construction.

**The transposition residual is real but is not the main term.** Reading real pairs of recordings at close
velocities, the gap one unscaled envelope leaves between two keys runs **5.22 dB at one semitone**, 6.70 dB
at an octave and **9.66 dB at two octaves** (median; p90 12–16 dB). Most of it is note-to-note variation
rather than transposition — the tick clock adds roughly **4.4 dB** across two octaves on top of a ~5 dB
floor. That floor is why the clustered run reads 10–17 dB of dispersion, and it is the strongest argument
for Phase E: narrowing the zone cannot go below it, but one instrument per sample removes it entirely.

Proposed message: **_Measured: what a carrier is worth at eight bits, and what the tick clock costs it_**

---

## Phase B — what the probe does (design)

Read-only probe at `scratchpad/carrier_worth.py`. **No production code.** Material is on disk; read the
`.sample` containers under `artifacts/1_looped` directly.

Reframed per the user: a **per-clip triage**, not a go/no-go — how far the carrier closes the 16→8 gap and
which clips can go to 8. That is the shape Phase D consumes.

The grid, per recording, over the stored span:

| | 16 bit | 8 bit plain | 8 bit compressed |
|---|---|---|---|
| recording | baseline | | |
| carrier | | | |

The compressed column exists because `compresses(sweep, 8)` is `True` by default.

**Per recording:** segmental SNR and the composite per cell; how much of the decline the 36 dB envelope
reached and how much stayed in the carrier; and the **transposition residual** — the envelope runs on the
tracker's tick clock and does not scale with pitch, so a carrier played a fifth up is played down by a curve
fitted at the root. Read the gap across each zone's real key span.

**In aggregate:** the distribution of "carrier@8 − recording@16" per clip and the share of clips within each
damage threshold. That distribution is Phase D's depth-selection prior.

Phase 7 settled the carrier at **+5.4…+26.1 dB segmental SNR at 8 bits** (median ≈ +15 dB ≈ 2.5 bits), and
the crest-factor reading above (9.6 dB ≈ 1.6 bits on a long note) is consistent with it. B asks where the
remaining gap sits, and Phase I says how it sounds.

---

## Phase C — the crossfade: build the meter, then find the missing component

`crossfade_loop` (`dsp/loop.py:426-450`) already runs OpenMPT's direction, and the endpoint identity is
*exact* (at `progress = 1.0` the gain is exactly 1). What is wrong is a component.

**C-i — the meter was rebuilt, and it did not validate. Probe at `scratchpad/seam_loss.py`.**

Rebuilt from the surviving description: per band, how far the blended level sits under the equal-power
interpolation of the two sides' own levels, bands under −40 dB of the loudest left out. Read over **1159
loops / 5808 carried bands** of the Piano corpus, against three laws — plain equal-power, the shipped
banded law, and the shipped law with its lift uncapped:

| law | past 1 dB | past 3 dB | deepest | median |
|---|---|---|---|---|
| plain | 2977 | 2195 | −33.30 dB | −1.17 dB |
| banded (shipped) | 3852 | 2405 | −30.54 dB | −2.14 dB |
| uncapped | 3852 | 2397 | −30.54 dB | −2.14 dB |

**This does not reproduce the retracted figures** (170 → 23 of 768 past 1 dB), and the corpora are not
comparable either, so per the plan's own rule the meter is unvalidated and **no conclusion about the
weighting law may rest on it**. Its reference model is the suspect part: equal-power interpolation
under-predicts what a *correlated* pair of sides puts out, so it scores the plain law generously.

**One thing it does settle.** Banded and uncapped are identical to three decimal places — 3852/3852 past
1 dB, 2405/2397 past 3 dB, the same deepest notch and the same median. **`_MAX_SEAM_GAIN = 2.0` is
effectively inert**: it changes 8 bands in 5808 (0.14 %). So candidate #1's *cap* is not the fault, and if
the anti-correlation lift is the missing component it is the lift itself, not its ceiling. Settling that
needs the OpenMPT exports.

**C-i (redo) — rebuild the meter against ground truth.** `seam_step` (`dsp/loop.py:484-499`) is a single-sample endpoint ratio
normalized by local frame-to-frame motion, blind to a level notch or a phasey blend. Rebuild the per-band
seam-loss meter from the `66e7c2f` spec as a **probe, not a gate**, on the blend's own band bank. Reproduce
the retracted figures on today's code first — if they do not reproduce, the meter is wrong before anything
else is.

**C-ii — a tunable XFadeSample.** Reimplement OpenMPT's crossfade with its parameters exposed: fade length,
fade law exponent (0…1, 0.5 = constant power), and the after-loop fade. Diff against ours on the same
recording and loop points under the C-i meter. **The user will supply OpenMPT exports later** to calibrate
it; the parameterization is built so those exports tune it rather than force a rewrite.

**Four candidates, in order:**

1. **The anti-correlation lift.** `_seam_weights` (`dsp/loop.py:366-386`) divides both sides by
   `sqrt(1 + rho·sin2θ)`, floored so gain caps at `_MAX_SEAM_GAIN = 2.0` (+6 dB). Where a band
   anti-correlates, both sides are boosted — restoring RMS by amplifying a cancellation residual. OpenMPT
   has no such term. Likeliest source of a phasey mid-fade once per round.
2. **Loop placement spent on the rescale** — `loop_at_rate`'s ±0.5-frame rounding after analysis-rate snapping.
3. **No guard frame past `loop.end`** (`dsp/surrogate/encode.py:35`; asserted at
   `tests/dsp/surrogate/test_encode.py:75`, `:214`). OpenMPT writes `sample[loopEnd] = sample[loopStart]`.
4. **`level_loop` before the blend** — confirmed asymmetric, and measured at 0.13 dB on a carrier. **Phase D
   deletes this stage**; Phase I can leave it in (it keeps the path identical to stage 1's auditions).

**Then the recarry.** Whatever law survives, the blend perturbs the level, so re-read the level over the
blended region and divide it out, keeping the envelope fitted before the blend. In a carrier world this is
the only level correction the seam needs, which is what lets the correlation-derived gain go.

**Phase I inherits every one of these fixes unchanged**, since it stores through `encode` → `prepare_loop`.

---

## Phase D — carrier storage through the optimizer

Depends on B's triage, C's law, and what Phase I proves audibly. **This moves the freeze.**

**What is stored** is what Phase I already builds — `src/optisample/carrier/` is adopted here rather than
rewritten. The new work is doing it for a slot serving many keys under a budget.

**Breaking the ordering circle.** `slot_members` reads the delivered level off the stored PCM
(`optimize/export/voices.py:210-213`), so encoding precedes the envelope today. Break it as `normalize.py`
and Phase I do: fit the shared shape from the **members' own reference levels** (the `true` term at
`voices.py:152-153`), write the envelope, then encode each sample divided by that envelope's played gain on
its own clock. No iteration — the residual the envelope cannot express is exactly what stays in the carrier
by construction. This also reaches `plan_samples`, because `sample_gains` needs the whole encoded set.

**The dynamics stage is now an axis, not a step. Shipped.** `stored_encodings` offers each stored span
both plain and compressed wherever the depth leaves a grid shallow enough for compression to buy headroom,
and the objective picks between them -- which is what the README always claimed and the code never did.
At 16 bits nothing is offered compressed, so **the demo freeze reproduces exactly** (piano 0.176988
ungrouped / 0.177521 grouped at 70 644 B, clean 0.176837 both). Phase B's evidence says the objective will
now decline compression at 8 bits rather than being handed it.

**Depth as a priced axis.** `stored_encodings` offers each clip at 8 and 16 bits and
`frontier.lower_convex_hull` picks. Two consequences: **redefine the encoding index contract**, and
**revisit `compresses(sweep, depth)`** — a carrier already spends the range the compressor was buying, and
the coupling is the source of the listening set's median 3.49 LU level gap. Plus the **blanket mode** the
user asked for: keep everything at 16 through the run and downcast everything at the reduce step.

**What `sample_gains` means now.** It still carries inter-sample balance; the envelope carries intra-note
level. `_makeup` (`optimize/export/samples.py:80-94`) reads `stored.playback_gain`, which becomes the
carrier's peak rather than the recording's — check the balance still comes out right.

**Worth folding in:** stage 2 re-settles every loop from scratch, ignoring `loops.json` and the containers
(`artifacts/reduced.py:234-256`). Once Phase I proves the container is a sound production input, that
duplication is a cheap win.

---

## Phase E — one instrument per sample

`pack_slots(units, layers, per_instrument)` (`optimize/layers/slots.py:140`) already cuts a band into
instruments owning ascending key runs; `plan_slots` (`:164-170`) passes `target.max_samples_per_instrument`.
Passing `1` is the change. Guard `per_instrument = 0`, which today raises an unguarded `ValueError`.

The cost is bytes and must stay honest. IT charges **558 B** per populated instrument and numbers 255:

- `reserved_slots` (`:173-181`) reserves the worst case — a sample per key played, per band — because zones
  are unknown before the solve. At `per_instrument = 1` that is 183 instruments on a 61-key 3-layer slice
  and **264 on a full 88-key keyboard, past the 255 the format numbers**. The pessimistic reserve, not the
  real count, is what breaks.
- The fix is built: `solve_grouping(..., reserve=int)` (`optimize/grouping/solve.py:168-174`, charged in
  `_charged` at `:50-60`) charges every stored sample a byte reserve and shapes the partition toward fewer,
  wider zones. Set `reserve = populated_instrument_bytes(storage)` and drop the up-front worst case. It must
  compose with `grouping/reserve.py`'s use of the same field as the sample-cap shadow price: the two add.

Phase 8b measured the split at 7.78 → 5.30 dB and 9.22 → 7.26 dB — modest, because most of the residual is
attack, not decline. Under carriers the argument is stronger. **Re-measure.** Phase I's per-band
`dispersion_db` is the first evidence and arrives long before E does.

---

## Phase F — can clustering pick the groups? (research)

**Phase I front-loads most of this phase's apparatus**, so by the time F is reached the question is narrow:
does cutting on the contour beat contiguous key ranges at holding one envelope?

`cluster/space.py` builds the geometry from four blocks, one of which is `envelope` — the contour, how long
the note took to reach each fall depth and how straight it falls — weighted 0.25 in
`opticonfig/cluster/space.yaml`. Raise `envelope_weight`, cut, and read `SharedTrajectory.dispersion_db` per
group against the dispersion of the same samples grouped by key range as today. If clustering wins, the
grouping rule has an answer; if not, E stands.

---

## Verification

**The freeze.** I, B and C keep it; **D breaks it deliberately**; D and E each re-measure and state what moved.

- Run the demo baseline **once before anything changes** — `artifacts/demo/` is present:
  `optimize artifacts/demo/{piano,strings}.notes.json --budget-kb 96 --no-render --seed 137 --strategy both`
  reproduced piano **0.176988** ungrouped / **0.177521** grouped at 70 644 B, strings grouped
  **5.950426 / 96 244 B** and ungrouped **11.209624 / 86 108 B**; clean recomputation piano **0.176837**,
  strings **5.951006** / **11.211237**.
- The 512 KiB Piano run on disk reads objective **15.925959** (`plan_objective` 15.926074), 24 samples, 2
  layers. Re-derive before Phase D touches anything.
- The handover's 62-note slice contract (**1.201655 / 130 154 B**) predates Phase A's floor, so it is
  re-baselined the first time the slice is rebuilt — record the new figure, do not read the move as a
  regression.
- `plan_objective` and `metrics.json`'s `objective` disagree because the sweep draws dither from one stream
  in sweep order, so changing how many encodings a pitch offers re-phases every later pitch's dither. **This
  matters most in D**, which changes exactly that count. The clean side is the one to trust.

**By ear.** Phase I is itself the audition — compare it against `artifacts/3_optimized/Piano/grouped`, and
compare `--depth 8` against `--depth 16` on the same cut. C needs the same recording looped both ways, held
several rounds. D needs recording-at-8 against carrier-at-8. Write these into `artifacts/listening/`.

**Gates, after every phase.** `make lint` (mypy + pylint), `make test` with `-n 4` under the 99 % coverage
gate, and `pre-commit` — which must run under a pty, `script -qec "uv run pre-commit run --files ..."
/dev/null`, because `yamlfmt` crashes in a non-TTY through its own `os.ttyname` bug. Narrowing pytest needs
`--override-ini="testpaths="` with paths last, plus `--no-cov` for subsets. Ask before long runs.

New tests mirror ownership: `tests/carrier/`, `tests/cluster/test_instruments.py`, and the CLI subcommand in
`tests/test_cli.py` alongside the existing per-command cases.

**Working agreements.** Stop after each phase with one sentence naming the next and a proposed *Did: what*
commit message. Never commit unless asked. Read `docs/guidelines.md` before coding. **Rewrite `plan.md` in
the repo root to match this plan as the first act**, so the next session inherits the corrected version.

---

## Still live — Phase 3, the metric's ground truth

Blocked on the user's listening time, not on code. `artifacts/listening/Piano` holds 72 questions from 419
priced encodings over 38 pitches. Answer with `marimo edit notebooks/listen.py`; read back with
`optisample rank artifacts/listening/Piano`. Then the three fixes the labels adjudicate: frame-level
weighting by absolute level, a noise-to-mask term, and percentile rather than mean aggregation.

Two things to know first: every `compress` pair carries a median 3.49 LU level gap, which D may retire — and
those pairs could only have come from a non-default run, since `depth: 16` disables the compressor. And read
over the shared stretch, the two sides sit a median **11.5 LU** below their reference (p90 16.9, max 24.7),
worsening as velocity falls, with `023_p030_F#1_v018` rendering both sides as digital silence. It shows on
`rate` questions where the axis moves no level, so it is the render path. Worth chasing on its own.

## Settled — do not re-derive

- The carrier is worth **+5.4…+26.1 dB segmental SNR at 8 bits** (median ≈ +15 dB ≈ 2.5 bits) on the stored
  span alone (Phase 7); read end to end through the envelope it is worth **+5.98 dB** against the recording
  at the same depth, and **+1.14 dB against the recording at 16 bits** (Phase B, 399 recordings).
- **The eighth bit costs a carrier 0.09 dB and a recording 3.32 dB** (Phase B). This is the trade.
- **The dynamics compressor costs 11–14 dB of segmental SNR** and is switched on by depth alone
  (`_COMPRESSIBLE_DEPTH = 8`). Decouple it before any 8-bit run (Phase B).
- A shared 24-node shape takes today's ramp from **6.20 → 2.08 dB** and **4.15 → 3.36 dB** (8b); writing it
  to the format's grid costs 0.06 dB (8c). The node budget is **not** binding — 11 nodes read within 0.5 dB
  of 24.
- The swing inside a loop is the material's own (6d): looping adds −0.29 dB median. Dead and measured:
  sub-sample end placement, equal-energy seam weighting, whole-beat-cycle loop lengths, per-band flattening.
- `dsp/spectral.split_bands` earned its keep at the seam and nowhere further.

## Deferred

Phase 9 (one global λ across the interval partition and across instruments) — re-measure first. Unheard:
Phase 5's level-reading A/B.
