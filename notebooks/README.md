# Notebooks

Interactive [marimo](https://marimo.io) notebooks for inspecting the pipeline by ear and eye.
They are a **thin** layer: all reusable logic lives in `notebooks/utils/` (type-checked, linted, and
tested like the rest of the code), so the notebook files themselves stay small and are validated by
running them, not by static analysis.

## Run

From the repo root:

```bash
uv run marimo edit notebooks/pipeline.py    # the CLI as controls, its output as an explorer
uv run marimo edit notebooks/cluster.py     # a run's sets as one space you can hover and play
uv run marimo edit notebooks/explore.py     # samples and the metric layer
uv run marimo run  notebooks/pipeline.py    # read-only app
```

## `pipeline.py` — pipeline console

Every CLI flag as a control and every stage's output as an explorer. A run shells out to
`optisample <command>` exactly as a terminal would, and the invocation behind each button is printed
above it, so anything found here is reproducible from a shell.

**Controls** — the source dataset and run root; subset fraction; budget; tracker format; strategy;
interpolation; dedupe key; content floor; rate ladder and bit depth; looping; ground-truth render;
workers; dither seed. One bundle stands behind every stage, so a control moved once reaches whichever
stage runs next.

**Stages** — `subset` carves the share of a dataset that still spans its pitch and velocity ranges,
`loop` settles the loop each recording is stored around, `reduce` writes the survivors an allocation
picks up from, and `optimize` allocates the budget. Each stage prefers what the one before it left:
looping reads the subset once it exists, reduction reads the looped dataset, and allocation reads the
reduced one, so the sweep is reached having paid only the ingest. Every transcript is filed under
`<run root>/logs/`.

**Looping explorer** — the loop each recording keeps with the seam step, level drift and timbre distance it
measured, the recordings stored over the span they play instead, and every cheaper candidate the ladder
climbed past beside the gate it fell outside. Then the **auditions**: each recording beside its loop played
out — wrapped several times under the fitted decline, which is how a seam step or a level pulse is heard.

**Reduction explorer** — each axis read as `before -> after`, every kept recording measured against
the material its pitch asks of it, the format each pitch is stored at, and the survivors written. Then
the **auditions**: the recording a pitch was judged against, beside every encoding the sweep runs
rendered to audio — the stored format made audible before the sweep is paid for.

**Allocation explorer** — the byte accounting, the encoding each kept item spends its bytes on (one
table shape for both strategies), and where every item landed on the rate-distortion plane. Then the
per-note fidelity summing back to the plan's objective, and per pitch the **A/B pair** the objective
actually scored: the recording and the module's reconstruction, side by side with spectrograms and
the sub-scores behind the number.

The run root defaults to the repository root, so `subset/`, `looped/`, `reduced/` and `artifacts/` from
a terminal run are picked up as they are. The explorers read whatever is on disk, so a long run started
in a shell can be inspected here without re-running it.

## `cluster.py` — sample space

Every recording of as many of a run's sets as you name — one instrument at one stage, the same instrument
at two, or two instruments side by side — placed as a point in one space built from what it sounds like,
cut into groups, and each group stood for by a real take you can play. Three things are held out of the
geometry on purpose. **The inaudible depth**: the first stage writes its recordings past a 30 Hz roll-off
that is 60 dB down by 10 Hz, so every stage read here holds what a listener has — rumble keeps its level
while a note decays, which is exactly where the deep anchors read.
**Level**: every reading is taken past its own frame's mean, so a note at v020 and
the same note at v100 differ by their timbre alone. **Length**: time is anchored to each recording's own
decline — anchor *k* is the moment that note had fallen *k* dB below its own peak — so a two-second take
and an eight-second take of one sound are read at the same points.

**Controls** — the run root, browsed for rather than spelled out, beside the strategy an allocated stage's
plan is read under. Every set that run left under it is then listed, one line per instrument and stage, and
as many of them as you name are read into a single space. The blocks are standardized and scaled across the
whole gathered corpus, so several sets are read under one frame: each set's coordinates answer for the
company it was read beside, and differ from what that same set would hold read on its own. Then whether to
read past each note's release, and how many workers share the reading; the frequency axis (the note's own
partials, or the mel bands the optimizer scores with), the fall depths, the harmonic count, the cepstral
coefficients and the anchor span; the four block weights and the share of the corpus a depth must reach to
be read at all; the algorithm, linkage, group count and which member stands for its group; then the layout,
2D or 3D, and what to color by — the set a take came from among the choices, which is what tells several of
them apart by eye.

The reading of the recordings is the expensive half and is done once — moving a weight or a group count
re-reads the geometry off blocks already read, so those controls respond immediately while naming another
set or another frequency axis re-reads the audio.

**The space** — every recording where the layout placed it, hovering everything it was read for. A column
holding names (the group, the note) draws one color and one legend entry apiece, so clicking the legend
isolates a group; a column holding measurements is shaded along a scale. The takes standing for their groups
are ringed, each in its own group's color, so a selection is picked out by eye whatever the field is
colored by. Clicking any point of a flat picture plays that recording — normalized, so a quiet take is as
audible as a loud one — and sends it to the examine panel below; a box is turned and read by eye, and a
recording is picked out of one by naming it in that panel.

Each group goes by the key of the take standing for it — `p060_v100` is a group stood for by middle C
struck at velocity 100 — and is drawn in the color that key turns: the pitch sets the hue and the velocity
fills it in, opening a quarter of the way up so the softest takes keep a color of their own. The field
therefore carries the keyboard, and one key is drawn the same color in every picture, at every stage and
across datasets. Where two groups are stood for by takes of one key, the second onward carries a number.

PCA and MDS draw the space's own geometry, so what is read off the picture holds in the numbers. t-SNE
and UMAP are additional variants for the eye — they place each recording beside the company it keeps, so
a group reads as a cluster while the room between clusters follows the neighborhoods. They appear in the
layout list when scikit-learn and umap-learn are installed; without them the notebook draws the pair that
stands on the space's own distances. **Keys** leaves the space aside and lays the corpus out as the keyboard
holds it: the note across, the velocity it was struck at up, and in three dimensions how long each take
rings. That says which keys the sets kept a recording of and how a group sits across them, takes sharing a
key standing on one point whichever set each of them came from — colored, ringed and clicked exactly as
the space is, since it is the same field drawn on other axes. Every number the panels report is the
space's own whichever layout is on screen.

**How many groups** — the silhouette against the number of groups, with the count on screen ringed, and
the top of the tree with the height the cut reads it at drawn across it.

**Groups and members** — what each group gathered (size, pitch and velocity range, playing time, spread
from its medoid, and how many sets it drew on, which is what names a sound several of them share) and how
tightly it holds together block by block, which says where the grouping came from. Then one group's
recordings, ordered from its medoid outwards. Every take is named by the set it came from, so two sets
holding a recording of one file stem stay apart wherever they are read.

**Examine and play** — the picked recording, from either picture or by name, heard and seen: a player, its
waveform, its spectrogram, the
level it holds against the decline and the curve fitted to it, what it sounded like at each depth it
reached, its own readings, the depths it arrived at beside the ones the space reads, and how far it
stands from every group — the company it nearly kept.

**Representatives** — every group's take in one row of players, so a whole selection is auditioned at
once.

**Feed the selection back** — those same takes written out, holding every note they answer for, each
carried over exactly as the stage states it, beside a copy of each recording. Each set the picks came from
lands as a dataset of its own, under `<write to>/<instrument>/<stage>/`, so a selection gathered across
sets feeds each of them back on its own terms. What lands is a NoteExtractor dataset like any other, so
`optisample pipeline <written>/<instrument>/<stage>/<instrument>.notes.json` runs the rest of the pipeline
over a set chosen for what it sounds like — a selection rule the pipeline's own stages have none of, since
dedup keeps a recording by the identity and the length it carries. The button writes only when pressed,
and the panel is offered when every named set is one of the three dataset stages; what an allocation
stored is read back through its own plan.

Point the run root at a `optisample pipeline --out` directory; the set list reads whichever of `0_subset`,
`1_looped`, `2_reduced` and `3_optimized` that run wrote, one entry per instrument each of them holds.

## `explore.py` — sample & metric inspector

- **Per instrument:** budget roll-up (full-length 16-/8-bit storage vs the byte budget) and the
  material (notes the song actually plays).
- **Per sample:** audio playback, waveform, dynamic-range-floored spectrogram, and a descriptor +
  footprint table.
- **Degradation lab:** apply a preview degradation (bit-depth, lowpass, gain, resample) and hear the
  original vs. the result while the composite fidelity, per-metric breakdown, and diagnostics update.
- **Smoke-test panel:** the fixed P1 degradation set with a stacked bar decomposing the composite
  (bar height ≈ fidelity) — the interactive form of the P1 smoke test.

On first run it generates a synthetic demo project under `notebooks/_demo/` (git-ignored) — one
`.notes.json` + samples directory per preset — and loads them into a combined manifest. Point the
*demo directory* field at your own directory of `.notes.json` files to inspect real recordings.
