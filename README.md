# OptiSample

OptiSample turns an instrument's recorded sample grid into a byte-budgeted Impulse Tracker (`.IT`)
module: it stores one sample per key, collapses the velocity axis to one representative per pitch, and
reproduces dynamics through a velocity→volume map.

## Requirements

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
against each WAV filename's leading index token. Deduplication to one recording per `(pitch, velocity)`
is the optimizer's job.

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
flags: `--strategy {both,grouped,ungrouped}`, `--no-render`, `--rate`/`--depth` (repeatable sweep
values), `--no-loop`, `--interpolation`, `--seed`.

Generate a synthetic demo dataset (one `.notes.json` + samples directory per preset) to exercise the
pipeline without real recordings:

```bash
optisample synth demo_out --sample-rate 22050
optisample optimize demo_out/piano.notes.json --budget-kb 96 --no-render
```
