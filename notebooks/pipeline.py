import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    from typing import get_args

    root = Path.cwd()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import marimo as mo

    from notebooks.utils import context, panels, reports, runs, viz

    return Path, context, get_args, mo, panels, reports, root, runs, viz


@app.cell
def _(get_args):
    from optisample.artifacts.paths import plan_paths
    from optisample.config.reduce import DedupeKey
    from optisample.config.render import Interpolation
    from optisample.config.tracker import TrackerFormat
    from optisample.optimize.operating_points import sweep_rates

    interpolations = list(get_args(Interpolation))
    return DedupeKey, TrackerFormat, interpolations, plan_paths, sweep_rates


@app.cell
def _(context):
    notebook = context.notebook_context()
    return (notebook,)


@app.cell
def _(mo):
    mo.md("""
        # OptiSample — pipeline console

        Every CLI flag as a control, every stage's output as an explorer. A run here shells out to
        `optisample <command>` exactly as a terminal would, and the invocation behind each button is
        printed above it, so anything you find is reproducible from a shell.

        | stage | writes |
        |---|---|
        | `subset` | the share of a dataset that still spans its pitch and velocity ranges |
        | `loop` | the loop each recording is stored around, and every settled loop played out as audio |
        | `reduce` | the survivors, the format each key stores at, and every swept encoding as audio |
        | `optimize` | the byte-budgeted plan, the module, and per-note A/B against the recording |
        """)
    return


@app.cell
def _(Path, mo, root, runs):
    _found = runs.discover(root / "subset") or runs.discover(root / "reduced") or runs.discover(root / "demo_out")
    _default = _found or runs.Dataset(Path("demo_out/piano.notes.json"), Path("demo_out/piano"))

    source_notes = mo.ui.text(value=str(_default.notes_json), label="source .notes.json", full_width=True)
    source_samples = mo.ui.text(value=str(_default.samples_dir), label="source samples dir", full_width=True)
    out_root = mo.ui.text(
        value=str(root), label="run root — subset/ looped/ reduced/ artifacts/ land here", full_width=True
    )
    instrument = mo.ui.text(value="Piano", label="instrument id")
    mo.vstack([mo.md("## Dataset"), source_notes, source_samples, out_root, instrument])
    return instrument, out_root, source_notes, source_samples


@app.cell
def _(DedupeKey, TrackerFormat, interpolations, mo, notebook, sweep_rates):
    _config = notebook.config
    _rates = [str(rate) for rate in sweep_rates(_config.optimize.sweep, 48_000)]

    fraction = mo.ui.slider(0.05, 1.0, step=0.05, value=0.1, label="subset fraction", show_value=True)
    budget_kb = mo.ui.number(16.0, 8192.0, step=16.0, value=512.0, label="budget (KiB)")
    workers = mo.ui.slider(1, 16, value=max(1, _config.runtime.workers), label="workers", show_value=True)
    seed = mo.ui.number(0, 9999, step=1, value=0, label="dither seed")

    tracker_format = mo.ui.dropdown(
        [kind.value for kind in TrackerFormat], value=_config.export.tracker.format.value, label="format"
    )
    strategy = mo.ui.dropdown(["both", "ungrouped", "grouped"], value="ungrouped", label="strategy")
    interpolation = mo.ui.dropdown(interpolations, value=_config.export.render.interpolation, label="interpolation")

    dedupe_key = mo.ui.dropdown(
        [key.value for key in DedupeKey], value=_config.reduce.dedupe.key.value, label="dedupe key"
    )
    content_floor_db = mo.ui.slider(
        20.0,
        100.0,
        step=5.0,
        value=_config.reduce.bandwidth.content_floor_db,
        label="content floor (dB) — how much band a stored rate carries",
        show_value=True,
    )

    rates = mo.ui.multiselect(_rates, value=[], label="rate ladder — empty takes the config's")
    depth = mo.ui.dropdown(["16", "8"], value=str(_config.optimize.sweep.depth), label="bit depth")
    loop = mo.ui.checkbox(value=True, label="allow looping")
    render = mo.ui.checkbox(value=True, label="openmpt123 ground-truth render")

    mo.vstack(
        [
            mo.md("## Flags"),
            mo.hstack([fraction, budget_kb, workers, seed], justify="start", gap=2),
            mo.hstack([tracker_format, strategy, interpolation], justify="start", gap=2),
            mo.hstack([dedupe_key, content_floor_db], justify="start", gap=2),
            mo.hstack([rates, depth], justify="start", gap=2),
            mo.hstack([loop, render], justify="start", gap=2),
        ]
    )
    return (
        budget_kb,
        content_floor_db,
        dedupe_key,
        depth,
        fraction,
        interpolation,
        loop,
        rates,
        render,
        seed,
        strategy,
        tracker_format,
        workers,
    )


@app.cell
def _(
    Path,
    budget_kb,
    content_floor_db,
    dedupe_key,
    depth,
    fraction,
    instrument,
    interpolation,
    loop,
    out_root,
    rates,
    render,
    runs,
    seed,
    source_notes,
    source_samples,
    strategy,
    tracker_format,
    workers,
):
    fields = runs.Fields(
        source=runs.Dataset(Path(source_notes.value), Path(source_samples.value)),
        instrument_id=instrument.value,
        out_root=Path(out_root.value),
        budget_kb=float(budget_kb.value),
        fraction=float(fraction.value),
        tracker_format=tracker_format.value,
        strategy=strategy.value,
        interpolation=interpolation.value,
        dedupe_key=dedupe_key.value,
        content_floor_db=float(content_floor_db.value),
        rates=tuple(int(rate) for rate in rates.value),
        depth=int(depth.value),
        loop=bool(loop.value),
        workers=int(workers.value),
        seed=int(seed.value),
        render=bool(render.value),
    )
    loop_input = runs.looping_source(fields)
    reduce_input = runs.reduction_source(fields)
    optimize_input = runs.allocation_source(fields)
    return fields, loop_input, optimize_input, reduce_input


@app.cell
def _(fields, loop_input, mo, optimize_input, reduce_input, runs):
    def shell_line(command):
        return " ".join(["optisample", *command[3:]])

    mo.md(f"""
        ### The commands these controls stand for

        ```bash
        {shell_line(runs.subset_command(fields))}

        {shell_line(runs.loop_command(fields, loop_input))}

        {shell_line(runs.reduce_command(fields, reduce_input))}

        {shell_line(runs.optimize_command(fields, optimize_input))}
        ```

        `loop` reads `{loop_input.notes_json}` · `reduce` reads `{reduce_input.notes_json}` ·
        `optimize` reads `{optimize_input.notes_json}` — each stage prefers the dataset the one before it
        left, so a settled loop reaches the allocation over the recordings its frames index into.
        """)
    return (shell_line,)


@app.cell
def _(mo):
    def stage_view(outcome, stage):
        if outcome is None:
            return mo.md(f"*`{stage}` has not been run from here — the explorers read whatever is already on disk.*")

        status = "finished" if outcome.ok else f"**FAILED (exit {outcome.returncode})**"
        return mo.vstack(
            [
                mo.md(f"`{stage}` {status} in **{outcome.elapsed_s:.1f}s** · transcript `{outcome.log}`"),
                mo.plain_text(outcome.output or "(no output)"),
            ]
        )

    return (stage_view,)


@app.cell
def _(mo):
    subset_button = mo.ui.run_button(label="Run subset")
    loop_button = mo.ui.run_button(label="Run loop")
    reduce_button = mo.ui.run_button(label="Run reduce")
    optimize_button = mo.ui.run_button(label="Run optimize")
    mo.vstack(
        [
            mo.md("### Run — a button blocks its own cell until the stage finishes"),
            mo.hstack([subset_button, loop_button, reduce_button, optimize_button], justify="start", gap=1),
        ]
    )
    return loop_button, optimize_button, reduce_button, subset_button


@app.cell
def _(fields, mo, runs, stage_view, subset_button):
    subset_outcome = None
    if subset_button.value:
        with mo.status.spinner(title="carving the subset..."):
            subset_outcome = runs.run_stage("subset", runs.subset_command(fields), fields)

    stage_view(subset_outcome, "subset")
    return (subset_outcome,)


@app.cell
def _(fields, loop_button, loop_input, mo, runs, stage_view):
    loop_outcome = None
    if loop_button.value:
        with mo.status.spinner(title="settling loops: measuring seams, climbing the ladder, auditioning..."):
            loop_outcome = runs.run_stage("loop", runs.loop_command(fields, loop_input), fields)

    stage_view(loop_outcome, "loop")
    return (loop_outcome,)


@app.cell
def _(fields, mo, reduce_button, reduce_input, runs, stage_view):
    reduce_outcome = None
    if reduce_button.value:
        with mo.status.spinner(title="reducing: probing, deduplicating, narrowing, auditioning..."):
            reduce_outcome = runs.run_stage("reduce", runs.reduce_command(fields, reduce_input), fields)

    stage_view(reduce_outcome, "reduce")
    return (reduce_outcome,)


@app.cell
def _(fields, mo, optimize_button, optimize_input, runs, stage_view):
    optimize_outcome = None
    if optimize_button.value:
        with mo.status.spinner(title="allocating: sweeping, solving, exporting, scoring..."):
            optimize_outcome = runs.run_stage("optimize", runs.optimize_command(fields, optimize_input), fields)

    stage_view(optimize_outcome, "optimize")
    return (optimize_outcome,)


@app.cell
def _(mo):
    preview_normalize = mo.ui.checkbox(
        value=False, label="peak-normalize previews (audible, but level differences vanish)"
    )
    mo.vstack([mo.md("---"), preview_normalize])
    return (preview_normalize,)


@app.cell
def _(mo):
    mo.md("""## Looping — the loop each recording is stored around""")
    return


@app.cell
def _(fields, loop_outcome, mo, panels, reports):
    loop_outcome
    looped_root = fields.looped_root
    loops_doc = None
    if reports.has_loops(looped_root, fields.instrument_id):
        loops_doc = reports.read_loop_document(looped_root, fields.instrument_id)

    mo.stop(loops_doc is None, mo.md(f"*No `loops.json` under `{looped_root}` yet — run **loop**.*"))
    _settled = reports.loop_rows(loops_doc)
    mo.vstack(
        [
            mo.md(
                f"**{loops_doc.instrument_id}** settled at **{loops_doc.sample_rate} Hz** — "
                f"{len(_settled)} of {len(loops_doc.recordings)} recordings stored around a loop."
            ),
            mo.md("**Settled loops** — what each one repeats, and what measuring it said:"),
            panels.table(_settled, page_size=10),
            mo.md("**Stored over the span they play** — the recordings no loop was settled for, and what was tried:"),
            panels.table(reports.unlooped_rows(loops_doc), page_size=10),
            mo.md("**Climbed past** — every cheaper candidate, and the gate it fell outside:"),
            panels.table(reports.rejected_loop_rows(loops_doc), page_size=10),
        ]
    )
    return loops_doc, looped_root


@app.cell
def _(fields, looped_root, mo, reports):
    _keys = reports.loop_audition_keys(looped_root, fields.instrument_id)
    mo.stop(not _keys, mo.md("*No loop auditions were written.*"))
    loop_audition_key = mo.ui.dropdown(options=_keys, value=_keys[0], label="audition recording")
    mo.vstack(
        [mo.md("### Auditions — each loop played out against the recording it was taken from"), loop_audition_key]
    )
    return (loop_audition_key,)


@app.cell
def _(fields, loop_audition_key, looped_root, mo, panels, preview_normalize, reports):
    mo.hstack(
        [
            panels.file_player(clip.path, label=clip.label, normalize=preview_normalize.value, autoplay=False)
            for clip in reports.loop_auditions(looped_root, fields.instrument_id, loop_audition_key.value)
        ],
        justify="start",
        wrap=True,
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md("""---\n## Reduction — what the pre-optimization stage decided""")
    return


@app.cell
def _(fields, mo, panels, reduce_outcome, reports):
    reduce_outcome
    reduced_root = fields.reduced_root
    reduced_doc = None
    if reports.has_reduction(reduced_root, fields.instrument_id):
        reduced_doc = reports.read_reduced(reduced_root, fields.instrument_id)

    mo.stop(reduced_doc is None, mo.md(f"*No `reduction.json` under `{reduced_root}` yet — run **reduce**.*"))
    _recordings = reports.recording_rows(reduced_doc.reduction)
    _shortfalls = [row for row in _recordings if not row["covers"]]
    mo.vstack(
        [
            mo.md(
                f"**{reduced_doc.instrument_id}** reduced under `{reduced_doc.dedupe_key}` at "
                f"**{reduced_doc.sample_rate} Hz** — {len(reduced_doc.samples)} survivors written."
            ),
            panels.table(reports.reduction_rows(reduced_doc.reduction)),
            mo.md(
                f"**Kept recordings** — {len(_shortfalls)} of {len(_recordings)} hold less than their pitch asks "
                "of them, which the objective absorbs by scoring against a shorter reference."
            ),
            panels.table(_recordings, page_size=10),
            mo.md("**Stored format** — the band each pitch asked for, and the format it is stored at:"),
            panels.table(reports.stored_format_rows(reduced_doc.reduction), page_size=10),
            mo.md("**Survivors** — the dataset an allocation picks up from:"),
            panels.table(reports.survivor_rows(reduced_doc), page_size=10),
        ]
    )
    return reduced_doc, reduced_root


@app.cell
def _(fields, mo, reduced_root, reports):
    _pitches = reports.audition_pitches(reduced_root, fields.instrument_id)
    mo.stop(not _pitches, mo.md("*No auditions were written.*"))
    audition_pitch = mo.ui.dropdown(options=_pitches, value=_pitches[0], label="audition pitch")
    mo.vstack(
        [mo.md("### Auditions — every swept encoding, made audible before the sweep is paid for"), audition_pitch]
    )
    return (audition_pitch,)


@app.cell
def _(audition_pitch, fields, mo, panels, preview_normalize, reduced_root, reports):
    mo.hstack(
        [
            panels.file_player(clip.path, label=clip.label, normalize=preview_normalize.value, autoplay=False)
            for clip in reports.auditions(reduced_root, fields.instrument_id, audition_pitch.value)
        ],
        justify="start",
        wrap=True,
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md("""---\n## Allocation — what the budget bought""")
    return


@app.cell
def _(fields, mo, optimize_outcome, reports):
    optimize_outcome
    _strategies = reports.available_strategies(fields.artifacts)
    mo.stop(not _strategies, mo.md(f"*No plan under `{fields.artifacts}` yet — run **optimize**.*"))
    plan_strategy = mo.ui.dropdown(options=_strategies, value=_strategies[0], label="strategy")
    plan_strategy
    return (plan_strategy,)


@app.cell
def _(fields, mo, panels, plan_paths, plan_strategy, reports, viz):
    paths = plan_paths(fields.artifacts, plan_strategy.value)
    plan = reports.read_plan(paths)
    items = reports.plan_item_rows(plan)
    mo.vstack(
        [
            panels.table(reports.budget_rows(plan)),
            mo.md("**Instruments** — what the plan is written as, one row each:"),
            panels.table(reports.instrument_rows(plan)),
            mo.md("**Per-item allocation** — the encoding each kept item spends its bytes on:"),
            panels.table(items, page_size=15),
            panels.image(viz.rd_scatter(items, title=f"{plan.strategy}: what the budget bought")),
        ]
    )
    return items, paths, plan


@app.cell
def _(mo, panels, paths, reports, viz):
    metrics = reports.read_metrics(paths)
    note_rows = reports.note_metric_rows(metrics)
    mo.vstack(
        [
            mo.md(
                f"**Per-note fidelity** — objective **{metrics.objective:.4f}**, "
                f"the plan states **{metrics.plan_objective:.4f}**. Lower is better."
            ),
            panels.table(note_rows, page_size=12),
            panels.image(viz.objective_bar(note_rows)),
        ]
    )
    return metrics, note_rows


@app.cell
def _(mo, paths, reports):
    _compared = reports.compared_notes(paths)
    mo.stop(not _compared, mo.md("*No A/B pairs were written.*"))
    _options = {note.label: note for note in _compared}
    compare_pitch = mo.ui.dropdown(options=_options, value=_compared[0].label, label="A/B pitch")
    mo.vstack([mo.md("### Reference vs. module — the very pair the objective scored"), compare_pitch])
    return (compare_pitch,)


@app.cell
def _(compare_pitch, metrics, mo, notebook, panels, paths, preview_normalize, reports):
    _reference_path, _render_path = reports.comparison(paths, compare_pitch.value)
    _style = notebook.spectrogram
    mo.vstack(
        [
            mo.hstack(
                [
                    panels.signal_panel(
                        _reference_path, label="reference", style=_style, normalize=preview_normalize.value
                    ),
                    panels.signal_panel(_render_path, label="module", style=_style, normalize=preview_normalize.value),
                ]
            ),
            mo.md("**Scored classes at this pitch** — every note the objective measured through this sample:"),
            panels.table(reports.event_rows(metrics, compare_pitch.value)),
        ]
    )
    return


@app.cell
def _(mo, paths, reports):
    module_render = reports.module_render(paths)
    mo.vstack(
        [
            mo.md("### The written module"),
            mo.md(
                f"Rendered to `{module_render}` ({module_render.stat().st_size / 1e6:.1f} MB) — open it in a player."
                if module_render
                else "*No ground-truth render: openmpt123 was absent, or the run was asked to skip it.*"
            ),
            mo.accordion({"report.txt": mo.plain_text(reports.report_text(paths))}),
        ]
    )
    return (module_render,)


if __name__ == "__main__":
    app.run()
