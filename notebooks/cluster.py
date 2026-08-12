import marimo

__generated_with = "0.23.14"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    root = Path.cwd()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import marimo as mo

    from notebooks.utils import clusters, context, layouts, panels, scatter, viz

    return Path, clusters, context, layouts, mo, panels, root, scatter, viz


@app.cell
def _():
    from optisample.cluster.corpus import describe_corpus
    from optisample.cluster.partition import hierarchy, partition, sweep
    from optisample.cluster.representative import grouping
    from optisample.cluster.selection import selection, write_selection
    from optisample.cluster.space import pairwise_distances
    from optisample.cluster.stages import (
        StageSettings,
        available_instruments,
        available_stages,
        stage_dataset,
        stage_recordings,
    )
    from optisample.config.cluster import (
        DescriptorConfig,
        FrequencyBasis,
        LinkageMethod,
        PartitionAlgorithm,
        PartitionConfig,
        Representative,
        SpaceConfig,
    )
    from optisample.progress import NO_PROGRESS

    return (
        DescriptorConfig,
        FrequencyBasis,
        LinkageMethod,
        NO_PROGRESS,
        PartitionAlgorithm,
        PartitionConfig,
        Representative,
        SpaceConfig,
        StageSettings,
        available_instruments,
        available_stages,
        describe_corpus,
        grouping,
        hierarchy,
        pairwise_distances,
        partition,
        selection,
        stage_dataset,
        stage_recordings,
        sweep,
        write_selection,
    )


@app.cell
def _(context):
    notebook = context.notebook_context()
    return (notebook,)


@app.cell
def _(mo):
    mo.md("""
        # OptiSample — sample space

        Every recording of one pipeline stage placed as a point in a space built from **what it sounds
        like**, cut into groups, and each group stood for by a real take you can play.

        Two things are held out of the geometry on purpose. **Level**: every reading is taken past its own
        frame's mean, so a note at v020 and the same note at v100 differ by their timbre alone. **Length**:
        time is anchored to each recording's own decline — anchor *k* is the moment that note had fallen
        *k* dB below its own peak — so a two-second take and an eight-second take of one sound are read at
        the same points.

        | block | reads |
        |---|---|
        | `onset` | the attack timbre, up to the frame the recording settles at |
        | `sustain` | the timbre held at each fall depth of its own decline |
        | `movement` | how far that timbre travels while the note rings |
        | `envelope` | the contour — how long it took to reach each depth, and how straight it falls |

        Each block is scaled to a mean pairwise squared distance of one and then weighed, so the weights
        below are dimensionless and plain distance in the space is the weighted distance across the blocks.
        """)
    return


@app.cell
def _(mo, root):
    run_root = mo.ui.text(
        value=str(root / "artifacts"), label="run root — the stage directories sit here", full_width=True
    )
    mo.vstack([mo.md("## The run"), run_root])
    return (run_root,)


@app.cell
def _(Path, available_instruments, mo, run_root):
    instruments = available_instruments(Path(run_root.value))
    mo.stop(
        not instruments,
        mo.md("*No dataset under that run root yet — run `optisample pipeline` and point the root at its `--out`.*"),
    )
    instrument = mo.ui.dropdown(
        options=list(instruments), value=instruments[0], label="dataset — the instrument this run carried"
    )
    strategy = mo.ui.dropdown(["grouped", "ungrouped"], value="grouped", label="strategy — the allocated stage's plan")
    mo.hstack([instrument, strategy], justify="start", gap=2)
    return instrument, strategy


@app.cell
def _(NO_PROGRESS, Path, StageSettings, available_stages, instrument, notebook, run_root, strategy):
    listing = StageSettings(
        instrument_id=instrument.value,
        strategy=strategy.value,
        dedupe=notebook.config.reduce.dedupe,
        trim=notebook.config.reduce.trim,
        keep_tail=False,
        progress=NO_PROGRESS,
    )
    stages = available_stages(Path(run_root.value), listing)
    return listing, stages


@app.cell
def _(mo, notebook, stages):
    mo.stop(
        not stages,
        mo.md(
            "*No stage directory under that run root yet — run `optisample pipeline` and point the root at its `--out`.*"
        ),
    )
    stage = mo.ui.dropdown(options={found.value: found for found in stages}, value=stages[0].value, label="stage")
    keep_tail = mo.ui.checkbox(
        value=False, label="read past each note's release — deepens the falls the subset stage reaches"
    )
    workers = mo.ui.slider(1, 16, value=max(1, notebook.config.runtime.workers), label="workers", show_value=True)
    mo.vstack([mo.md("## The stage"), mo.hstack([stage, keep_tail, workers], justify="start", gap=2)])
    return keep_tail, stage, workers


@app.cell
def _(FrequencyBasis, mo, notebook):
    _read = notebook.config.cluster.descriptor
    frequency_basis = mo.ui.dropdown(
        options={basis.value: basis for basis in FrequencyBasis},
        value=_read.frequency_basis.value,
        label="frequency axis — the note's own partials, or the mel bands the optimizer scores with",
    )
    anchor_depths = mo.ui.multiselect(
        options={f"-{depth:g} dB": depth for depth in (0.0, 3.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 48.0, 60.0)},
        value=[f"-{depth:g} dB" for depth in _read.anchor_depths_db],
        label="fall depths",
    )
    harmonics = mo.ui.slider(4, 32, value=_read.harmonics, label="harmonics", show_value=True)
    cepstral_coefficients = mo.ui.slider(
        0,
        24,
        value=_read.cepstral_coefficients,
        label="cepstral coefficients — 0 reads the bands as they stand",
        show_value=True,
    )
    anchor_span_s = mo.ui.slider(
        0.02, 0.30, step=0.01, value=_read.anchor_span_s, label="anchor span (s)", show_value=True
    )
    mo.vstack(
        [
            mo.md("## How a recording is read"),
            frequency_basis,
            anchor_depths,
            mo.hstack([harmonics, cepstral_coefficients, anchor_span_s], justify="start", gap=2),
        ]
    )
    return anchor_depths, anchor_span_s, cepstral_coefficients, frequency_basis, harmonics


@app.cell
def _(DescriptorConfig, anchor_depths, anchor_span_s, cepstral_coefficients, frequency_basis, harmonics, mo, notebook):
    mo.stop(not anchor_depths.value, mo.md("*Pick at least one fall depth — a recording is read at one at the least.*"))
    reading = DescriptorConfig.model_validate(
        {
            **notebook.config.cluster.descriptor.model_dump(),
            "frequency_basis": frequency_basis.value,
            "anchor_depths_db": tuple(sorted(anchor_depths.value)),
            "anchor_span_s": float(anchor_span_s.value),
            "harmonics": int(harmonics.value),
            "cepstral_coefficients": int(cepstral_coefficients.value),
        }
    )
    return (reading,)


@app.cell
def _(
    NO_PROGRESS,
    Path,
    StageSettings,
    describe_corpus,
    instrument,
    keep_tail,
    mo,
    notebook,
    reading,
    run_root,
    stage,
    stage_recordings,
    strategy,
    workers,
):
    settings = StageSettings(
        instrument_id=instrument.value,
        strategy=strategy.value,
        dedupe=notebook.config.reduce.dedupe,
        trim=notebook.config.reduce.trim,
        keep_tail=bool(keep_tail.value),
        progress=NO_PROGRESS,
    )
    with mo.status.spinner(title=f"reading {stage.value}: decoding recordings, then reading each into its blocks..."):
        corpus = stage_recordings(Path(run_root.value), stage.value, settings)
        described = describe_corpus(
            corpus,
            features=notebook.config.loop.features,
            config=reading,
            workers=int(workers.value),
            progress=NO_PROGRESS,
        )

    mo.md(
        f"**{corpus.instrument_id}** at `{stage.value}` — **{described.size}** recordings carrying "
        f"**{described.weights.sum():.1f}s** of playing time, read at "
        f"`{reading.frequency_basis.value}` over {len(reading.anchor_depths_db)} fall depths."
    )
    return corpus, described, settings


@app.cell
def _(mo, notebook):
    _weighing = notebook.config.cluster.space
    onset_weight = mo.ui.slider(0.0, 3.0, step=0.05, value=_weighing.onset_weight, label="onset", show_value=True)
    sustain_weight = mo.ui.slider(0.0, 3.0, step=0.05, value=_weighing.sustain_weight, label="sustain", show_value=True)
    movement_weight = mo.ui.slider(
        0.0, 3.0, step=0.05, value=_weighing.movement_weight, label="movement", show_value=True
    )
    envelope_weight = mo.ui.slider(
        0.0, 3.0, step=0.05, value=_weighing.envelope_weight, label="envelope", show_value=True
    )
    min_reached_share = mo.ui.slider(
        0.05,
        1.0,
        step=0.05,
        value=_weighing.min_reached_share,
        label="depth kept where this share of the corpus reaches it",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md("## What the space weighs — a block at 1.0 has the same say as any other at 1.0"),
            mo.hstack([onset_weight, sustain_weight, movement_weight, envelope_weight], justify="start", gap=2),
            min_reached_share,
        ]
    )
    return envelope_weight, min_reached_share, movement_weight, onset_weight, sustain_weight


@app.cell
def _(
    SpaceConfig,
    described,
    envelope_weight,
    min_reached_share,
    movement_weight,
    onset_weight,
    pairwise_distances,
    sustain_weight,
):
    weighing = SpaceConfig(
        onset_weight=float(onset_weight.value),
        sustain_weight=float(sustain_weight.value),
        movement_weight=float(movement_weight.value),
        envelope_weight=float(envelope_weight.value),
        min_reached_share=float(min_reached_share.value),
    )
    space = described.space(weighing)
    distances = pairwise_distances(space.coordinates)
    return distances, space, weighing


@app.cell
def _(LinkageMethod, PartitionAlgorithm, Representative, described, mo, notebook):
    _cutting = notebook.config.cluster.partition
    _ceiling = max(3, min(_cutting.max_groups * 2, described.size - 1))
    algorithm = mo.ui.dropdown(
        options={rule.value: rule for rule in PartitionAlgorithm}, value=_cutting.algorithm.value, label="algorithm"
    )
    linkage_method = mo.ui.dropdown(
        options={method.value: method for method in LinkageMethod}, value=_cutting.linkage.value, label="linkage"
    )
    groups_wanted = mo.ui.slider(2, _ceiling, value=min(_cutting.groups, _ceiling), label="groups", show_value=True)
    sweep_to = mo.ui.slider(
        2, _ceiling, value=min(_cutting.max_groups, _ceiling), label="sweep climbs to", show_value=True
    )
    representative = mo.ui.dropdown(
        options={rule.value: rule for rule in Representative},
        value=_cutting.representative.value,
        label="stands for its group",
    )
    mo.vstack(
        [
            mo.md("## How the space is cut"),
            mo.hstack([algorithm, linkage_method, groups_wanted, sweep_to, representative], justify="start", gap=2),
        ]
    )
    return algorithm, groups_wanted, linkage_method, representative, sweep_to


@app.cell
def _(
    PartitionConfig,
    algorithm,
    clusters,
    described,
    distances,
    grouping,
    groups_wanted,
    linkage_method,
    partition,
    representative,
    space,
    sweep,
    sweep_to,
):
    cutting = PartitionConfig(
        algorithm=algorithm.value,
        linkage=linkage_method.value,
        groups=int(groups_wanted.value),
        max_groups=max(int(groups_wanted.value), int(sweep_to.value)),
        representative=representative.value,
    )
    climbed = sweep(space.coordinates, cutting)
    cut_now = partition(space.coordinates, groups=cutting.groups, config=cutting)
    groups = grouping(space.coordinates, cut_now.labels, described.weights)
    points = clusters.point_rows(described, groups, distances, rule=cutting.representative)
    representatives = [group.representative(cutting.representative) for group in groups]
    return climbed, cut_now, cutting, groups, points, representatives


@app.cell
def _(clusters, cut_now, mo, panels, space):
    mo.vstack(
        [
            mo.md(
                f"**{space.samples}** recordings placed by **{space.dimensions}** coordinates over "
                f"**{int(space.depths.sum())}** of the fall depths read — cut into **{cut_now.groups}** groups, "
                f"separation **{cut_now.silhouette:+.3f}**."
            ),
            mo.md("**What each block laid down** — `share` is how much of the space's spread it accounts for:"),
            panels.table(clusters.block_rows(space)),
        ]
    )
    return


@app.cell
def _(mo):
    get_examined, set_examined = mo.state(0)
    return get_examined, set_examined


@app.cell
def _(layouts, mo):
    _offered = layouts.available_layouts()
    layout = mo.ui.dropdown(options={found.value: found for found in _offered}, value=_offered[0].value, label="layout")
    components = mo.ui.dropdown(options={"2D": 2, "3D": 3}, value="2D", label="components")
    colour_by = mo.ui.dropdown(
        options=["group", "note", "role", "pitch", "velocity", "dur_s", "playing_s", "rate", "depths", "to_medoid"],
        value="group",
        label="colour by",
    )
    play_on_click = mo.ui.checkbox(value=True, label="play the clicked recording, normalized")
    mo.vstack(
        [
            mo.md("## The space"),
            mo.hstack([layout, components, colour_by, play_on_click], justify="start", gap=2),
        ]
    )
    return colour_by, components, layout, play_on_click


@app.cell
def _(components, layout, layouts, mo, space):
    with mo.status.spinner(title=f"laying the space out by {layout.value}..."):
        placed = layouts.draw(space.coordinates, layout=layout.value, components=int(components.value), seed=0)

    return (placed,)


@app.cell
def _(colour_by, layout, mo, placed, points, representatives, scatter, set_examined, stage):
    space_plot = mo.ui.plotly(
        scatter.space_scatter(
            placed,
            points,
            colour_by=colour_by.value,
            representatives=representatives,
            title=f"{stage.value} — {layout.value}, coloured by {colour_by.value}",
        ),
        on_change=lambda clicked: set_examined(lambda standing: scatter.picked(clicked or [], standing)),
    )
    _said = [
        mo.md(
            "Hover a point for everything it was read for; click one to hear it and to examine it below. The "
            "legend isolates a group, and the ringed diamonds are the takes standing for theirs, each ringed "
            "in its own group's colour."
        )
    ]
    if not layout.value.preserves_distance:
        _said.append(
            mo.md(
                "This layout places each recording beside the company it keeps, so a group reads as a cluster "
                "while the room between clusters follows the neighbourhoods. Every number reported below is "
                "the space's own."
            )
        )

    mo.vstack([*_said, space_plot])
    return


@app.cell
def _(described, examined, mo, panels, play_on_click, points):
    mo.stop(
        not play_on_click.value,
        mo.md(
            "*A click sends its recording to the examine panel below; tick **play the clicked recording** to hear it here as well.*"
        ),
    )
    _heard = described.recordings[examined]
    panels.player(
        _heard.signal,
        _heard.sample_rate,
        label=f"{points[examined]['group']} · {_heard.label} — {_heard.note} at velocity {_heard.key.velocity}",
        normalize=True,
        autoplay=True,
    )
    return


@app.cell
def _(colour_by, mo, points, representatives, scatter, set_examined, stage):
    key_plot = mo.ui.plotly(
        scatter.key_scatter(
            points,
            colour_by=colour_by.value,
            representatives=representatives,
            title=f"{stage.value} — the keys the corpus covers, coloured by {colour_by.value}",
        ),
        on_change=lambda clicked: set_examined(lambda standing: scatter.picked(clicked or [], standing)),
    )
    mo.vstack(
        [
            mo.md(
                "**Pitch against velocity** — the corpus as the keyboard holds it. This says which keys the "
                "stage kept a recording of and how a group sits across them; takes sharing a key stand on one "
                "point, and a click picks one out just as the space does."
            ),
            key_plot,
        ]
    )
    return


@app.cell
def _(PartitionAlgorithm, climbed, clusters, cut_now, cutting, hierarchy, mo, panels, scatter, space):
    _swept = clusters.sweep_rows(climbed, chosen=cut_now.groups)
    _best = max(climbed, key=lambda found: found.silhouette)
    _drawn = [
        mo.md(
            f"## How many groups — this corpus tells apart most cleanly at **{_best.groups}** "
            f"({_best.silhouette:+.3f})"
        ),
        mo.ui.plotly(scatter.sweep_curve(_swept, title="separation against the number of groups")),
    ]
    if cutting.algorithm is PartitionAlgorithm.HIERARCHICAL:
        _drawn.append(
            mo.ui.plotly(
                scatter.dendrogram_figure(
                    hierarchy(space.coordinates, cutting.linkage),
                    groups=cut_now.groups,
                    leaves=min(30, space.samples),
                    title="the top of the tree, and the height the cut reads it at",
                )
            )
        )

    mo.vstack([*_drawn, panels.table(_swept, page_size=8)])
    return


@app.cell
def _(clusters, cutting, described, groups, mo, panels, space):
    mo.vstack(
        [
            mo.md("## Groups — what each one gathered, and the take standing for it"),
            panels.table(clusters.group_rows(described, groups, rule=cutting.representative)),
            mo.md("**How tightly each group holds together, block by block** — where the grouping came from:"),
            panels.table(clusters.group_block_rows(space, groups), page_size=12),
        ]
    )
    return


@app.cell
def _(clusters, groups, mo):
    _options = {clusters.group_name(group.label): index for index, group in enumerate(groups)}
    group_pick = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="group")
    mo.vstack([mo.md("## Members — ordered by how far they stand from their medoid"), group_pick])
    return (group_pick,)


@app.cell
def _(clusters, group_pick, groups, panels, points):
    panels.table(clusters.member_rows(points, groups[group_pick.value]), page_size=12)
    return


@app.cell
def _(mo, points, set_examined):
    _options = {str(row["sample"]): index for index, row in enumerate(points)}
    examined_pick = mo.ui.dropdown(
        options=_options,
        value=next(iter(_options)),
        label="recording",
        on_change=lambda place: set_examined(int(place)),
    )
    preview_normalize = mo.ui.checkbox(
        value=False, label="peak-normalize players (audible, but level differences vanish)"
    )
    mo.vstack(
        [
            mo.md("## Examine and play — a click on either picture picks the recording, or name one here"),
            mo.hstack([examined_pick, preview_normalize], justify="start", gap=2),
        ]
    )
    return (preview_normalize,)


@app.cell
def _(described, get_examined):
    examined = min(get_examined(), described.size - 1)
    return (examined,)


@app.cell
def _(
    clusters,
    cutting,
    described,
    distances,
    examined,
    groups,
    mo,
    notebook,
    panels,
    preview_normalize,
    reading,
    space,
    viz,
):
    _recording = described.recordings[examined]
    _descriptor = described.descriptors[examined]
    mo.vstack(
        [
            mo.md(
                f"### `{_recording.label}` — {_recording.note} at velocity {_recording.key.velocity}, "
                f"{_recording.duration_s:.2f}s at {_recording.sample_rate} Hz, carrying "
                f"{_recording.weight:.2f}s of playing time · `{_recording.file}`"
            ),
            panels.player(
                _recording.signal,
                _recording.sample_rate,
                label="the recording as it was clustered",
                normalize=bool(preview_normalize.value),
                autoplay=False,
            ),
            panels.waveform(_recording.signal, _recording.sample_rate, title=_recording.label),
            panels.spectrogram(
                _recording.signal, _recording.sample_rate, style=notebook.spectrogram, title=_recording.label
            ),
            panels.image(
                viz.decline_figure(
                    _recording.signal,
                    _recording.sample_rate,
                    nodes=reading.envelope_nodes,
                    title="the level it holds, against the decline and the curve fitted to it",
                )
            ),
            panels.image(
                viz.profile_figure(
                    _descriptor.sustain,
                    depths_db=reading.anchor_depths_db,
                    reached=_descriptor.reached,
                    title=f"what it sounded like at each depth it reached ({reading.frequency_basis.value})",
                )
            ),
            mo.md("**Its own readings:**"),
            panels.table(clusters.descriptor_rows(_descriptor, _recording)),
            mo.md("**Where it stands on its own decline** — the depths it reached, and the ones the space reads:"),
            panels.table(clusters.anchor_rows(_descriptor, reading.anchor_depths_db, space)),
            mo.md("**How far it stands from every group** — the company it nearly kept:"),
            panels.table(clusters.reach_rows(examined, described, groups, distances, rule=cutting.representative)),
        ]
    )
    return


@app.cell
def _(described, mo, panels, points, preview_normalize, representatives):
    mo.vstack(
        [
            mo.md("## Representatives — every group's take, auditioned in one row"),
            mo.hstack(
                [
                    panels.player(
                        described.recordings[place].signal,
                        described.recordings[place].sample_rate,
                        label=f"{points[place]['group']} — {described.recordings[place].label}",
                        normalize=bool(preview_normalize.value),
                        autoplay=False,
                    )
                    for place in representatives
                ],
                justify="start",
                wrap=True,
                gap=1,
            ),
        ]
    )
    return


@app.cell
def _(Path, mo, run_root, stage):
    selection_dir = mo.ui.text(
        value=str(Path(run_root.value) / "selection" / stage.value.value),
        label="write to",
        full_width=True,
    )
    write_now = mo.ui.run_button(label="write the selection as a dataset")
    _offered = (
        [selection_dir, write_now]
        if stage.value.is_dataset
        else [
            mo.md(
                "*The allocated stage holds what one plan stored, which a run reads back through that "
                "plan — pick a dataset stage to write a selection out of.*"
            )
        ]
    )
    mo.vstack(
        [
            mo.md("""
                ## Feed the selection back

                The takes standing for their groups, written out as a NoteExtractor dataset: every note
                those recordings answer for, carried over exactly as this stage states it, beside a copy of
                each recording. That makes a set chosen by **what it sounds like** something `loop`,
                `reduce` and `optimize` read the way they read a subset.
                """),
            *_offered,
        ]
    )
    return selection_dir, write_now


@app.cell
def _(
    Path,
    clusters,
    cutting,
    described,
    groups,
    instrument,
    mo,
    panels,
    run_root,
    selection,
    selection_dir,
    stage,
    stage_dataset,
    write_now,
    write_selection,
):
    mo.stop(
        not write_now.value,
        mo.md("*Nothing written yet — press the button to write the selection above out as a dataset.*"),
    )
    _chosen = selection(described.corpus, groups, rule=cutting.representative)
    with mo.status.spinner(title=f"writing {_chosen.size} recordings to {selection_dir.value}..."):
        _written = write_selection(
            stage_dataset(Path(run_root.value), stage.value, instrument.value),
            _chosen,
            Path(selection_dir.value),
        )

    _dataset = _written.dataset
    mo.vstack(
        [
            mo.md(
                f"Wrote **{_dataset.recordings}** recordings and the **{_dataset.kept_notes}** of this "
                f"stage's **{_dataset.source_notes}** notes they answer for, spanning pitches "
                f"**{_dataset.pitches[0]}–{_dataset.pitches[1]}** and velocities "
                f"**{_dataset.velocities[0]}–{_dataset.velocities[1]}**. That written material is what a "
                f"later stage weighs its allocation by; back in the space these same takes stand for "
                f"**{_written.selection.covered}** recordings carrying "
                f"**{_written.selection.playing_s:.1f}s** of playing time, which is the reach the table "
                f"below reads each of them by."
            ),
            mo.md(
                f"Run the rest of the pipeline over it:\n```\nuv run optisample pipeline {_dataset.source.path} "
                f"--budget-kb 512 --out artifacts-selection\n```"
            ),
            panels.table(clusters.pick_rows(_written.selection)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
