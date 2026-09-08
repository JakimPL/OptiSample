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
    from optisample.cluster.stages import ReadingSettings, available_sources, gathered_recordings
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
        ReadingSettings,
        Representative,
        SpaceConfig,
        available_sources,
        describe_corpus,
        gathered_recordings,
        grouping,
        hierarchy,
        pairwise_distances,
        partition,
        selection,
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

    Every recording of the sets you pick placed as a point in a space built from **what it sounds
    like**, cut into groups, and each group stood for by a real take you can play. A run leaves
    several sets behind — an instrument at two stages, or two instruments at one — and as many of
    them as you name are gathered into a single space.

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
    That standardizing and that scaling are taken across the whole gathered corpus, so several sets
    placed in one space are read under a single frame: each set's coordinates answer for the company
    it was read beside, and differ from what that same set would hold read on its own.
    """)
    return


@app.cell
def _(mo, root):
    _artifacts = root / "artifacts"
    browsing_from = _artifacts if _artifacts.is_dir() else root
    run_root = mo.ui.file_browser(
        initial_path=browsing_from,
        selection_mode="directory",
        multiple=False,
        label="run root — the stage directories sit here",
    )
    strategy = mo.ui.dropdown(["grouped", "ungrouped"], value="grouped", label="strategy — the allocated stage's plan")
    mo.vstack([mo.md("## The run"), run_root, strategy])
    return browsing_from, run_root, strategy


@app.cell
def _(NO_PROGRESS, ReadingSettings, notebook, strategy):
    listing = ReadingSettings(
        strategy=strategy.value,
        dedupe=notebook.config.reduce.dedupe,
        trim=notebook.config.reduce.trim,
        keep_tail=False,
        progress=NO_PROGRESS,
    )
    return (listing,)


@app.cell
def _(available_sources, browsing_from, listing, run_root):
    picked_root = run_root.path(0) or browsing_from
    offered_sets = available_sources(picked_root, listing)
    return offered_sets, picked_root


@app.cell
def _(mo, notebook, offered_sets):
    mo.stop(
        not offered_sets,
        mo.md("*No dataset under that run root yet — run `optisample pipeline` and point the root at its `--out`.*"),
    )
    sets = mo.ui.multiselect(
        options={source.label: source for source in offered_sets},
        value=[offered_sets[0].label],
        label="sets — every one named here is read into the same space",
    )
    keep_tail = mo.ui.checkbox(
        value=False, label="read past each note's release — deepens the falls the subset stage reaches"
    )
    workers = mo.ui.slider(1, 16, value=max(1, notebook.config.runtime.workers), label="workers", show_value=True)
    mo.vstack([mo.md("## The sets"), sets, mo.hstack([keep_tail, workers], justify="start", gap=2)])
    return keep_tail, sets, workers


@app.cell
def _(mo, offered_sets, sets):
    chosen_sets = tuple(source for source in offered_sets if source in sets.value)
    mo.stop(
        not chosen_sets,
        mo.md("*Name at least one set — a space is gathered from the recordings of one at the least.*"),
    )
    return (chosen_sets,)


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
    return (
        anchor_depths,
        anchor_span_s,
        cepstral_coefficients,
        frequency_basis,
        harmonics,
    )


@app.cell
def _(
    DescriptorConfig,
    anchor_depths,
    anchor_span_s,
    cepstral_coefficients,
    frequency_basis,
    harmonics,
    mo,
    notebook,
):
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
    ReadingSettings,
    chosen_sets,
    clusters,
    describe_corpus,
    gathered_recordings,
    keep_tail,
    mo,
    notebook,
    reading,
    strategy,
    workers,
):
    settings = ReadingSettings(
        strategy=strategy.value,
        dedupe=notebook.config.reduce.dedupe,
        trim=notebook.config.reduce.trim,
        keep_tail=bool(keep_tail.value),
        progress=NO_PROGRESS,
    )
    gathered = clusters.sources_label(chosen_sets)
    with mo.status.spinner(title=f"reading {gathered}: decoding recordings, then reading each into its blocks..."):
        corpus = gathered_recordings(chosen_sets, settings)
        described = describe_corpus(
            corpus,
            features=notebook.config.loop.features,
            config=reading,
            workers=int(workers.value),
            progress=NO_PROGRESS,
        )

    mo.stop(
        not described.size,
        mo.md("*Those sets left no recording to place — every take they hold stayed under the silence floor.*"),
    )
    mo.md(
        f"**{gathered}** — **{described.size}** recordings carrying "
        f"**{described.corpus.weights.sum():.1f}s** of playing time, read at "
        f"`{reading.frequency_basis.value}` over {len(reading.anchor_depths_db)} fall depths."
    )
    return described, gathered


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
    return (
        envelope_weight,
        min_reached_share,
        movement_weight,
        onset_weight,
        sustain_weight,
    )


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
    return distances, space


@app.cell
def _(
    LinkageMethod,
    PartitionAlgorithm,
    Representative,
    described,
    mo,
    notebook,
):
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
    min_duration_s = mo.ui.slider(
        0.0,
        2.0,
        step=0.05,
        value=_cutting.min_duration_s,
        label="a representative rings for at least (s)",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md("## How the space is cut"),
            mo.hstack([algorithm, linkage_method, groups_wanted, sweep_to], justify="start", gap=2),
            mo.hstack([representative, min_duration_s], justify="start", gap=2),
        ]
    )
    return (
        algorithm,
        groups_wanted,
        linkage_method,
        min_duration_s,
        representative,
        sweep_to,
    )


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
    min_duration_s,
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
        min_duration_s=float(min_duration_s.value),
    )
    climbed = sweep(space.coordinates, cutting)
    cut_now = partition(space.coordinates, groups=cutting.groups, config=cutting)
    groups = grouping(space.coordinates, cut_now.labels, described.readings, config=cutting)
    named = clusters.named_groups(described, groups)
    points = clusters.point_rows(described, named, distances)
    representatives = [group.representative for group in groups]
    return climbed, cut_now, cutting, groups, named, points, representatives


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
    layout = mo.ui.dropdown(
        options={found.value: found for found in _offered},
        value=_offered[0].value,
        label="layout — the space's own geometry, or the keys the corpus covers",
    )
    components = mo.ui.dropdown(options={"2D": 2, "3D": 3}, value="2D", label="components")
    color_by = mo.ui.dropdown(
        options=[
            "group",
            "set",
            "note",
            "role",
            "pitch",
            "velocity",
            "dur_s",
            "playing_s",
            "rate",
            "depths",
            "to_medoid",
        ],
        value="group",
        label="color by",
    )
    play_on_click = mo.ui.checkbox(value=True, label="play the clicked recording, normalized")
    mo.vstack(
        [
            mo.md("## The space — and the keyboard it was recorded from"),
            mo.hstack([layout, components, color_by, play_on_click], justify="start", gap=2),
        ]
    )
    return color_by, components, layout, play_on_click


@app.cell
def _(clusters, color_by, named, scatter):
    coloring = scatter.Coloring(column=color_by.value, groups=clusters.group_colors(named))
    return (coloring,)


@app.cell
def _(components, layout, layouts, mo, points, space):
    with mo.status.spinner(title=f"placing the recordings by {layout.value}..."):
        placement = layouts.place(
            space.coordinates, points, layout=layout.value, components=int(components.value), seed=0
        )
    return (placement,)


@app.cell
def _(
    color_by,
    coloring,
    gathered,
    layout,
    mo,
    placement,
    points,
    representatives,
    scatter,
    set_examined,
):
    _picture = scatter.field(
        placement,
        points,
        coloring=coloring,
        representatives=representatives,
        title=f"{gathered} — {layout.value}, colored by {color_by.value}",
    )
    space_plot = mo.ui.plotly(
        _picture.figure,
        on_change=lambda clicked: set_examined(lambda standing: scatter.picked(_picture, clicked or [], standing)),
    )
    _said = [
        mo.md(
            "Hover a point for everything it was read for; click one to hear it and to examine it below. The "
            "legend isolates a group, and the ringed diamonds are the takes standing for theirs, each ringed "
            "in its own group's color."
        )
    ]
    if not layout.value.reads_the_space:
        _said.append(
            mo.md(
                "This is the corpus as the keyboard holds it: the note across, the velocity it was struck at "
                "up, and in three dimensions how long the take rings. It says which keys the sets kept a "
                "recording of and how a group sits across them; takes sharing a key stand on one point, "
                "whichever set each of them came from."
            )
        )
    elif not layout.value.preserves_distance:
        _said.append(
            mo.md(
                "This layout places each recording beside the company it keeps, so a group reads as a cluster "
                "while the room between clusters follows the neighborhoods. Every number reported below is "
                "the space's own."
            )
        )

    if not placement.is_plane:
        _said.append(
            mo.md(
                "A box is turned and read by eye, and a click reaches Python from a plane — so a recording "
                "drawn in three dimensions is picked out by naming it in the examine panel below."
            )
        )

    mo.vstack([*_said, space_plot])
    return


@app.cell
def _(clusters, described, examined, mo, panels, play_on_click, points):
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
        label=(
            f"{points[examined]['group']} · {clusters.sample_name(_heard)} — "
            f"{_heard.note} at velocity {_heard.key.velocity}"
        ),
        normalize=True,
        autoplay=True,
    )
    return


@app.cell
def _(
    PartitionAlgorithm,
    climbed,
    clusters,
    cut_now,
    cutting,
    hierarchy,
    mo,
    panels,
    scatter,
    space,
):
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
def _(clusters, described, mo, named, panels, space):
    mo.vstack(
        [
            mo.md("## Groups — what each one gathered, and the take standing for it"),
            mo.md(
                "Each group goes by the key of the take standing for it, and is drawn in the color that "
                "key turns: the pitch sets the hue and the velocity fills it in, so the field carries the "
                "keyboard and one key is the same color in every picture and at every stage."
            ),
            panels.table(clusters.group_rows(described, named)),
            mo.md("**How tightly each group holds together, block by block** — where the grouping came from:"),
            panels.table(clusters.group_block_rows(space, named), page_size=12),
        ]
    )
    return


@app.cell
def _(mo, named):
    _options = {group.name: index for index, group in enumerate(named)}
    group_pick = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="group")
    mo.vstack([mo.md("## Members — ordered by how far they stand from their medoid"), group_pick])
    return (group_pick,)


@app.cell
def _(clusters, group_pick, named, panels, points):
    panels.table(clusters.member_rows(points, named[group_pick.value]), page_size=12)
    return


@app.cell
def _(mo, points, set_examined):
    _options = {f"{row['set']} · {row['sample']}": index for index, row in enumerate(points)}
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
            mo.md("## Examine and play — a click on the field picks the recording, or name one here"),
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
    described,
    distances,
    examined,
    mo,
    named,
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
                f"### `{clusters.sample_name(_recording)}` — {_recording.note} at velocity "
                f"{_recording.key.velocity}, {_recording.duration_s:.2f}s at {_recording.sample_rate} Hz, "
                f"carrying {_recording.weight:.2f}s of playing time · `{_recording.file}`"
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
            panels.table(clusters.reach_rows(examined, described, named, distances)),
        ]
    )
    return


@app.cell
def _(
    clusters,
    described,
    mo,
    panels,
    points,
    preview_normalize,
    representatives,
):
    mo.vstack(
        [
            mo.md("## Representatives — every group's take, auditioned in one row"),
            mo.hstack(
                [
                    panels.player(
                        described.recordings[place].signal,
                        described.recordings[place].sample_rate,
                        label=f"{points[place]['group']} — {clusters.sample_name(described.recordings[place])}",
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
def _(chosen_sets, mo, picked_root):
    selection_dir = mo.ui.text(value=str(picked_root / "selection"), label="write to", full_width=True)
    write_now = mo.ui.run_button(label="write the selection as a dataset")
    _offered = (
        [selection_dir, write_now]
        if all(source.stage.is_dataset for source in chosen_sets)
        else [
            mo.md(
                "*The allocated stage holds what one plan stored, which a run reads back through that "
                "plan — name dataset sets alone to write a selection out of.*"
            )
        ]
    )
    mo.vstack(
        [
            mo.md("""
                ## Feed the selection back

                The takes standing for their groups, written out as NoteExtractor datasets: every note
                those recordings answer for, carried over exactly as its own set states it, beside a copy
                of each recording. That makes a choice made by **what it sounds like** something `loop`,
                `reduce` and `optimize` read the way they read a subset. Each set the picks came from
                lands as a dataset of its own, under `<write to>/<instrument>/<stage>/`, so a selection
                gathered across sets feeds each of them back on its own terms.
                """),
            *_offered,
        ]
    )
    return selection_dir, write_now


@app.cell
def _(
    Path,
    clusters,
    described,
    groups,
    mo,
    named,
    panels,
    selection,
    selection_dir,
    write_now,
    write_selection,
):
    mo.stop(
        not write_now.value,
        mo.md("*Nothing written yet — press the button to write the selection above out as a dataset.*"),
    )
    _chosen = selection(described.corpus, groups)
    with mo.status.spinner(title=f"writing {_chosen.size} recordings to {selection_dir.value}..."):
        _written = write_selection(_chosen, Path(selection_dir.value))

    mo.vstack(
        [
            mo.md(
                f"Wrote **{_written.recordings}** recordings and the **{_written.kept_notes}** notes they "
                f"answer for, one dataset per set the picks came from. That written material is what a "
                f"later stage weighs its allocation by; back in the space these same takes stand for "
                f"**{_written.selection.covered}** recordings carrying "
                f"**{_written.selection.playing_s:.1f}s** of playing time, which is the reach the table "
                f"below reads each of them by."
            ),
            *[
                mo.md(
                    f"**{entry.source.label}** — **{entry.dataset.recordings}** recordings and the "
                    f"**{entry.dataset.kept_notes}** of that set's **{entry.dataset.source_notes}** notes "
                    f"they answer for, spanning pitches "
                    f"**{entry.dataset.pitches[0]}–{entry.dataset.pitches[1]}** and velocities "
                    f"**{entry.dataset.velocities[0]}–{entry.dataset.velocities[1]}**. Run the rest of the "
                    f"pipeline over it:\n```\nuv run optisample pipeline {entry.dataset.source.path} "
                    f"--budget-kb 512 --out artifacts-selection/{entry.source.instrument_id}/"
                    f"{entry.source.stage.value}\n```"
                )
                for entry in _written.written
            ],
            panels.table(clusters.pick_rows(_written.selection, named)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
