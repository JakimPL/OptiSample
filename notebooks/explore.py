import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import sys
    from pathlib import Path

    root = Path.cwd()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import marimo as mo

    from notebooks.utils import audioio, degrade, loading, views, viz

    return audioio, degrade, io, loading, mo, root, views, viz


@app.cell
def _():
    from optisample.config import load_config
    from optisample.io.tracker.target import export_target
    from optisample.metrics import build_composite

    config = load_config()
    composite = build_composite(config.analysis.metrics)
    storage = export_target(config.export.tracker).storage
    return composite, config, storage


@app.cell
def _(mo):
    mo.md("""
        # OptiSample — sample & metric inspector

        A thin viewer over the P0 ingest and P1 measurement layers: **hear** the recorded/synthetic
        samples, **see** their descriptors and storage footprint per instrument, and **check the
        metrics by inspection** by degrading a sample and watching the score move.

        Reconstruction fidelity is judged on OpenMPT-rendered notes (sinc interpolation); the
        velocity→volume map is a conversion-time artifact. Rendering the *optimized* instrument is
        deferred (P4) — this notebook inspects inputs and the metric layer only.
        """)
    return


@app.cell
def _(config, loading, mo, root):
    _demo_dir = root / "notebooks" / "_demo"
    _default_demo = loading.ensure_demo(_demo_dir, config.synth)
    demo_dir_input = mo.ui.text(
        value=str(_default_demo), label="demo directory (.notes.json + samples)", full_width=True
    )
    mo.md(f"**Demo directory** (defaults to a freshly generated synthetic dataset)\n\n{demo_dir_input}")
    return (demo_dir_input,)


@app.cell
def _(demo_dir_input, loading, mo):
    manifest = loading.load(demo_dir_input.value)
    mo.md(f"Loaded **{manifest.project.name}** — instruments: `{'`, `'.join(loading.instrument_ids(manifest))}`")
    return (manifest,)


@app.cell
def _(loading, manifest, mo):
    instrument_dropdown = mo.ui.dropdown(
        options=loading.instrument_ids(manifest),
        value=loading.instrument_ids(manifest)[0],
        label="Instrument",
    )
    instrument_dropdown
    return (instrument_dropdown,)


@app.cell
def _(instrument_dropdown, loading, manifest, mo, storage, views):
    instrument = loading.get_instrument(manifest, instrument_dropdown.value)
    signals = {loading.sample_label(sample): loading.load_signal(sample) for sample in instrument.samples}
    _frame_counts = [signal.size for signal, _ in signals.values()]
    _budget = views.budget_summary(instrument, _frame_counts, storage)
    mo.vstack(
        [
            mo.md(f"### Instrument: `{instrument.id}`"),
            mo.md("**Budget** — full-length storage vs the instrument's byte budget (motivates the optimizer):"),
            mo.ui.table([_budget], selection=None),
            mo.md("**Material** — the notes the song actually plays (weight = count × duration):"),
            mo.ui.table(views.material_rows(instrument), selection=None),
        ]
    )
    return instrument, signals


@app.cell
def _(config, instrument, loading, mo, signals, storage, views):
    _rows = [
        views.sample_row(sample, *signals[loading.sample_label(sample)], config.analysis.spectral, storage)
        for sample in instrument.samples
    ]
    mo.vstack([mo.md("**Per-sample descriptors & footprint**"), mo.ui.table(_rows, selection=None)])
    return


@app.cell
def _(instrument, loading, mo):
    sample_dropdown = mo.ui.dropdown(
        options=loading.sample_labels(instrument),
        value=loading.sample_labels(instrument)[0],
        label="Sample",
    )
    sample_dropdown
    return (sample_dropdown,)


@app.cell
def _(audioio, config, io, mo, sample_dropdown, signals, viz):
    reference, ref_sr = signals[sample_dropdown.value]
    mo.vstack(
        [
            mo.md(f"#### `{sample_dropdown.value}` — listen & inspect"),
            mo.audio(io.BytesIO(audioio.to_wav_bytes(reference, ref_sr))),
            mo.image(viz.figure_png(viz.waveform_figure(reference, ref_sr, title="waveform"))),
            mo.image(
                viz.figure_png(
                    viz.spectrogram_figure(
                        reference,
                        ref_sr,
                        params=config.analysis.spectral.stft,
                        dynamic_range_db=config.analysis.metrics.preprocess.dynamic_range_db,
                        title="spectrogram (dB rel. peak)",
                    )
                )
            ),
        ]
    )
    return ref_sr, reference


@app.cell
def _(mo):
    kind_dropdown = mo.ui.dropdown(
        options=["quantize", "lowpass", "gain", "resample"], value="quantize", label="degradation"
    )
    bits_slider = mo.ui.slider(4, 16, value=8, label="bits", show_value=True)
    cutoff_slider = mo.ui.slider(500, 20_000, step=500, value=3_000, label="lowpass cutoff (Hz)", show_value=True)
    factor_slider = mo.ui.slider(0.05, 1.0, step=0.05, value=0.3, label="gain factor", show_value=True)
    target_sr_slider = mo.ui.slider(
        4_000, 44_100, step=1_000, value=22_050, label="resample target (Hz)", show_value=True
    )
    mo.vstack(
        [
            mo.md("### Degradation lab — hear it, watch the metric move"),
            kind_dropdown,
            bits_slider,
            cutoff_slider,
            factor_slider,
            target_sr_slider,
        ]
    )
    return bits_slider, cutoff_slider, factor_slider, kind_dropdown, target_sr_slider


@app.cell
def _(
    audioio,
    bits_slider,
    composite,
    config,
    cutoff_slider,
    degrade,
    factor_slider,
    io,
    kind_dropdown,
    mo,
    ref_sr,
    reference,
    target_sr_slider,
    views,
    viz,
):
    _spec = degrade.DegradeSpec(
        kind=kind_dropdown.value,
        bits=int(bits_slider.value),
        cutoff_hz=float(cutoff_slider.value),
        factor=float(factor_slider.value),
        target_sr=int(target_sr_slider.value),
    )
    _candidate = degrade.apply(_spec, reference, ref_sr)
    _row = views.compare(reference, _candidate, ref_sr, _spec.label, composite)
    mo.vstack(
        [
            mo.md(f"**Candidate: {_spec.label}**  · previews are peak-normalized, so read level from `loudness_dLU`."),
            mo.hstack(
                [
                    mo.vstack([mo.md("original"), mo.audio(io.BytesIO(audioio.to_wav_bytes(reference, ref_sr)))]),
                    mo.vstack([mo.md(_spec.label), mo.audio(io.BytesIO(audioio.to_wav_bytes(_candidate, ref_sr)))]),
                ]
            ),
            mo.ui.table([_row], selection=None),
            mo.image(
                viz.figure_png(
                    viz.spectrogram_figure(
                        _candidate,
                        ref_sr,
                        params=config.analysis.spectral.stft,
                        dynamic_range_db=config.analysis.metrics.preprocess.dynamic_range_db,
                        title=f"{_spec.label} spectrogram",
                    )
                )
            ),
        ]
    )
    return


@app.cell
def _(composite, config, degrade, mo, ref_sr, reference, views, viz):
    _smoke_rows = views.compare_specs(reference, ref_sr, degrade.smoke_specs(), composite)
    mo.vstack(
        [
            mo.md(
                "### P1 smoke-test panel\n\n"
                "The fixed degradation set: 16-bit is transparent, 8-bit is clearly audible, the 3 kHz "
                "cut loses brightness, and the level drop moves only loudness. Bar height ≈ fidelity."
            ),
            mo.ui.table(_smoke_rows, selection=None),
            mo.image(viz.figure_png(viz.contribution_bar(_smoke_rows, config.analysis.metrics.weights))),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
