import numpy as np
from scipy.stats import spearmanr

from optisample.calibrate.context import (
    CalibrationContext,
    NoteProbe,
    RendererAgreement,
)
from optisample.calibrate.modules import single_note_module
from optisample.dsp.surrogate import StoredSample, render
from optisample.io.render import render_module
from optisample.metrics.base import Signal
from optisample.metrics.composite import evaluate


def render_note_surrogate(
    stored: StoredSample,
    probe: NoteProbe,
    out_rate: int,
) -> Signal:
    """Render ``probe`` from ``stored`` with the numpy surrogate at ``out_rate``."""
    return render(
        stored,
        out_rate,
        pitch=probe.pitch,
        volume=probe.volume,
        duration_s=probe.duration_s,
    )


def render_note_openmpt(
    stored: StoredSample,
    probe: NoteProbe,
    context: CalibrationContext,
) -> Signal:
    """Render ``probe`` from ``stored`` with openmpt123, trimmed to ``probe.duration_s``."""
    module = single_note_module(stored, probe, context.playback, context.target)
    audio, rate = render_module(module, context.render)
    frames = round(probe.duration_s * rate)
    return np.asarray(audio[:frames], dtype=np.float64)


def renderer_agreement(
    stored: StoredSample,
    probe: NoteProbe,
    context: CalibrationContext,
) -> RendererAgreement:
    """Compare the surrogate and openmpt123 renders of the same note (see :class:`RendererAgreement`)."""
    surrogate = render_note_surrogate(stored, probe, context.render.sample_rate)
    openmpt = render_note_openmpt(stored, probe, context)
    report = evaluate(
        surrogate,
        openmpt,
        context.render.sample_rate,
        context.composite,
    )
    return RendererAgreement(
        probe=probe,
        distance=report.fidelity,
        loudness_delta_lu=report.diagnostics["loudness_delta_lu"],
        breakdown=report.breakdown,
    )


def distortion_vs_source(
    reference: Signal,
    stored: StoredSample,
    probe: NoteProbe,
    context: CalibrationContext,
) -> tuple[float, float]:
    """Distortion of ``stored`` against ``reference`` (source at the analysis rate), surrogate then openmpt.

    ``reference`` must already be at ``context.render.sample_rate`` (the analysis rate); it is length-matched
    to each render internally. Returns ``(surrogate_distortion, openmpt_distortion)`` -- the two numbers
    whose *ranking* across encodings should agree.
    """
    surrogate = render_note_surrogate(stored, probe, context.render.sample_rate)
    openmpt = render_note_openmpt(stored, probe, context)
    surrogate_distortion = evaluate(
        reference,
        surrogate,
        context.render.sample_rate,
        context.composite,
    ).fidelity
    openmpt_distortion = evaluate(
        reference,
        openmpt,
        context.render.sample_rate,
        context.composite,
    ).fidelity

    return surrogate_distortion, openmpt_distortion


def rank_correlation(
    surrogate_distortions: Signal,
    openmpt_distortions: Signal,
) -> float:
    """Spearman rank correlation between the surrogate's and openmpt123's distortions (1 = same order).

    Returns ``nan`` when there are fewer than two points to rank.
    """
    if len(surrogate_distortions) < 2:
        return float("nan")

    correlation, _ = spearmanr(surrogate_distortions, openmpt_distortions)
    return float(correlation)
