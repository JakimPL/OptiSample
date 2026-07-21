"""Render one note two ways -- numpy surrogate and ``openmpt123`` -- and measure how well they agree.

The optimizer minimizes distortion under the fast surrogate (:mod:`optisample.dsp.surrogate`); these
functions confirm that doing so tracks reality. :func:`renderer_agreement` scores how closely the two
engines match on the same stored sample, and :func:`distortion_vs_source` + :func:`rank_correlation`
check that the surrogate ranks operating points the way ``openmpt123`` does -- the property that lets the
budget solver pick the winners ground truth would pick.

Every comparison runs through the loudness-normalized composite, so IT's gain staging (which attenuates
the absolute level heavily) surfaces as a reported ``loudness_delta_lu`` rather than as timbre error.
Repitching (playing a stored sample at a non-root key) is a genuine part of the calibration: the
surrogate resamples in numpy while ``openmpt123`` uses its configured interpolation filter, and that is
where the two engines diverge most.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from optisample.calibrate.context import CalibrationContext, NoteProbe, RendererAgreement
from optisample.calibrate.modules import single_note_module
from optisample.config.render import RenderConfig
from optisample.dsp.surrogate import StoredSample, render
from optisample.io.it_writer import ITPlayback, it_playback
from optisample.io.render import render_module
from optisample.metrics.base import Signal
from optisample.metrics.composite import evaluate


def render_note_surrogate(stored: StoredSample, probe: NoteProbe, out_rate: int) -> Signal:
    """Render ``probe`` from ``stored`` with the numpy surrogate at ``out_rate``."""
    return render(stored, out_rate, pitch=probe.pitch, volume=probe.volume, duration_s=probe.duration_s)


def render_note_openmpt(
    stored: StoredSample, probe: NoteProbe, render_config: RenderConfig, playback: ITPlayback
) -> Signal:
    """Render ``probe`` from ``stored`` with openmpt123, trimmed to ``probe.duration_s``."""
    audio, rate = render_module(single_note_module(stored, probe, playback), render_config)
    frames = int(round(probe.duration_s * rate))
    return np.asarray(audio[:frames], dtype=np.float64)


def renderer_agreement(stored: StoredSample, probe: NoteProbe, ctx: CalibrationContext) -> RendererAgreement:
    """Compare the surrogate and openmpt123 renders of the same note (see :class:`RendererAgreement`)."""
    playback = it_playback(ctx.playback)
    surrogate = render_note_surrogate(stored, probe, ctx.render.sample_rate)
    openmpt = render_note_openmpt(stored, probe, ctx.render, playback)
    report = evaluate(surrogate, openmpt, ctx.render.sample_rate, ctx.composite)
    return RendererAgreement(
        probe=probe,
        distance=report.fidelity,
        loudness_delta_lu=report.diagnostics["loudness_delta_lu"],
        breakdown=report.breakdown,
    )


def distortion_vs_source(
    reference: Signal, stored: StoredSample, probe: NoteProbe, ctx: CalibrationContext
) -> tuple[float, float]:
    """Distortion of ``stored`` against ``reference`` (source at the analysis rate), surrogate then openmpt.

    ``reference`` must already be at ``ctx.render.sample_rate`` (the analysis rate); it is length-matched
    to each render internally. Returns ``(surrogate_distortion, openmpt_distortion)`` -- the two numbers
    whose *ranking* across encodings should agree.
    """
    playback = it_playback(ctx.playback)
    surrogate = render_note_surrogate(stored, probe, ctx.render.sample_rate)
    openmpt = render_note_openmpt(stored, probe, ctx.render, playback)
    surrogate_distortion = evaluate(reference, surrogate, ctx.render.sample_rate, ctx.composite).fidelity
    openmpt_distortion = evaluate(reference, openmpt, ctx.render.sample_rate, ctx.composite).fidelity
    return surrogate_distortion, openmpt_distortion


def rank_correlation(surrogate_distortions: Signal, openmpt_distortions: Signal) -> float:
    """Spearman rank correlation between the surrogate's and openmpt123's distortions (1 = same order).

    Returns ``nan`` when there are fewer than two points to rank.
    """
    if len(surrogate_distortions) < 2:
        return float("nan")
    correlation, _pvalue = spearmanr(surrogate_distortions, openmpt_distortions)
    return float(correlation)
