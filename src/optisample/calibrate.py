"""Calibrate the numpy surrogate renderer against ``openmpt123`` -- the ground-truth engine.

The optimizer minimizes distortion measured with the fast surrogate (:mod:`optisample.dsp.surrogate`);
this module checks that doing so optimizes *reality*. It renders the same stored sample and note
command two ways and answers two questions:

* :func:`renderer_agreement` -- how closely do the surrogate and ``openmpt123`` agree on the *same*
  stored sample? (A faithful surrogate makes the objective trustworthy.)
* :func:`distortion_vs_source` + :func:`rank_correlation` -- across a set of encodings, does the
  surrogate rank operating points the way ``openmpt123`` does? (Rank preservation is what lets the
  budget solver pick the same winners it would pick against ground truth.)

Every comparison runs through the loudness-normalized composite, so IT's gain staging -- which
attenuates the absolute level heavily -- is not mistaken for timbre error; the level gap is reported
separately as ``loudness_delta_lu``. Repitching (playing a stored sample at a non-root key) is a
genuine part of the calibration: the surrogate resamples in numpy, openmpt123 uses its configured
interpolation filter, and this is where the two engines can most diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.dsp.surrogate import MAX_VOLUME, StoredSample, render
from optisample.io.it_writer import (
    MAX_ROWS,
    TICKS_PER_ROW_BASE,
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    identity_note_map,
    it_playback,
)
from optisample.io.render import render_module
from optisample.metrics.base import Signal
from optisample.metrics.composite import CompositeFidelity, evaluate
from optisample.optimize.export import c5speed_for_pitch


@dataclass(frozen=True)
class NoteProbe:
    """One note to render for calibration: which key, how loud (0..64), and for how long."""

    pitch: int
    volume: int = MAX_VOLUME
    duration_s: float = 0.5


@dataclass(frozen=True)
class RendererAgreement:
    """How closely the surrogate and openmpt123 agree on one rendered note.

    ``breakdown`` is the raw per-metric distance (not weighted), which is what makes this a
    *calibration* result rather than a single opaque score: it shows *which* term drives any
    disagreement. Log-scale terms (``mcd``, ``logmel_l1``) can inflate on near-silent frames even when
    the waveforms are audibly identical, so read ``distance`` alongside the breakdown.
    """

    probe: NoteProbe
    distance: float  # weighted composite distance (loudness-matched); 0 = identical
    loudness_delta_lu: float  # raw level gap surrogate vs openmpt, reported not penalized
    breakdown: dict[str, float]  # per-metric raw distances (mrstft, logmel_l1, spectral_shape, mcd, ...)


@dataclass(frozen=True)
class CalibrationContext:
    """Everything the calibrator renders and scores a note with.

    ``render`` is how ``openmpt123`` renders the module, ``playback`` is the IT global playback the
    module carries, and ``composite`` is the loudness-matched metric the two engines are compared under.
    """

    render: RenderConfig
    playback: PlaybackConfig
    composite: CompositeFidelity


def single_note_module(
    stored: StoredSample, probe: NoteProbe, playback: ITPlayback, *, name: str = "calib"
) -> ITModule:
    """Wrap ``stored`` in a minimal one-sample, one-note module that plays ``probe`` from the start.

    The sample's ``C5Speed`` is set so its root key plays at the natural rate; triggering ``probe.pitch``
    then transposes exactly as the surrogate does. The note is held (no cut) and the pattern is sized to
    outlast ``probe.duration_s`` -- callers trim the render to the duration they asked for.
    """
    row_seconds = playback.speed * TICKS_PER_ROW_BASE / playback.tempo
    rows = min(MAX_ROWS, int(probe.duration_s / row_seconds) + 2)
    sample = ITSample(
        name=name,
        pcm=stored.pcm,
        depth_bits=stored.depth_bits,
        c5speed=c5speed_for_pitch(stored.sample_rate, stored.root_pitch),
    )
    instrument = ITInstrument(name=name, note_map=identity_note_map({probe.pitch: 1}))
    pattern = ITPattern(rows=rows, cells=((0, 0, ITCell(note=probe.pitch, instrument=1, volume=probe.volume)),))
    return ITModule(
        name=name, samples=(sample,), instruments=(instrument,), patterns=(pattern,), orders=(0,), playback=playback
    )


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
