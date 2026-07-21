"""Value objects the calibrator passes between rendering and scoring.

A :class:`NoteProbe` says which note to render; a :class:`CalibrationContext` bundles the ``openmpt123``
render settings, the IT playback the module carries, and the loudness-matched composite metric; a
:class:`RendererAgreement` carries the per-note comparison result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.dsp.surrogate import MAX_VOLUME
from optisample.metrics.composite import CompositeFidelity

_DEFAULT_PROBE_DURATION_S: Final = 0.5  # calibration notes last half a second unless a probe overrides it


@dataclass(frozen=True)
class NoteProbe:
    """One note to render for calibration: which key, how loud (0..64), and for how long."""

    pitch: int
    volume: int = MAX_VOLUME
    duration_s: float = _DEFAULT_PROBE_DURATION_S


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
