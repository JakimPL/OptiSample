"""Calibrate the numpy surrogate renderer against ``openmpt123`` -- the ground-truth engine.

The optimizer minimizes distortion measured with the fast surrogate; this package checks that doing so
optimizes reality. It is split into the value objects the calibrator passes around
(:mod:`~optisample.calibrate.context`), the minimal IT module handed to ``openmpt123``
(:mod:`~optisample.calibrate.modules`), and the render-and-compare workflow
(:mod:`~optisample.calibrate.agreement`). This package exposes the public surface; import each name from here.
"""

from optisample.calibrate.agreement import (
    distortion_vs_source,
    rank_correlation,
    render_note_openmpt,
    render_note_surrogate,
    renderer_agreement,
)
from optisample.calibrate.context import CalibrationContext, NoteProbe, RendererAgreement
from optisample.calibrate.modules import single_note_module

__all__ = [
    "CalibrationContext",
    "NoteProbe",
    "RendererAgreement",
    "distortion_vs_source",
    "rank_correlation",
    "render_note_openmpt",
    "render_note_surrogate",
    "renderer_agreement",
    "single_note_module",
]
