"""Deterministic synthetic instruments so the pipeline is testable without real recordings.

Split into pure synthesis (:mod:`~optisample.synth.archetypes` — archetype waveforms from a
:class:`~optisample.config.synth.SynthConfig`) and serialization (:mod:`~optisample.synth.generate`
— rendering a preset grid to WAVs and a ``manifest.yaml``). This package exposes the public surface;
import each name from here.
"""

from optisample.synth.archetypes import (
    Archetype,
    NoteSpec,
    draw_phases,
    render_sample,
    synthesize,
)
from optisample.synth.generate import DemoSettings, generate_demo

__all__ = [
    "Archetype",
    "DemoSettings",
    "NoteSpec",
    "draw_phases",
    "generate_demo",
    "render_sample",
    "synthesize",
]
