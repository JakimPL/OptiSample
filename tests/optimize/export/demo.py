"""The demo instrument the exporter tests optimize: a 2x2 pitch/velocity grid and the song it plays."""

from __future__ import annotations

from pathlib import Path

from optisample.model import InstrumentSpec, NoteEvent, SourceSample

SR = 44_100
PITCHES = (60, 67)
VELOCITIES = (50, 100)

DEFAULT_BUDGET_KB = 64.0


def demo_material() -> list[NoteEvent]:
    """The song the demo instrument auditions: three events across its two keys."""
    return [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=8),
        NoteEvent(pitch=60, velocity=50, duration_s=0.5, count=3),
        NoteEvent(pitch=67, velocity=100, duration_s=0.4, count=6),
    ]


def demo_instrument(budget_kb: float = DEFAULT_BUDGET_KB) -> InstrumentSpec:
    """The demo instrument: every ``(pitch, velocity)`` of the grid, playing :func:`demo_material`."""
    samples = [
        SourceSample(file=Path(f"{pitch}_{velocity}.wav"), pitch=pitch, velocity=velocity)
        for pitch in PITCHES
        for velocity in VELOCITIES
    ]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=demo_material())
