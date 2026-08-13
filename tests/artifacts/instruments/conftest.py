from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optisample.config import OptiConfig
from optisample.io.tracker.envelope import EnvelopeGrid, envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.base import Signal
from optisample.music import midi_to_freq

SR = 22_050
ROOT_PITCH = 60
TEMPO_BPM = 115.0
RECORDED_PEAK = 0.3  # where a captured note sits under full scale, which is the room its levelled form needs

_DB_PER_DECADE = 20.0

Recorder = Callable[..., Signal]


@pytest.fixture
def sample_rate() -> int:
    """The rate every recording in this tree is read at."""
    return SR


@pytest.fixture
def recorded() -> Recorder:
    """Factory: a struck note at one pitch, falling ``decay_db`` decibels over the seconds it runs."""

    def _recorded(
        *,
        pitch: int = ROOT_PITCH,
        seconds: float = 1.5,
        decay_db: float = 30.0,
        peak: float = RECORDED_PEAK,
    ) -> Signal:
        moments = np.arange(round(seconds * SR), dtype=np.float64) / SR
        fall = 10.0 ** (-decay_db * moments / (_DB_PER_DECADE * seconds))
        tone = np.sin(2 * np.pi * midi_to_freq(pitch) * moments)
        return np.asarray(peak * tone * fall, dtype=np.float64)

    return _recorded


@pytest.fixture
def release_s(config: OptiConfig) -> float:
    """How long a released note takes to fall silent, which is the one part a recording never states."""
    return config.export.envelope.release_s


@pytest.fixture
def grid_for(config: OptiConfig) -> Callable[[ExportTarget], EnvelopeGrid]:
    """Factory: the grid one format writes a curve onto at the tempo these recordings were played at."""

    def _grid(target: ExportTarget) -> EnvelopeGrid:
        return envelope_grid(
            target,
            tempo=target.tempo(TEMPO_BPM),
            release_s=config.export.envelope.release_s,
        )

    return _grid
