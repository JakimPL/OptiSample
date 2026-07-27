from __future__ import annotations

import numpy as np
import pytest

from optisample.config.synth import SynthConfig
from optisample.synth import Archetype, NoteSpec, synthesize

QUICK_RATE = 8_000


def _note(velocity: int) -> NoteSpec:
    return NoteSpec(pitch=60, velocity=velocity, controller=0.0, duration_s=0.5, sample_rate=QUICK_RATE)


@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_higher_velocity_is_louder(archetype: Archetype, synth_config: SynthConfig) -> None:
    rng = np.random.default_rng(0)
    soft = synthesize(archetype, _note(40), rng, synth_config)
    loud = synthesize(archetype, _note(120), rng, synth_config)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2)))

    assert rms(loud) > rms(soft)


@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_higher_velocity_is_brighter(archetype: Archetype, synth_config: SynthConfig) -> None:
    rng = np.random.default_rng(0)
    soft = synthesize(archetype, _note(40), rng, synth_config)
    loud = synthesize(archetype, _note(120), rng, synth_config)

    def centroid(x: np.ndarray) -> float:
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.size, 1.0 / QUICK_RATE)
        total = float(np.sum(mag))
        return float(np.sum(freqs * mag) / total) if total > 0 else 0.0

    assert centroid(loud) > centroid(soft)
