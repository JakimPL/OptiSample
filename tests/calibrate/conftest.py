"""Shared fixtures for the calibrator tests: the calibration context and signal factories."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.calibrate import CalibrationContext
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.synth import SynthConfig
from optisample.dsp.surrogate import StoredSample
from optisample.io.tracker.target import ExportTarget
from optisample.metrics import CompositeFidelity
from optisample.synth import NoteSpec, render_sample

SAMPLE_RATE = 44_100


@pytest.fixture
def calibration_context(
    render_config: RenderConfig,
    playback_config: PlaybackConfig,
    target: ExportTarget,
    composite: CompositeFidelity,
) -> CalibrationContext:
    """The calibration context (render settings, clock, target format, composite) from the bundled config."""
    return CalibrationContext(render=render_config, playback=playback_config, target=target, composite=composite)


@pytest.fixture
def stored() -> Callable[..., StoredSample]:
    """Factory: a short stored 220 Hz sample at a given root pitch and rate."""

    def _stored(root_pitch: int = 60, rate: int = 22_050, frames: int = 8_000) -> StoredSample:
        time = np.arange(frames) / rate
        pcm = 0.6 * np.sin(2 * np.pi * 220 * time)
        return StoredSample(pcm=pcm, sample_rate=rate, depth_bits=16, root_pitch=root_pitch)

    return _stored


@pytest.fixture
def recording(synth_config: SynthConfig) -> Callable[..., NDArray[np.float64]]:
    """Factory: render one archetype note (a test-signal generator, fixture-independent test data)."""

    def _recording(archetype: str, pitch: int, dur: float) -> NDArray[np.float64]:
        spec = NoteSpec(pitch, 100, 0.0, dur, SAMPLE_RATE)
        return render_sample(archetype, spec, np.random.default_rng(pitch), synth_config)

    return _recording
