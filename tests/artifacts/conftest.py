"""Shared demo instrument, audio, dump settings and pre-built plans for the artifact tests.

The dump-tree test builds its own module-scoped ``generous`` dump (run once, asserted many ways); these
function-scoped fixtures serve the unit-level tests (``test_units``/``test_serialize``/``test_context``)
that need a plan or a :class:`DumpContext` without re-running a full dump.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts.context import DumpContext, DumpSettings
from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.metrics import build_composite
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.grouping import optimize_instrument_grouped
from optisample.optimize.orchestrate import optimize_instrument, prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import AudioMap

SR = 44_100
PITCHES = (60, 62, 64)
_CONFIG = load_config()


@pytest.fixture
def tiny_settings() -> OptimizeSettings:
    """Cheap swept settings straight from the bundled config (dither off → the re-encode is deterministic)."""
    grid = SweepConfig.model_validate(
        {**_CONFIG.sweep.model_dump(), "rates": (11_025,), "depths": (8,), "dither": False}
    )
    return OptimizeSettings(
        sweep=grid,
        encode=_CONFIG.encode,
        composite=build_composite(_CONFIG.metrics),
        velocity=_CONFIG.velocity,
        method=_CONFIG.optimize.method,
    )


@pytest.fixture
def no_render_settings(tiny_settings: OptimizeSettings) -> DumpSettings:
    """Dump settings that skip the openmpt123 ground-truth render (fast, deterministic)."""
    return DumpSettings(
        optimize=tiny_settings, render=_CONFIG.render, playback=_CONFIG.playback, render_ground_truth=False
    )


@pytest.fixture
def demo_audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    return {(pitch, 100): piano_note(pitch, 100, 0.6, seed=pitch * 137 + 100) for pitch in PITCHES}


@pytest.fixture
def demo_instrument() -> InstrumentSpec:
    samples = [SourceSample(file=f"{pitch}.wav", pitch=pitch, velocity=100) for pitch in PITCHES]
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=4) for pitch in PITCHES]
    return InstrumentSpec(id="piano", budget_kb=48.0, samples=samples, material=material)


@pytest.fixture
def dump_context(
    demo_instrument: InstrumentSpec, demo_audio: AudioMap, no_render_settings: DumpSettings
) -> DumpContext:
    _, context, tasks = prepare_run(demo_instrument, demo_audio, SR, no_render_settings.optimize)
    return DumpContext(
        audio=demo_audio,
        sample_rate=SR,
        material=tuple(demo_instrument.material or []),
        eval_context=context,
        tasks_by_pitch={task.pitch: task for task in tasks},
        settings=no_render_settings,
    )


@pytest.fixture
def ungrouped_plan(
    demo_instrument: InstrumentSpec, demo_audio: AudioMap, no_render_settings: DumpSettings
) -> InstrumentPlan:
    return optimize_instrument(demo_instrument, demo_audio, SR, no_render_settings.optimize)


@pytest.fixture
def grouped_plan(
    demo_instrument: InstrumentSpec, demo_audio: AudioMap, no_render_settings: DumpSettings
) -> GroupedInstrumentPlan:
    return optimize_instrument_grouped(demo_instrument, demo_audio, SR, no_render_settings.optimize)
