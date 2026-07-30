from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts.context import DumpContext, DumpSettings
from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.tracker.target import export_target
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.grouping.optimize import optimize_instrument_grouped
from optisample.optimize.orchestrate import optimize_instrument, prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import AudioMap, StoredRecordings

Recordings = Callable[..., StoredRecordings]

SR = 44_100
PITCHES = (60, 62, 64)
_CONFIG = load_config()
_STORED_CEILING_HZ = 5_000.0  # the band these fixtures store, which lands them on the 11 kHz rung


def _narrow_band_reduce() -> ReduceConfig:
    """The bundled reduction, storing only the band ``_STORED_CEILING_HZ`` names.

    That ceiling is what puts these fixtures' samples on the 11 kHz rung, so the budgets they assert
    against stay at the scale of a handful of stored samples.
    """
    raw = _CONFIG.reduce.model_dump()
    return ReduceConfig.model_validate({**raw, "bandwidth": {**raw["bandwidth"], "ceiling_hz": _STORED_CEILING_HZ}})


@pytest.fixture
def tiny_settings() -> OptimizeSettings:
    """Cheap swept settings straight from the bundled config (dither off → the re-encode is deterministic)."""
    grid = SweepConfig.model_validate(
        {**_CONFIG.optimize.sweep.model_dump(), "rates": (11_025,), "depth": 8, "dither": False}
    )
    return OptimizeSettings(
        loop=_CONFIG.loop,
        sweep=grid,
        reduce=_narrow_band_reduce(),
        layers=_CONFIG.optimize.layers,
        encode=_CONFIG.encode,
        metrics=_CONFIG.analysis.metrics,
        velocity=_CONFIG.optimize.velocity,
        method=_CONFIG.optimize.budget.method,
        energy_exponent=_CONFIG.optimize.budget.energy_exponent,
        max_samples=_CONFIG.optimize.budget.max_samples,
        target=export_target(_CONFIG.export.tracker),
    )


@pytest.fixture
def no_render_settings(tiny_settings: OptimizeSettings) -> DumpSettings:
    """Dump settings that skip the openmpt123 ground-truth render (fast, deterministic)."""
    return DumpSettings(
        optimize=tiny_settings,
        render=_CONFIG.export.render,
        playback=_CONFIG.export.playback,
        envelope=_CONFIG.export.envelope,
        render_ground_truth=False,
    )


@pytest.fixture
def demo_audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    return {SampleKey(pitch, 100): piano_note(pitch, 100, 0.6, seed=pitch * 137 + 100) for pitch in PITCHES}


@pytest.fixture
def demo_instrument() -> InstrumentSpec:
    samples = [SourceSample(file=f"{pitch}.wav", pitch=pitch, velocity=100) for pitch in PITCHES]
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=4) for pitch in PITCHES]
    return InstrumentSpec(id="piano", budget_kb=48.0, samples=samples, material=material)


@pytest.fixture
def dump_context(
    demo_instrument: InstrumentSpec,
    demo_audio: AudioMap,
    no_render_settings: DumpSettings,
    recordings: Recordings,
) -> DumpContext:
    stored = recordings(demo_audio, SR)
    return DumpContext(
        instrument=demo_instrument,
        recordings=stored,
        inputs=prepare_run(demo_instrument, stored, no_render_settings.optimize),
        settings=no_render_settings,
    )


@pytest.fixture
def ungrouped_plan(
    demo_instrument: InstrumentSpec,
    demo_audio: AudioMap,
    no_render_settings: DumpSettings,
    recordings: Recordings,
) -> InstrumentPlan:
    return optimize_instrument(demo_instrument, recordings(demo_audio, SR), no_render_settings.optimize)


@pytest.fixture
def grouped_plan(
    demo_instrument: InstrumentSpec,
    demo_audio: AudioMap,
    no_render_settings: DumpSettings,
    recordings: Recordings,
) -> GroupedInstrumentPlan:
    return optimize_instrument_grouped(demo_instrument, recordings(demo_audio, SR), no_render_settings.optimize)
