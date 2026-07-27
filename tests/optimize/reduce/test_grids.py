from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dsp import EncodeConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import ReduceConfig
from optisample.metrics.composite import build_composite
from optisample.optimize.reduce.bandwidth import ClipDemand, candidate_params, useful_rate_hz
from optisample.optimize.reduce.grids import (
    ClipRequest,
    GridContext,
    narrow_grid,
    narrow_grids,
)
from optisample.parallel import IN_PROCESS
from optisample.progress import NO_PROGRESS
from trackmod.module.storage import Storage

SR = 22_050
_PITCHES = (60, 67)
_NOTE_S = 0.4
_RATES = (16_000, 8_000, 4_000)
_KEPT = 2  # candidates the shortlist keeps, so a test states the size it expects back
_OWN_KEY = 0
_ONE_KEY = 1

ReduceFactory = Callable[..., ReduceConfig]
SweepFactory = Callable[..., SweepConfig]


@dataclass(frozen=True)
class _Clip:
    """A stand-in pitch task, carrying a field beyond the three narrowing reads of a stored clip."""

    pitch: int
    representative: NDArray[np.float64]
    max_duration_s: float
    scored_classes: int


def _decayed_noise(duration_s: float, seed: int) -> NDArray[np.float64]:
    """A decaying noise burst, broadband enough that every candidate rate keeps part of it."""
    frames = round(duration_s * SR)
    generator = np.random.default_rng(seed)
    envelope = np.exp(-3.0 * np.arange(frames, dtype=np.float64) / frames)
    return np.asarray(0.5 * envelope * generator.standard_normal(frames), dtype=np.float64)


@pytest.fixture
def context(
    metrics_config: MetricsConfig,
    encode_config: EncodeConfig,
    storage: Storage,
    sweep: SweepFactory,
    reduce: ReduceFactory,
) -> GridContext:
    """A narrowing context over the explicit ``_RATES`` grid, keeping ``_KEPT`` encodings per clip."""
    return GridContext(
        sample_rate=SR,
        metrics=metrics_config,
        encode=encode_config,
        storage=storage,
        sweep=sweep(rates=_RATES),
        bandwidth=reduce(bandwidth={"candidates": _KEPT}).bandwidth,
        byte_target=8_000,
    )


@pytest.fixture
def clips() -> tuple[_Clip, ...]:
    return tuple(
        _Clip(pitch=pitch, representative=_decayed_noise(4.0, seed=pitch), max_duration_s=_NOTE_S, scored_classes=1)
        for pitch in _PITCHES
    )


@pytest.fixture
def demand() -> ClipDemand:
    """What a key sounding its own recording asks of it: the note's span, at the pitch recorded."""
    return ClipDemand(trim_s=_NOTE_S, delta_semitones=_OWN_KEY, key_count=_ONE_KEY)


# --- what a context carries -------------------------------------------------------------------------


def test_a_context_assembles_the_metric_its_config_names(context: GridContext, metrics_config: MetricsConfig) -> None:
    """The recipe travels rather than the metric, so every process narrowing a grid scores the same way."""
    assert context.composite == build_composite(metrics_config)


# --- narrowing one pitch ----------------------------------------------------------------------------


def test_a_grid_states_the_rate_its_own_content_justifies(
    clips: tuple[_Clip, ...], context: GridContext, demand: ClipDemand
) -> None:
    clip = clips[0]
    grid = narrow_grid(clip, context)
    assert grid.pitch == clip.pitch
    assert grid.useful_rate_hz == useful_rate_hz(clip.representative, demand, SR, context.bandwidth)


def test_a_grid_holds_the_encodings_the_sweep_will_run_for_that_pitch(
    clips: tuple[_Clip, ...], context: GridContext, demand: ClipDemand
) -> None:
    """The shortlist recorded here is the one the cost model reads back, so both must agree exactly."""
    clip = clips[0]
    assert narrow_grid(clip, context).shortlist == candidate_params(clip.representative, demand, context)
    assert len(narrow_grid(clip, context).shortlist) == _KEPT


def test_a_clip_is_read_through_the_three_fields_narrowing_needs(
    clips: tuple[_Clip, ...], context: GridContext
) -> None:
    """Anything a pitch task carries beyond them stays with the caller, which is what a worker is spared."""
    clip = clips[0]
    request = ClipRequest(pitch=clip.pitch, representative=clip.representative, max_duration_s=clip.max_duration_s)
    assert narrow_grid(request, context) == narrow_grid(clip, context)


# --- narrowing every pitch --------------------------------------------------------------------------


def test_every_clip_earns_one_grid_in_the_order_it_was_given(clips: tuple[_Clip, ...], context: GridContext) -> None:
    grids = narrow_grids(clips, context, workers=IN_PROCESS, progress=NO_PROGRESS)
    assert [grid.pitch for grid in grids] == [clip.pitch for clip in clips]
    assert grids == tuple(narrow_grid(clip, context) for clip in clips)


def test_narrowing_nothing_answers_with_no_grids(context: GridContext) -> None:
    assert narrow_grids((), context, workers=IN_PROCESS, progress=NO_PROGRESS) == ()
