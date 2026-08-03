from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.calibrate.ranking import PairQuota, RankingGrid, RankingSettings
from optisample.config.optimize import SweepConfig
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.orchestrate import RunInputs, prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.tasks import AudioMap, Event, StoredRecordings

SR = 44_100
PITCHES = (60, 62)
_SEED = 137
_NOTE_S = 0.5
_RECORDED_S = 0.6
_LADDER = (11_025, 16_000, 22_050)
_STORED_DEPTH = 16


@pytest.fixture
def priced_audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    """One recording per pitch a listening set is built over."""
    return {SampleKey(pitch, 100): piano_note(pitch, 100, _RECORDED_S, seed=pitch * 137) for pitch in PITCHES}


@pytest.fixture
def priced_instrument() -> InstrumentSpec:
    """An instrument whose material plays each recorded pitch once."""
    return InstrumentSpec(
        id="piano",
        budget_kb=48.0,
        samples=[SourceSample(file=f"{pitch}.wav", pitch=pitch, velocity=100) for pitch in PITCHES],
        material=[NoteEvent(pitch=pitch, velocity=100, duration_s=_NOTE_S, count=1) for pitch in PITCHES],
    )


@pytest.fixture
def ranking_run_settings(
    sweep: Callable[..., SweepConfig],
    optimize_settings: Callable[..., OptimizeSettings],
) -> OptimizeSettings:
    """A run settled over a short rate ladder, so a widened grid has rungs to step down to."""
    return optimize_settings(sweep=sweep(rates=_LADDER, depth=_STORED_DEPTH), seed=_SEED)


@pytest.fixture
def priced_run(
    priced_instrument: InstrumentSpec,
    priced_audio: AudioMap,
    recordings: Callable[..., StoredRecordings],
    ranking_run_settings: OptimizeSettings,
) -> RunInputs:
    """The prepared run a listening set is assembled from, over the loops the stage would settle."""
    return prepare_run(priced_instrument, recordings(priced_audio, SR), ranking_run_settings)


@pytest.fixture
def ranking_settings() -> RankingSettings:
    """A small listening set: both depths, one rung down, and a couple of pairs of each question."""
    return RankingSettings(
        grid=RankingGrid(depths=(16, 8), rate_steps=1),
        quota=PairQuota(loop=1, rate=1, depth=1, compress=0, trade=1),
        byte_tolerance=0.15,
        min_duration_s=_NOTE_S / 2,
        repeats=0,
        seed=_SEED,
    )
