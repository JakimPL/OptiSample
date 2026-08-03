from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts.ranking import ListeningSet, dump_ranking
from optisample.calibrate.ranking import (
    PairQuota,
    RankingGrid,
    RankingSettings,
)
from optisample.config.optimize import SweepConfig
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.orchestrate.audio import LoadedInstrument
from optisample.optimize.orchestrate.looping import LoopedInstrument, run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.trim import NO_SCREEN
from optisample.optimize.tasks import AudioMap

SR = 44_100
PITCHES = (60, 62)
NOTE_S = 0.5
REPEATED = 1
_SEED = 137
_LADDER = (11_025, 16_000, 22_050)


@pytest.fixture
def sample_rate() -> int:
    """The rate the listening fixtures are recorded and measured at."""
    return SR


@pytest.fixture
def listening_audio(piano_note: Callable[..., NDArray[np.float64]]) -> AudioMap:
    """One recording per pitch a written listening set is built over."""
    return {SampleKey(pitch, 100): piano_note(pitch, 100, 0.6, seed=pitch * 137) for pitch in PITCHES}


@pytest.fixture
def listening_instrument() -> InstrumentSpec:
    """An instrument whose material plays each recorded pitch once."""
    return InstrumentSpec(
        id="piano",
        budget_kb=48.0,
        samples=[SourceSample(file=f"{pitch}.wav", pitch=pitch, velocity=100) for pitch in PITCHES],
        material=[NoteEvent(pitch=pitch, velocity=100, duration_s=NOTE_S, count=1) for pitch in PITCHES],
    )


@pytest.fixture
def listening_settings(
    sweep: Callable[..., SweepConfig],
    optimize_settings: Callable[..., OptimizeSettings],
) -> OptimizeSettings:
    """A run settled over a short rate ladder, so a widened grid has rungs to step down to."""
    return optimize_settings(sweep=sweep(rates=_LADDER, depth=16), seed=_SEED)


@pytest.fixture
def ranking_settings() -> RankingSettings:
    """A small listening set: both depths, one rung down, and one question of each put twice."""
    return RankingSettings(
        grid=RankingGrid(depths=(16, 8), rate_steps=1),
        quota=PairQuota(loop=1, rate=1, depth=1, compress=0, trade=1),
        byte_tolerance=0.15,
        min_duration_s=NOTE_S / 2,
        repeats=REPEATED,
        seed=_SEED,
    )


@pytest.fixture
def looped(
    listening_instrument: InstrumentSpec,
    listening_audio: AudioMap,
    listening_settings: OptimizeSettings,
) -> LoopedInstrument:
    """The looped run a listening set is assembled from."""
    loaded = LoadedInstrument(
        instrument=listening_instrument,
        audio=dict(listening_audio),
        sample_rate=SR,
        screen=NO_SCREEN,
    )
    return run_loops(loaded, listening_settings)


@pytest.fixture
def written(
    looped: LoopedInstrument,
    listening_settings: OptimizeSettings,
    ranking_settings: RankingSettings,
    tmp_path: Path,
) -> ListeningSet:
    """One instrument's listening set as it lands on disk, which is what both reading and ranking start from."""
    return dump_ranking(looped, tmp_path, listening_settings, ranking_settings)
