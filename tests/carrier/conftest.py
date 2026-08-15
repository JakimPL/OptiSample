from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from optisample.carrier.source import CarrierSource
from optisample.carrier.store import CarrierSettings
from optisample.config import OptiConfig
from optisample.dsp.envelope import Decomposition, decompose, level_reading
from optisample.dsp.level import Clock, db_to_gain, unit_level
from optisample.dsp.loop import Loop
from optisample.dsp.surrogate import NO_LOOPS, EncodingParams, SettledLoop
from optisample.io.tracker.envelope import envelope_grid
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.keys import SampleKey
from optisample.metrics.base import Signal
from optisample.music import midi_to_freq

SR = 22_050
TEMPO = 125
ROOT_PITCH = 60
SEED = 137
RECORDED_PEAK = 0.3  # where a captured note sits under full scale, which is the room its flattened form needs
NO_SPIKE = 0.0  # what a recording declining as evenly as a written curve states carries over its decline

_DB_PER_DECADE = 20.0
_SPIKE_SHARE = 0.02  # how much of a recording a burst too short for a written curve to follow lasts

Recorder = Callable[..., Signal]
Sourcer = Callable[..., CarrierSource]


@pytest.fixture
def sample_rate() -> int:
    """The rate every recording in this tree is read at."""
    return SR


@pytest.fixture
def target(config: OptiConfig) -> ExportTarget:
    """The format these instruments are written through, which states two level grids per sample."""
    return export_target(config.export.tracker)


@pytest.fixture
def struck() -> Recorder:
    """Factory: a struck note at one pitch, falling ``decay_db`` decibels over the seconds it runs."""

    def _struck(
        *,
        pitch: int = ROOT_PITCH,
        seconds: float = 1.5,
        decay_db: float = 30.0,
        peak: float = RECORDED_PEAK,
        spike_db: float = NO_SPIKE,
    ) -> Signal:
        moments = np.arange(round(seconds * SR), dtype=np.float64) / SR
        fall = 10.0 ** (-decay_db * moments / (_DB_PER_DECADE * seconds))
        burst = slice(moments.size // 2, moments.size // 2 + round(_SPIKE_SHARE * moments.size))
        fall[burst] *= db_to_gain(spike_db)
        tone = np.sin(2 * np.pi * midi_to_freq(pitch) * moments)
        return np.asarray(peak * tone * fall, dtype=np.float64)

    return _struck


def split(signal: Signal, sample_rate: int, pitch: int, config: OptiConfig) -> Decomposition:
    """``signal`` as the level it moves through times the carrier that level scales."""
    return decompose(signal, level_reading(sample_rate, config.encode.envelope, midi_to_freq(pitch)))


@pytest.fixture
def source(struck: Recorder, config: OptiConfig) -> Sourcer:
    """Factory: one struck note as a carrier source, looped over the stretch a caller names."""

    def _source(
        *,
        pitch: int = ROOT_PITCH,
        seconds: float = 1.5,
        decay_db: float = 30.0,
        peak: float = RECORDED_PEAK,
        weight: float = 1.0,
        loop: Loop | None = None,
        spike_db: float = NO_SPIKE,
    ) -> CarrierSource:
        signal = struck(pitch=pitch, seconds=seconds, decay_db=decay_db, peak=peak, spike_db=spike_db)
        loops = NO_LOOPS if loop is None else (SettledLoop(loop=loop, level=unit_level(Clock.RECORDED)),)
        return CarrierSource(
            key=SampleKey(pitch=pitch, velocity=100),
            root_pitch=pitch,
            sample_rate=SR,
            decomposition=split(signal, SR, pitch, config),
            loops=loops,
            loop_index=None if loop is None else 0,
            weight=weight,
        )

    return _source


@pytest.fixture
def carrier_settings(config: OptiConfig, target: ExportTarget) -> Callable[..., CarrierSettings]:
    """Factory: what writing a set of carriers is carried out with, at the depth a caller asks for."""

    def _settings(
        *,
        depth: int = 16,
        rate: int = SR,
        written: ExportTarget | None = None,
        ratio: float | None = None,
    ) -> CarrierSettings:
        chosen = target if written is None else written
        held = config.export.instruments.compression
        return CarrierSettings(
            target=chosen,
            grid=envelope_grid(chosen, tempo=TEMPO, release_s=config.export.envelope.release_s),
            params=EncodingParams(target_rate=rate, depth_bits=depth),
            config=config.encode,
            compression=held if ratio is None else held.model_copy(update={"ratio": ratio}),
            seed=SEED,
        )

    return _settings


def levels_db(signal: Signal, sample_rate: int, pitch: int, config: OptiConfig) -> Signal:
    """The decibel level ``signal`` moves through, which is what a flattened waveform holds steady."""
    return np.asarray(split(signal, sample_rate, pitch, config).level, dtype=np.float64)


def level_range_db(values: Sequence[float]) -> float:
    """How far a level travels between its loudest and its quietest moment."""
    return float(np.max(values) - np.min(values))
