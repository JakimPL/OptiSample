from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.dsp import EncodeConfig, LoopConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import ReduceConfig
from optisample.dsp.spectral import bandlimit
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.reduce.bandwidth import (
    ClipDemand,
    candidate_params,
    useful_rate_hz,
)
from optisample.optimize.reduce.grids import GridContext
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.reduce.summary import (
    ReductionInputs,
    summarize_reduction,
)
from optisample.parallel import IN_PROCESS
from optisample.progress import NO_PROGRESS
from trackmod.module.storage import Storage

SR = 22_050
_PITCHES = (60, 67)
_NOTE_S = 0.4
_RATES = (16_000, 8_000, 4_000)  # an explicit ladder, so a test states the size it expects back
_GRID_RATES = len(_RATES) + 1  # the ladder, plus the clip's own rate, which every clip is also offered
_ENCODINGS_PER_RATE = 3  # one 16-bit entry and both compressions of the 8-bit one, at one loop setting
_FULL_GRID = _ENCODINGS_PER_RATE * _GRID_RATES
_NO_TRANSPOSE = 0
_RATE_PER_BANDWIDTH = 2.0  # Nyquist, which turns a content-edge tolerance into a rate tolerance
_SHARED = 2  # workers, enough to run the pre-pass apart without asking the machine for every core

ReduceFactory = Callable[..., ReduceConfig]
SweepFactory = Callable[..., SweepConfig]


@dataclass(frozen=True)
class _Clip:
    """A stand-in pitch task: what the pre-pass reads about the sample one pitch stores."""

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
    """A narrowing context over the explicit ``_RATES`` grid, keeping two encodings per clip."""
    return GridContext(
        sample_rate=SR,
        metrics=metrics_config,
        encode=encode_config,
        storage=storage,
        sweep=sweep(rates=_RATES),
        bandwidth=reduce(bandwidth={"candidates": 2}).bandwidth,
        byte_target=8_000,
    )


@pytest.fixture
def inputs(context: GridContext, config_dedupe: ReduceConfig, loop_config: LoopConfig) -> ReductionInputs:
    return ReductionInputs(
        reduce=config_dedupe,
        loop=loop_config,
        context=context,
        workers=IN_PROCESS,
        progress=NO_PROGRESS,
    )


@pytest.fixture
def config_dedupe(reduce: ReduceFactory) -> ReduceConfig:
    return reduce()


@pytest.fixture
def instrument() -> InstrumentSpec:
    """Two played pitches recorded at three velocities each, so dedup has something to collapse."""
    samples = [
        SourceSample(file=f"{pitch}_{velocity}.wav", pitch=pitch, velocity=velocity)
        for pitch in _PITCHES
        for velocity in (40, 80, 120)
    ]
    material = [
        NoteEvent(pitch=pitch, velocity=80, duration_s=_NOTE_S, count=3)
        for pitch in _PITCHES
        for _ in range(2)  # two note events per pitch, which merging collapses into one class
    ]
    return InstrumentSpec(id="piano", budget_kb=64.0, samples=samples, material=material)


@pytest.fixture
def audio() -> dict[SampleKey, NDArray[np.float64]]:
    """One survivor per pitch, each long enough to cover the material played at it."""
    return {SampleKey(pitch, 80): _decayed_noise(4.0, seed=pitch) for pitch in _PITCHES}


@pytest.fixture
def clips(audio: dict[SampleKey, NDArray[np.float64]]) -> tuple[_Clip, ...]:
    return tuple(
        _Clip(pitch=key.pitch, representative=signal, max_duration_s=_NOTE_S, scored_classes=1)
        for key, signal in sorted(audio.items())
    )


# --- what each axis reduced to --------------------------------------------------------------------


def test_the_summary_states_both_sides_of_every_reduction(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
) -> None:
    summary = summarize_reduction(instrument, clips, audio, inputs)
    assert (summary.listed_recordings, summary.kept_recordings) == (len(instrument.samples), len(audio))
    assert (summary.played_notes, summary.scored_classes) == (len(instrument.material), len(clips))
    assert summary.grid_size == _FULL_GRID


def test_every_played_pitch_earns_the_grid_the_sweep_will_run(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
    context: GridContext,
) -> None:
    """The shortlist recorded here is the one the cost model reads back, so both must agree exactly."""
    summary = summarize_reduction(instrument, clips, audio, inputs)
    demand = ClipDemand(trim_s=_NOTE_S, delta_semitones=_NO_TRANSPOSE, byte_target=context.byte_target)
    assert summary.shortlists() == {
        clip.pitch: candidate_params(clip.representative, demand, context) for clip in clips
    }


def test_the_shortlisted_total_adds_up_the_pitches(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
) -> None:
    summary = summarize_reduction(instrument, clips, audio, inputs)
    assert summary.shortlisted == sum(len(grid.shortlist) for grid in summary.grids)
    assert summary.shortlisted < summary.grid_size * len(clips)


def test_a_pitch_keeps_the_rate_its_own_content_justifies(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
    context: GridContext,
) -> None:
    """A broadband recording fills the band every score is read over, so it asks for the whole rate."""
    summary = summarize_reduction(instrument, clips, audio, inputs)
    tolerance = _RATE_PER_BANDWIDTH * context.bandwidth.content_band_hz
    assert all(grid.useful_rate_hz == pytest.approx(float(SR), abs=tolerance) for grid in summary.grids)


def test_a_muffled_recording_reports_the_lower_rate_it_narrowed_the_grid_by(
    instrument: InstrumentSpec,
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
    context: GridContext,
) -> None:
    """The rate the summary states is the measurement the shortlist was drawn around, over the same span."""
    muffled = {key: bandlimit(signal, SR, 0.0, 1_800.0) for key, signal in audio.items()}
    clips = tuple(
        _Clip(pitch=key.pitch, representative=signal, max_duration_s=_NOTE_S, scored_classes=1)
        for key, signal in sorted(muffled.items())
    )
    summary = summarize_reduction(instrument, clips, muffled, inputs)
    demand = ClipDemand(trim_s=_NOTE_S, delta_semitones=_NO_TRANSPOSE, byte_target=context.byte_target)
    for grid, clip in zip(summary.grids, clips):
        assert grid.useful_rate_hz == useful_rate_hz(clip.representative, demand, SR, context.bandwidth)
        assert grid.useful_rate_hz < float(SR)


def test_a_pre_pass_shared_across_processes_states_the_same_reduction(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
) -> None:
    """Each pitch is narrowed off its own clip and a fixed dither seed, so scheduling cannot move it."""
    shared = summarize_reduction(instrument, clips, audio, replace(inputs, workers=_SHARED))
    assert shared == summarize_reduction(instrument, clips, audio, inputs)


# --- what the material asks of a kept recording ---------------------------------------------------


def test_a_recording_covering_its_material_leaves_no_shortfall(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
) -> None:
    summary = summarize_reduction(instrument, clips, audio, inputs)
    assert summary.shortfalls == ()
    assert all(recording.duration_s >= recording.required_duration_s for recording in summary.recordings)


def test_a_recording_shorter_than_its_material_is_reported_as_one(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
) -> None:
    """The shortfall the objective would otherwise absorb by scoring a truncated reference."""
    truncated = {**audio, SampleKey(_PITCHES[0], 80): audio[SampleKey(_PITCHES[0], 80)][: round(0.05 * SR)]}
    summary = summarize_reduction(instrument, clips, truncated, inputs)
    assert [recording.key.pitch for recording in summary.shortfalls] == [_PITCHES[0]]
    assert summary.shortfalls[0].duration_s == pytest.approx(0.05, abs=1e-3)


def test_recordings_are_reported_in_key_order(
    instrument: InstrumentSpec,
    clips: tuple[_Clip, ...],
    audio: dict[SampleKey, NDArray[np.float64]],
    inputs: ReductionInputs,
) -> None:
    """Key order is what makes a report and a document line up run after run."""
    reversed_audio = dict(reversed(list(audio.items())))
    summary = summarize_reduction(instrument, clips, reversed_audio, inputs)
    assert [recording.key for recording in summary.recordings] == sorted(audio)
