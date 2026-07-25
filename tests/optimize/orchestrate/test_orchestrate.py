from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.io.audio import read_wav, write_wav
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.orchestrate import optimize_instrument, run_instrument
from optisample.optimize.orchestrate.audio import load_instrument_audio
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import InstrumentPlan
from optisample.synth import NoteSpec, render_sample

SR = 44_100
PITCHES = (60, 67)
VELOCITIES = (50, 100)

# render_sample is a test-signal generator here; its synth config is fixture-independent test data.
_SYNTH = load_config().synth


def note(pitch: int, velocity: int, dur: float = 0.5) -> NDArray[np.float64]:
    return render_sample(
        "piano",
        NoteSpec(pitch, velocity, 0.0, dur, SR),
        np.random.default_rng(pitch * 200 + velocity),
        _SYNTH,
    )


def demo_audio() -> dict[tuple[int, int], NDArray[np.float64]]:
    return {(p, v): note(p, v, dur=0.6) for p in PITCHES for v in VELOCITIES}


def demo_material() -> list[NoteEvent]:
    return [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=8),
        NoteEvent(pitch=60, velocity=50, duration_s=0.5, count=3),
        NoteEvent(pitch=67, velocity=100, duration_s=0.4, count=6),
    ]


def instrument(budget_kb: float) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{p}_{v}.wav"), pitch=p, velocity=v) for p in PITCHES for v in VELOCITIES]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=demo_material())


@pytest.fixture
def grid(sweep: Callable[..., SweepConfig]) -> SweepConfig:
    """The swept grid these tests exercise: 2 rates x 2 depths, over the bundled defaults."""
    return sweep(rates=(44_100, 11_025), depths=(16, 8))


@pytest.fixture
def optimize(grid: SweepConfig, optimize_settings: Callable[..., OptimizeSettings]) -> Callable[..., InstrumentPlan]:
    """Optimize the demo instrument at a byte budget, defaulting to the exact solver."""

    def _optimize(budget_kb: float, method: str = "exact") -> InstrumentPlan:
        settings = optimize_settings(sweep=grid, method=method)
        return optimize_instrument(instrument(budget_kb), demo_audio(), SR, settings)

    return _optimize


def test_plan_respects_budget_and_covers_every_material_pitch(optimize: Callable[..., InstrumentPlan]) -> None:
    plan = optimize(64.0)
    assert plan.used_bytes <= plan.sample_budget_bytes
    assert plan.module_bytes <= plan.module_budget_bytes
    assert tuple(p.pitch for p in plan.pitches) == PITCHES
    assert plan.objective == pytest.approx(sum(p.weight * p.chosen.distortion for p in plan.pitches))


def test_tighter_budget_costs_fewer_bytes_and_more_distortion(optimize: Callable[..., InstrumentPlan]) -> None:
    generous = optimize(64.0)
    tight = optimize(16.0)
    assert tight.used_bytes < generous.used_bytes
    assert tight.objective >= generous.objective - 1e-9


def test_exact_and_lagrangian_are_both_feasible(optimize: Callable[..., InstrumentPlan]) -> None:
    exact = optimize(48.0, "exact")
    lagrangian = optimize(48.0, "lagrangian")
    assert exact.used_bytes <= exact.sample_budget_bytes
    assert lagrangian.used_bytes <= lagrangian.sample_budget_bytes
    assert lagrangian.objective >= exact.objective - 1e-9  # exact is optimal


def test_representative_velocity_is_the_loudest_used_at_each_pitch(optimize: Callable[..., InstrumentPlan]) -> None:
    plan = optimize(64.0)
    reps = {p.pitch: p.representative_velocity for p in plan.pitches}
    assert reps[60] == 100  # pitch 60 is played at 50 and 100 → store the loud one
    assert reps[67] == 100


def test_each_pitch_hull_is_a_valid_rd_frontier(optimize: Callable[..., InstrumentPlan], grid: SweepConfig) -> None:
    plan = optimize(64.0)
    configs = len(grid.rates or ()) * len(grid.depths)  # 2 rates x 2 depths swept per pitch
    for pitch in plan.pitches:
        hull = pitch.hull
        assert 1 <= len(hull) <= configs
        assert all(a.stored_bytes < b.stored_bytes for a, b in zip(hull, hull[1:]))  # ascending bytes
        assert all(a.distortion > b.distortion for a, b in zip(hull, hull[1:]))  # descending distortion


def test_infeasible_budget_raises(optimize: Callable[..., InstrumentPlan]) -> None:
    with pytest.raises(BudgetInfeasibleError):
        optimize(2.0)


def test_material_pitch_without_a_recording_raises(
    grid: SweepConfig, optimize_settings: Callable[..., OptimizeSettings]
) -> None:
    audio = {(60, 100): note(60, 100, dur=0.6)}
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("p60.wav"), pitch=60, velocity=100)],
        material=[NoteEvent(pitch=99, velocity=100, duration_s=0.4, count=1)],  # pitch 99 not recorded
    )
    with pytest.raises(ValueError, match="no recorded sample for pitch 99"):
        optimize_instrument(inst, audio, SR, optimize_settings(sweep=grid))


def test_velocity_map_is_derived_and_anchored_at_full_volume(optimize: Callable[..., InstrumentPlan]) -> None:
    plan = optimize(64.0)
    anchors = {a.velocity: a.volume for a in plan.velocity_map.anchors}
    assert set(anchors) == set(VELOCITIES)
    assert max(anchors.values()) == 64  # loudest recorded velocity anchors the map


def test_rd_curve_brackets_the_chosen_allocation(optimize: Callable[..., InstrumentPlan]) -> None:
    plan = optimize(48.0)
    fits = [pt for pt in plan.curve if pt.total_bytes <= plan.sample_budget_bytes]
    assert fits, "at least the cheapest curve point must fit"
    assert plan.objective <= fits[0].objective + 1e-9  # exact is no worse than the cheapest hull point


def test_run_instrument_reads_wavs_from_disk(
    tmp_path: Path, grid: SweepConfig, optimize_settings: Callable[..., OptimizeSettings]
) -> None:
    samples = []
    for pitch in PITCHES:
        for velocity in VELOCITIES:
            path = tmp_path / f"p{pitch}_v{velocity}.wav"
            write_wav(path, note(pitch, velocity, dur=0.6), SR)
            samples.append(SourceSample(file=path, pitch=pitch, velocity=velocity))
    inst = InstrumentSpec(id="piano", budget_kb=64.0, samples=samples, material=demo_material())
    plan = run_instrument(inst, optimize_settings(sweep=grid))
    assert plan.used_bytes <= plan.sample_budget_bytes
    assert tuple(p.pitch for p in plan.pitches) == PITCHES


def test_load_instrument_audio_keeps_first_sample_per_key(tmp_path: Path) -> None:
    first = note(60, 100, dur=0.6)
    second = note(60, 100, dur=0.3)  # same (pitch, velocity), different recording
    first_path = tmp_path / "0000_p60_v100.wav"
    second_path = tmp_path / "0001_p60_v100.wav"
    write_wav(first_path, first, SR)
    write_wav(second_path, second, SR)
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[
            SourceSample(file=first_path, pitch=60, velocity=100),
            SourceSample(file=second_path, pitch=60, velocity=100),
        ],
        material=[NoteEvent(pitch=60, velocity=100, duration_s=0.4, count=1)],
    )
    audio, _ = load_instrument_audio(inst)
    expected, _ = read_wav(first_path)
    np.testing.assert_array_equal(audio[(60, 100)], expected)  # the earliest listed recording wins the key


def test_load_instrument_audio_trims_lead_in_from_the_front(tmp_path: Path) -> None:
    signal = note(60, 100, dur=0.6)
    path = tmp_path / "0000_p60_v100.wav"
    write_wav(path, signal, SR)
    lead_in_s = 0.05
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[SourceSample(file=path, pitch=60, velocity=100, lead_in_s=lead_in_s)],
        material=[NoteEvent(pitch=60, velocity=100, duration_s=0.4, count=1)],
    )
    audio, _ = load_instrument_audio(inst)
    full, _ = read_wav(path)
    trimmed = round(lead_in_s * SR)
    np.testing.assert_array_equal(audio[(60, 100)], full[trimmed:])  # frame 0 lands on the note onset


def test_load_instrument_audio_downmixes_stereo_and_resamples(tmp_path: Path) -> None:
    mono = note(60, 100, dur=0.6)
    stereo_path = tmp_path / "p60_v100.wav"
    write_wav(stereo_path, np.stack([mono, mono], axis=1), SR)  # 2-channel
    half_path = tmp_path / "p67_v100.wav"
    write_wav(half_path, note(67, 100, dur=0.6), SR // 2)  # different rate
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[
            SourceSample(file=stereo_path, pitch=60, velocity=100),
            SourceSample(file=half_path, pitch=67, velocity=100),
        ],
        material=[NoteEvent(pitch=60, velocity=100, duration_s=0.4, count=1)],
    )
    audio, sample_rate = load_instrument_audio(inst)
    assert sample_rate == SR  # first sample's rate wins; the 22 kHz one is resampled up
    assert audio[(60, 100)].ndim == 1  # stereo downmixed to mono
    # written as 22.05 kHz, resampled up to 44.1 kHz → twice the frames
    assert audio[(67, 100)].size == pytest.approx(mono.size * 2, abs=2)
