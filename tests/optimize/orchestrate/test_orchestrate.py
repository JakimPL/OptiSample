from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import load_config
from optisample.config.optimize import EVERY_SAMPLE, SweepConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.audio import read_wav, write_wav
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.orchestrate import optimize_instrument, run_instrument
from optisample.optimize.orchestrate.audio import decode_recordings, load_instrument_audio
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import InstrumentPlan
from optisample.optimize.tasks import StoredRecordings
from optisample.progress import NO_PROGRESS
from optisample.synth import NoteSpec, synthesize
from tests.conftest import TEST_CONFIG_DIR

Recordings = Callable[..., StoredRecordings]

SR = 44_100
PITCHES = (60, 67)
VELOCITIES = (50, 100)

# synthesize is a test-signal generator here; its synth config is fixture-independent test data.
_CONFIG = load_config(TEST_CONFIG_DIR)
_SYNTH = _CONFIG.synth
_REDUCE = _CONFIG.reduce
_LOOP = _CONFIG.loop


def note(pitch: int, velocity: int, dur: float = 0.5) -> NDArray[np.float64]:
    return synthesize(
        "piano",
        NoteSpec(pitch, velocity, 0.0, dur, SR),
        np.random.default_rng(pitch * 200 + velocity),
        _SYNTH,
    )


def demo_audio() -> dict[SampleKey, NDArray[np.float64]]:
    return {SampleKey(p, v): note(p, v, dur=0.6) for p in PITCHES for v in VELOCITIES}


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
    """The ladder these tests offer, over the bundled defaults; the reduction picks the rung it stores at."""
    return sweep(rates=(44_100, 11_025), depths=(16,))


@pytest.fixture
def optimize(
    grid: SweepConfig,
    optimize_settings: Callable[..., OptimizeSettings],
    recordings: Recordings,
) -> Callable[..., InstrumentPlan]:
    """Optimize the demo instrument at a byte budget, defaulting to the exact solver."""

    def _optimize(budget_kb: float, method: str = "exact") -> InstrumentPlan:
        settings = optimize_settings(sweep=grid, method=method)
        return optimize_instrument(instrument(budget_kb), recordings(demo_audio(), SR), settings)

    return _optimize


def test_plan_respects_budget_and_covers_every_material_pitch(optimize: Callable[..., InstrumentPlan]) -> None:
    plan = optimize(64.0)
    assert plan.used_bytes <= plan.sample_budget_bytes
    assert plan.module_bytes <= plan.module_budget_bytes
    assert tuple(p.pitch for p in plan.pitches) == PITCHES
    assert plan.objective == pytest.approx(sum(p.objective_weight * p.chosen.distortion for p in plan.pitches))


def test_a_tighter_budget_never_spends_more_or_scores_better(optimize: Callable[..., InstrumentPlan]) -> None:
    """The format is settled off the recordings, so a tighter budget is met by what the sample carries on as."""
    generous = optimize(64.0)
    tight = optimize(56.0)
    assert tight.used_bytes <= generous.used_bytes
    assert tight.objective >= generous.objective - 1e-9


def test_exact_and_lagrangian_are_both_feasible(optimize: Callable[..., InstrumentPlan]) -> None:
    exact = optimize(48.0, "exact")
    lagrangian = optimize(48.0, "lagrangian")
    assert exact.used_bytes <= exact.sample_budget_bytes
    assert lagrangian.used_bytes <= lagrangian.sample_budget_bytes
    assert lagrangian.objective >= exact.objective - 1e-9  # exact is optimal


def test_representative_key_is_the_loudest_used_at_each_pitch(optimize: Callable[..., InstrumentPlan]) -> None:
    plan = optimize(64.0)
    reps = {p.pitch: p.representative_key for p in plan.pitches}
    assert reps[60] == SampleKey(60, 100)  # pitch 60 is played at 50 and 100 → store the loud one
    assert reps[67] == SampleKey(67, 100)


def test_each_pitch_hull_is_a_valid_rd_frontier(optimize: Callable[..., InstrumentPlan]) -> None:
    plan = optimize(64.0)
    swept = {grid.pitch: len(grid.encodings) for grid in plan.reduction.grids}
    for pitch in plan.pitches:
        hull = pitch.hull
        # the played span plus one point per loop the pitch offered, less whatever the hull dropped
        assert 1 <= len(hull) <= swept[pitch.pitch]
        assert all(a.stored_bytes < b.stored_bytes for a, b in zip(hull, hull[1:]))  # ascending bytes
        assert all(a.distortion > b.distortion for a, b in zip(hull, hull[1:]))  # descending distortion


def test_infeasible_budget_raises(optimize: Callable[..., InstrumentPlan]) -> None:
    with pytest.raises(BudgetInfeasibleError):
        optimize(2.0)


def test_material_pitch_without_a_recording_raises(
    grid: SweepConfig,
    optimize_settings: Callable[..., OptimizeSettings],
    recordings: Recordings,
) -> None:
    audio = {SampleKey(60, 100): note(60, 100, dur=0.6)}
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("p60.wav"), pitch=60, velocity=100)],
        material=[NoteEvent(pitch=99, velocity=100, duration_s=0.4, count=1)],  # pitch 99 not recorded
    )
    with pytest.raises(ValueError, match="no recorded sample for pitch 99"):
        optimize_instrument(inst, recordings(audio, SR), optimize_settings(sweep=grid))


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


def test_load_instrument_audio_decodes_the_recording_dedup_kept(tmp_path: Path) -> None:
    long_path = tmp_path / "0000_p60_v100.wav"
    short_path = tmp_path / "0001_p60_v100.wav"
    write_wav(long_path, note(60, 100, dur=1.2), SR)
    write_wav(short_path, note(60, 100, dur=1.0), SR)  # same key, still covers the 0.2 s note
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[
            SourceSample(file=long_path, pitch=60, velocity=100),
            SourceSample(file=short_path, pitch=60, velocity=100),
        ],
        material=[NoteEvent(pitch=60, velocity=100, duration_s=0.2, count=1)],
    )
    audio = load_instrument_audio(inst, _REDUCE, _LOOP, NO_PROGRESS).audio
    expected, _ = read_wav(short_path)
    assert list(audio) == [SampleKey(60, 100)]  # both recordings competed for the one key
    np.testing.assert_array_equal(audio[SampleKey(60, 100)], expected)


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
    audio = load_instrument_audio(inst, _REDUCE, _LOOP, NO_PROGRESS).audio
    full, _ = read_wav(path)
    trimmed = round(lead_in_s * SR)
    np.testing.assert_array_equal(audio[SampleKey(60, 100)], full[trimmed:])  # frame 0 lands on the note onset


def test_load_instrument_audio_cuts_the_trail_off_the_end(tmp_path: Path) -> None:
    """The release padding goes the way the pre-roll does, so a decoded recording ends at the release."""
    path = tmp_path / "0000_p60_v100.wav"
    write_wav(path, note(60, 100, dur=0.6), SR)
    lead_in_s, trail_out_s = 0.05, 0.1
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[SourceSample(file=path, pitch=60, velocity=100, lead_in_s=lead_in_s, trail_out_s=trail_out_s)],
        material=[NoteEvent(pitch=60, velocity=100, duration_s=0.4, count=1)],
    )
    audio = load_instrument_audio(inst, _REDUCE, _LOOP, NO_PROGRESS).audio
    full, _ = read_wav(path)
    sounding = full[round(lead_in_s * SR) : len(full) - round(trail_out_s * SR)]
    np.testing.assert_array_equal(audio[SampleKey(60, 100)], sounding)


def test_load_instrument_audio_cuts_a_recording_to_the_length_bound(tmp_path: Path) -> None:
    """A recording running past the trim's bound is decoded down to the span the bound keeps."""
    path = tmp_path / "0000_p60_v100.wav"
    write_wav(path, note(60, 100, dur=0.6), SR)
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[SourceSample(file=path, pitch=60, velocity=100)],
        material=[NoteEvent(pitch=60, velocity=100, duration_s=0.2, count=1)],
    )
    bounded = ReduceConfig.model_validate(
        {**_REDUCE.model_dump(), "trim": {**_REDUCE.trim.model_dump(), "max_length_s": 0.25}}
    )
    audio = load_instrument_audio(inst, bounded, _LOOP, NO_PROGRESS).audio
    assert audio[SampleKey(60, 100)].size == round(0.25 * SR)


def test_load_instrument_audio_turns_away_a_recording_that_never_sounds(tmp_path: Path) -> None:
    """A failed render carries no signal, so it keeps no slot and the notes it served leave with it."""
    silent_path = tmp_path / "0000_p60_v100.wav"
    played_path = tmp_path / "0001_p67_v100.wav"
    write_wav(silent_path, np.full(round(0.6 * SR), 1.0e-6), SR)
    write_wav(played_path, note(67, 100, dur=0.6), SR)
    inst = InstrumentSpec(
        id="piano",
        budget_kb=64.0,
        samples=[
            SourceSample(file=silent_path, pitch=60, velocity=100),
            SourceSample(file=played_path, pitch=67, velocity=100),
        ],
        material=[
            NoteEvent(pitch=60, velocity=100, duration_s=0.2, count=3),
            NoteEvent(pitch=67, velocity=100, duration_s=0.2, count=1),
        ],
    )
    loaded = load_instrument_audio(inst, _REDUCE, _LOOP, NO_PROGRESS)
    assert list(loaded.audio) == [SampleKey(67, 100)]
    assert loaded.screen.silenced == (SampleKey(60, 100),)
    assert loaded.screen.unplayable == (60,)
    assert loaded.screen.dropped_notes == 3
    assert [event.pitch for event in loaded.instrument.material] == [67]


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
    loaded = load_instrument_audio(inst, _REDUCE, _LOOP, NO_PROGRESS)
    audio, sample_rate = loaded.audio, loaded.sample_rate
    assert sample_rate == SR  # the lowest key's rate wins; the 22 kHz one is resampled up
    assert audio[SampleKey(60, 100)].ndim == 1  # stereo downmixed to mono
    # written as 22.05 kHz, resampled up to 44.1 kHz → twice the frames
    assert audio[SampleKey(67, 100)].size == pytest.approx(mono.size * 2, abs=2)


def test_decoding_reads_every_listed_recording_and_not_the_survivors_alone(tmp_path: Path) -> None:
    """The decode a run gives its survivors, offered over a whole listed grid -- one entry per recording."""
    first = tmp_path / "0000_p60_v100.wav"
    second = tmp_path / "0001_p60_v100.wav"
    write_wav(first, note(60, 100, dur=1.2), SR)
    write_wav(second, note(60, 100, dur=1.0), SR)  # the same key, so dedup would keep only one of the two
    samples = [
        SourceSample(file=first, pitch=60, velocity=100),
        SourceSample(file=second, pitch=60, velocity=100),
    ]

    decoded = decode_recordings(samples, _REDUCE.trim, NO_PROGRESS, label="reading")

    assert decoded.sample_rate == SR
    assert len(decoded.signals) == len(samples)
    for signal, path in zip(decoded.signals, (first, second)):
        expected, _ = read_wav(path)
        np.testing.assert_array_equal(signal, expected)


def test_a_recording_carrying_no_signal_leaves_its_own_position_empty(tmp_path: Path) -> None:
    """Which recordings the screen turned away reads off the position, so a caller names its own losses."""
    silent = tmp_path / "0000_p60_v100.wav"
    played = tmp_path / "0001_p67_v100.wav"
    write_wav(silent, np.full(round(0.6 * SR), 1.0e-6), SR)
    write_wav(played, note(67, 100, dur=0.6), SR)
    samples = [
        SourceSample(file=silent, pitch=60, velocity=100),
        SourceSample(file=played, pitch=67, velocity=100),
    ]

    decoded = decode_recordings(samples, _REDUCE.trim, NO_PROGRESS, label="reading")

    assert decoded.signals[0] is None
    assert decoded.signals[1] is not None


def test_every_stored_sample_states_its_share_of_the_objective(optimize: Callable[..., InstrumentPlan]) -> None:
    """One reading every consumer can add up, whichever scale the strategy searched its hulls at."""
    plan = optimize(64.0)
    assert sum(unit.objective_share for unit in plan.sample_units()) == pytest.approx(plan.objective)


@pytest.mark.parametrize(
    ("asked", "expected"),
    [(EVERY_SAMPLE, None), (8, 8)],
    ids=["every-sample-the-format-numbers", "a-cap-the-format-has-room-for"],
)
def test_the_sample_cap_a_run_works_to(
    asked: int,
    expected: int | None,
    grid: SweepConfig,
    optimize_settings: Callable[..., OptimizeSettings],
) -> None:
    """What a run may store is what it asked for, and the format answers when it asked for everything."""
    settings = optimize_settings(sweep=grid, max_samples=asked)
    assert settings.sample_cap == (settings.target.max_samples if expected is None else expected)


def test_a_cap_above_what_the_format_numbers_is_held_to_the_format(
    grid: SweepConfig, optimize_settings: Callable[..., OptimizeSettings]
) -> None:
    """No plan the format could write holds more samples than it numbers, so that bound wins."""
    settings = optimize_settings(sweep=grid, max_samples=1)
    assert settings.sample_cap < optimize_settings(sweep=grid, max_samples=EVERY_SAMPLE).sample_cap
    beyond = optimize_settings(sweep=grid, max_samples=settings.target.max_samples + 1)
    assert beyond.sample_cap == beyond.target.max_samples


_ROOMY_KB = 64.0  # room the metrics alone leave unspent, which is what the penalty is there to claim


@pytest.fixture
def priced(
    sweep: Callable[..., SweepConfig],
    reduce: Callable[..., ReduceConfig],
    optimize_settings: Callable[..., OptimizeSettings],
    recordings: Recordings,
) -> Callable[..., InstrumentPlan]:
    """Optimize the demo instrument with the band a cheap rung gives up charged at ``discard_penalty``."""

    def _priced(discard_penalty: float, *, budget_kb: float = 64.0) -> InstrumentPlan:
        settings = optimize_settings(
            sweep=sweep(rates=(44_100, 11_025), depths=(16,), rate_headroom=1),
            reduce=reduce(bandwidth={"discard_penalty": discard_penalty}),
        )
        return optimize_instrument(instrument(budget_kb), recordings(demo_audio(), SR), settings)

    return _priced


def test_charging_for_discarded_band_buys_a_wider_rung(priced: Callable[..., InstrumentPlan]) -> None:
    """The trade the penalty exists to state: bytes move to band once a run says the band is worth them.

    The metrics read a narrowed sample as close to its reference, so the settled rung wins on its own
    terms; charging for the octaves it gives up is what lets the wider one compete for the same bytes.
    """
    unpriced = priced(0.0, budget_kb=_ROOMY_KB)
    charged = priced(1.0, budget_kb=_ROOMY_KB)

    assert sum(pitch.chosen.params.target_rate for pitch in charged.pitches) > sum(
        pitch.chosen.params.target_rate for pitch in unpriced.pitches
    )
    assert charged.used_bytes > unpriced.used_bytes  # the room the metrics left unspent, claimed
    assert charged.used_bytes <= charged.sample_budget_bytes


def test_a_run_charging_nothing_stores_what_the_headroom_free_run_stores(
    priced: Callable[..., InstrumentPlan],
    optimize: Callable[..., InstrumentPlan],
) -> None:
    """Offering the wider rungs changes nothing on its own, so the headroom costs a run that ignores it nothing."""
    assert [pitch.chosen.params.target_rate for pitch in priced(0.0).pitches] == [
        pitch.chosen.params.target_rate for pitch in optimize(64.0).pitches
    ]
