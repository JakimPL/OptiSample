from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.config.render import RenderConfig
from optisample.dsp.surrogate import EncodingParams
from optisample.io.render import openmpt123_available, render_module
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.samples import sample_name
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import BudgetBreakdown, GroupedInstrumentPlan, InstrumentPlan, Zone, ZoneOption
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from tests.optimize.export.demo import SR
from trackmod.core.notes.pitch import Note
from trackmod.module.protocol import TrackerModule
from trackmod.module.storage import Storage
from trackmod.spec.pitch import RATE_NOTE

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

_UNREACHABLE_PITCH = 5  # below MIDI 12, so no tracker keyboard numbers it


def test_one_sample_per_planned_pitch_each_sounding_its_own_key(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = build()
    song = module.song
    assert len(song.samples) == len(plan.pitches)
    assert len(song.instruments) == 1
    for index, pitch_plan in enumerate(plan.pitches):
        assignment = song.instruments[0].assignment(Note.from_midi(pitch_plan.pitch))
        assert assignment is not None
        assert assignment.sample == index
        # a key playing its own recording needs no transposition, so it sounds the reference note
        assert assignment.note == Note(RATE_NOTE)


def test_stored_samples_match_the_chosen_operating_points(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = build()
    for pitch_plan, sample in zip(plan.pitches, module.song.samples):
        assert sample.depth == pitch_plan.chosen.params.depth_bits
        assert sample.frames == pitch_plan.chosen.frames
        assert sample.rate == pitch_plan.chosen.params.target_rate  # the true stored rate, untransposed


def test_sample_names_carry_the_instrument_and_the_recorded_note(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = build()
    assert [sample.name for sample in module.song.samples] == [
        sample_name(plan.instrument_id, unit) for unit in plan.sample_units()
    ]


@pytest.mark.parametrize("builder", ["build", "grouped_build"])
def test_sample_bytes_equal_the_budgeted_amount(builder: str, request: pytest.FixtureRequest) -> None:
    plan, module = request.getfixturevalue(builder)()
    storage = module.storage
    stored = sum(storage.sample_bytes(frames=s.frames, depth=s.depth) for s in module.song.samples)
    assert stored == plan.used_bytes  # frames plus the records the format charges is what the solver budgeted


@pytest.mark.parametrize("builder", ["build", "grouped_build"])
def test_build_is_deterministic(builder: str, request: pytest.FixtureRequest) -> None:
    run = request.getfixturevalue(builder)
    _, module_a = run()
    _, module_b = run()
    assert module_a.to_bytes() == module_b.to_bytes()


def test_a_pitch_the_format_does_not_number_raises(
    export_context: ExportContext,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    piano_note: Callable[..., NDArray[np.float64]],
) -> None:
    audio = {SampleKey(_UNREACHABLE_PITCH, 100): piano_note(60, 100, seed=60 * 200 + 100)}
    instrument = InstrumentSpec(
        id="x",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("p.wav"), pitch=_UNREACHABLE_PITCH, velocity=100)],
        material=[NoteEvent(pitch=_UNREACHABLE_PITCH, velocity=100, duration_s=0.4)],
    )
    settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depths=(16, 8)))
    plan = optimize_instrument(instrument, audio, SR, settings)
    with pytest.raises(ValueError, match="outside the IT key range"):
        build_module(plan, audio, SR, instrument.material or [], export_context)


def test_grouped_module_shares_one_sample_across_a_merged_zone(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, module = grouped_build()
    assert len(plan.zones) == 1  # the tight budget merged both keys into one zone
    assert len(module.song.samples) == 1  # ... served by a single stored sample
    representative = plan.zones[0].representative
    for pitch in plan.zones[0].pitches:
        assignment = module.song.instruments[0].assignment(Note.from_midi(pitch))
        assert assignment is not None
        assert assignment.sample == 0  # every covered key reaches the shared sample
        assert assignment.note == Note(RATE_NOTE + pitch - representative)  # transposed from the representative


def test_grouped_sample_keeps_the_representatives_stored_rate(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, module = grouped_build()
    assert module.song.samples[0].rate == plan.zones[0].chosen.params.target_rate


def test_a_grouped_pitch_the_format_does_not_number_raises(
    export_context: ExportContext, storage: Storage, piano_note: Callable[..., NDArray[np.float64]]
) -> None:
    option = ZoneOption(
        representative=_UNREACHABLE_PITCH,
        params=EncodingParams(11_025, 8, 0.2),
        stored_bytes=100,
        distortion=0.0,
        frames=20,
    )
    zone = Zone(
        pitches=(_UNREACHABLE_PITCH,),
        representative_key=SampleKey(_UNREACHABLE_PITCH, 100),
        weight=1.0,
        chosen=option,
        hull=(option,),
    )
    plan = GroupedInstrumentPlan(
        instrument_id="x",
        budget=BudgetBreakdown(storage=storage, module_bytes=64 * 1024, sample_bytes=63 * 1024),
        velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
        zones=(zone,),
        total_bytes=100,
        objective=0.0,
    )
    audio = {SampleKey(_UNREACHABLE_PITCH, 100): piano_note(60, 100, seed=60 * 200 + 100)}
    with pytest.raises(ValueError, match="outside the IT key range"):
        build_module(plan, audio, SR, [], export_context)


# --- looping ---------------------------------------------------------------------------------------

# 245 Hz has an exact 180-frame period at 44.1 kHz, so a whole-period loop plays it back in tune.
_LOOP_FREQ_HZ = 245.0
_LOOP_RATE = 22_050


def periodic_tone(dur: float = 3.0) -> NDArray[np.float64]:
    time = np.arange(int(dur * SR), dtype=np.float64) / SR
    return 0.7 * np.sin(2.0 * np.pi * _LOOP_FREQ_HZ * time) + 0.2 * np.sin(4.0 * np.pi * _LOOP_FREQ_HZ * time)


@pytest.fixture
def looped_build(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    export_context: ExportContext,
) -> Callable[..., tuple[InstrumentPlan, TrackerModule]]:
    """Optimize a periodic pad with looping forced on, so the stored sample is attack + a short loop."""

    def _looped_build(hold_s: float = 3.0) -> tuple[InstrumentPlan, TrackerModule]:
        audio = {SampleKey(60, 100): periodic_tone(dur=3.0)}
        material = [NoteEvent(pitch=60, velocity=100, duration_s=hold_s, count=1)]
        instrument = InstrumentSpec(
            id="pad",
            budget_kb=64.0,
            samples=[SourceSample(file=Path("60.wav"), pitch=60, velocity=100)],
            material=material,
        )
        settings = optimize_settings(sweep=sweep(rates=(_LOOP_RATE,), depths=(16,), dither=False, loops=(True,)))
        plan = optimize_instrument(instrument, audio, SR, settings)
        return plan, build_module(plan, audio, SR, material, export_context)

    return _looped_build


def test_looped_plan_carries_loop_points_into_the_module(
    looped_build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = looped_build()
    assert plan.pitches[0].chosen.params.loop is True
    sample = module.song.samples[0]
    assert sample.loop is not None
    assert 0 <= sample.loop.begin < sample.loop.end <= sample.frames  # the loop lies inside the stored sample
    assert sample.loop.end < _LOOP_RATE  # storage is attack + a ~0.5 s loop, not the whole 3 s recording


@requires_openmpt
def test_looped_note_sustains_in_openmpt_past_the_stored_length(
    looped_build: Callable[..., tuple[InstrumentPlan, TrackerModule]], render_config: RenderConfig
) -> None:
    hold_s = 3.0
    _, module = looped_build(hold_s=hold_s)  # the stored sample is ~0.1 s; the note is held 3 s
    audio, rate = render_module(module, render_config)
    assert rate == render_config.sample_rate
    # the last second the note is held, long after a sample with no loop would have fallen silent
    held = audio[int((hold_s - 1.0) * rate) : int(hold_s * rate)]
    assert float(np.sqrt(np.mean(held**2))) > 0.05  # the loop keeps the note sounding


@pytest.mark.parametrize("builder", ["build", "grouped_build"])
def test_written_file_round_trips_through_xmodits(builder: str, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    xmodits = pytest.importorskip("xmodits")
    _, module = request.getfixturevalue(builder)()
    path = tmp_path / f"module{module.extension}"
    module.save(path)
    destination = tmp_path / "out"
    destination.mkdir()
    xmodits.dump(str(path), str(destination), format="wav")
    extracted = sorted(destination.glob("*.wav"))
    assert len(extracted) == len(module.song.samples)
    for wav, sample in zip(extracted, module.song.samples):
        data, sample_rate = sf.read(str(wav), dtype="float64", always_2d=False)
        assert np.asarray(data).size == sample.frames
        assert sample_rate == sample.rate  # the tagged rate is the rate the sample really stores
