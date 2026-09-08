from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray
from trackmod import BitDepth, TrackerModule
from trackmod.core.instruments.keymap import KeyAssignment
from trackmod.core.notes.pitch import Note
from trackmod.module.storage import Storage
from trackmod.spec.levels import MAX_VOLUME
from trackmod.spec.pitch import RATE_NOTE
from trackmod.trackers.xm.spec.sizes import NAME_BYTES as _NARROWEST_NAME_BYTES

from optisample.config.optimize import SweepConfig
from optisample.config.render import RenderConfig
from optisample.config.tracker import TrackerFormat
from optisample.dsp.surrogate import EncodingParams, StoredSample
from optisample.io.render import openmpt123_available, render_module
from optisample.io.tracker.target import ExportTarget
from optisample.io.tracker.voices import routed_voices
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.music import MIDI_MAX_VELOCITY
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.samples import sample_gains, sample_name
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers
from optisample.optimize.layers.slots import ONE_SLOT
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import (
    FIRST_LAYER,
    NO_RESERVE,
    BudgetBreakdown,
    GroupedInstrumentPlan,
    InstrumentPlan,
    SampleReserve,
    SampleUnit,
    Zone,
    ZoneOption,
)
from optisample.optimize.reduce.summary import ReductionSummary
from optisample.optimize.tasks import StoredRecordings
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from tests.optimize.export.demo import SR

Recordings = Callable[..., StoredRecordings]

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

_UNREACHABLE_PITCH = 5  # below MIDI 12, so no tracker keyboard numbers it
_WHOLE_TABLE = 255  # samples a format numbering them freely lets one plan store
_UNCHARGED = SampleReserve(cap=_WHOLE_TABLE, bytes_per_sample=NO_RESERVE, objective_uncapped=0.0)
_LOUDEST_VELOCITY = 100  # the velocity a graded map puts at full volume
_SOFTER_VELOCITY = 50  # ... and one it puts at half, so the pattern carries a dynamic of its own


def test_one_sample_per_planned_pitch_each_sounding_its_own_key(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = build()
    song = module.song
    assert len(routed_voices(song).samples) == len(plan.pitches)
    assert len(routed_voices(song).instruments) == 1
    for index, pitch_plan in enumerate(plan.pitches):
        assignment = routed_voices(song).instruments[0].assignment(Note.from_midi(pitch_plan.pitch))
        assert assignment is not None
        assert assignment.sample == index
        # a key playing its own recording needs no transposition, so it sounds the reference note
        assert assignment.note == Note(RATE_NOTE)


def test_stored_samples_match_the_chosen_operating_points(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = build()
    for pitch_plan, sample in zip(plan.pitches, routed_voices(module.song).samples):
        assert sample.depth == pitch_plan.chosen.params.depth
        assert sample.frames == pitch_plan.chosen.frames
        assert sample.rate == pitch_plan.chosen.params.target_rate  # the true stored rate, untransposed


def test_sample_names_carry_the_instrument_and_the_recording_stored(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
    target: ExportTarget,
) -> None:
    plan, module = build()
    assert [sample.name for sample in routed_voices(module.song).samples] == [
        sample_name(plan.instrument_id, unit, target) for unit in plan.sample_units()
    ]
    assert [sample.name for sample in routed_voices(module.song).samples] == ["piano C4 v100", "piano G4 v100"]


def test_sample_names_tell_one_key_s_velocity_layers_apart(
    layered_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    """A key stores a recording per band, so naming the recording is what keeps the two listed apart."""
    _, module = layered_build()
    names = [sample.name for sample in routed_voices(module.song).samples]
    assert len(set(names)) == len(names)
    assert all(len(name) <= _NARROWEST_NAME_BYTES for name in names)


@pytest.mark.parametrize("builder", ["build", "grouped_build"])
def test_sample_bytes_equal_the_budgeted_amount(builder: str, request: pytest.FixtureRequest) -> None:
    plan, module = request.getfixturevalue(builder)()
    storage = module.storage
    stored = sum(storage.sample_bytes(frames=s.frames, depth=s.depth) for s in routed_voices(module.song).samples)
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
    recordings: Recordings,
) -> None:
    audio = {SampleKey(_UNREACHABLE_PITCH, 100): piano_note(60, 100, seed=60 * 200 + 100)}
    instrument = InstrumentSpec(
        id="x",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("p.wav"), pitch=_UNREACHABLE_PITCH, velocity=100)],
        material=[NoteEvent(pitch=_UNREACHABLE_PITCH, velocity=100, duration_s=0.4)],
    )
    settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depth=BitDepth.SIXTEEN))
    plan = optimize_instrument(instrument, recordings(audio, SR), settings)
    with pytest.raises(ValueError, match="outside the IT key range"):
        build_module(plan, recordings(audio, SR), instrument.material or [], export_context)


def _encoded(gain: float, velocity: int = _LOUDEST_VELOCITY) -> tuple[SampleUnit, StoredSample]:
    """A stand-in encoded unit, carrying only the recorded velocity and the scaling the export reads."""
    unit = SampleUnit(
        label="unit",
        representative_key=SampleKey(60, velocity),
        layer=FIRST_LAYER,
        keys=(60,),
        params=EncodingParams(22_050, BitDepth.EIGHT),
        frames=8,
        stored_bytes=8,
        distortion=0.0,
        objective_share=0.0,
        hull_size=1,
        weight=1.0,
    )
    stored = StoredSample(
        pcm=np.zeros(8, dtype=np.float64), sample_rate=22_050, depth=BitDepth.EIGHT, root_pitch=60, gain=gain
    )
    return unit, stored


@pytest.fixture
def graded_map() -> VelocityVolumeMap:
    """A map putting the softer of two velocities at half volume, so a test can see it applied twice."""
    volumes = tuple(MAX_VOLUME if velocity >= _LOUDEST_VELOCITY else MAX_VOLUME // 2 for velocity in range(128))
    return VelocityVolumeMap(volumes, (VelocityAnchor(_LOUDEST_VELOCITY, -10.0, MAX_VOLUME),))


def test_the_sample_needing_most_of_the_gain_takes_the_top_step(
    graded_map: VelocityVolumeMap, target: ExportTarget
) -> None:
    """The balance storing hot flattens: a recording lifted twice as far comes back half as loud."""
    encoded = [_encoded(2.0), _encoded(4.0), _encoded(8.0)]
    assert sample_gains(encoded, graded_map, target) == (MAX_VOLUME, 32, 16)


def test_a_sample_recorded_at_a_softer_velocity_is_not_turned_down_twice(
    graded_map: VelocityVolumeMap, target: ExportTarget
) -> None:
    """Its notes already carry that dynamic in the pattern, so charging it here as well would double it."""
    loud, soft = _encoded(1.0), _encoded(2.0, velocity=_SOFTER_VELOCITY)
    assert sample_gains([loud, soft], graded_map, target) == (MAX_VOLUME, MAX_VOLUME)


def test_a_representative_the_map_silences_keeps_its_own_scaling(
    flat_velocity_map: VelocityVolumeMap, target: ExportTarget
) -> None:
    """The pattern silences that note whatever gain the sample carries, so the division is left out."""
    silent = VelocityVolumeMap(tuple(0 for _ in range(128)), flat_velocity_map.anchors)
    assert sample_gains([_encoded(1.0), _encoded(4.0)], silent, target) == (MAX_VOLUME, 16)


def test_a_recording_far_under_the_loudest_is_still_heard(
    flat_velocity_map: VelocityVolumeMap, target: ExportTarget
) -> None:
    """The format's 64 steps run out before a wide instrument does, so the quietest keeps the softest one."""
    assert sample_gains([_encoded(1.0), _encoded(10_000.0)], flat_velocity_map, target)[1] == 1


def test_a_format_pinning_its_gain_writes_every_sample_at_full(
    flat_velocity_map: VelocityVolumeMap, retarget: Callable[[TrackerFormat], ExportTarget]
) -> None:
    """FastTracker 2 has no per-sample multiplier to grade, so the PCM carries the balance instead."""
    encoded = [_encoded(2.0), _encoded(8.0)]
    assert sample_gains(encoded, flat_velocity_map, retarget(TrackerFormat.XM)) == (MAX_VOLUME, MAX_VOLUME)


def test_an_instrument_of_no_samples_needs_no_gains(flat_velocity_map: VelocityVolumeMap, target: ExportTarget) -> None:
    assert sample_gains([], flat_velocity_map, target) == ()


def test_the_written_module_grades_its_samples_against_the_one_needing_most(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    """End to end: the plan's samples reach the module carrying the balance the recordings were made with."""
    _, module = build()
    gains = [sample.gain for sample in routed_voices(module.song).samples]
    assert max(gains) == MAX_VOLUME
    assert all(0 < gain <= MAX_VOLUME for gain in gains)


def _assignment(module: TrackerModule, target: ExportTarget, pitch: int) -> KeyAssignment | None:
    """What the written instrument plays at ``pitch``, as the module itself states it."""
    return routed_voices(module.song).instruments[0].assignment(target.key(pitch))


def test_a_written_instrument_answers_far_past_the_keys_it_was_recorded_over(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
    target: ExportTarget,
) -> None:
    """End to end: a note reaching a key the material never played sounds the recording nearest it."""
    plan, module = build()
    answered = [pitch for pitch in range(target.min_pitch, target.max_pitch + 1) if _assignment(module, target, pitch)]
    played = sorted(pitch_plan.pitch for pitch_plan in plan.pitches)
    assert set(played) <= set(answered)  # every recording still answers its own key
    assert answered == list(range(answered[0], answered[-1] + 1))  # one unbroken stretch, no key left inside it
    assert answered[0] < played[0] and answered[-1] > played[-1]  # reaching past the keys that were recorded


def test_a_filled_key_sounds_its_sample_at_the_same_offset_its_own_keys_do(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
    target: ExportTarget,
) -> None:
    """One sample is tuned once for every key reaching it, which is what FastTracker 2 asks of it."""
    _, module = build()
    offsets: dict[int, set[int]] = {}
    for pitch in range(target.min_pitch, target.max_pitch + 1):
        assignment = _assignment(module, target, pitch)
        if assignment is not None:
            offsets.setdefault(assignment.sample, set()).add(target.key(pitch).value - assignment.note.value)

    assert all(len(spread) == 1 for spread in offsets.values())


def test_grouped_module_shares_one_sample_across_a_merged_zone(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, module = grouped_build()
    assert len(plan.zones) == 1  # the tight budget merged both keys into one zone
    assert len(routed_voices(module.song).samples) == 1  # ... served by a single stored sample
    representative = plan.zones[0].representative
    for pitch in plan.zones[0].pitches:
        assignment = routed_voices(module.song).instruments[0].assignment(Note.from_midi(pitch))
        assert assignment is not None
        assert assignment.sample == 0  # every covered key reaches the shared sample
        assert assignment.note == Note(RATE_NOTE + pitch - representative)  # transposed from the representative


def test_grouped_sample_keeps_the_representatives_stored_rate(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, TrackerModule]],
) -> None:
    plan, module = grouped_build()
    assert routed_voices(module.song).samples[0].rate == plan.zones[0].chosen.params.target_rate


def test_a_grouped_pitch_the_format_does_not_number_raises(
    export_context: ExportContext,
    storage: Storage,
    piano_note: Callable[..., NDArray[np.float64]],
    reduction: ReductionSummary,
    recordings: Recordings,
) -> None:
    option = ZoneOption(
        representative=_UNREACHABLE_PITCH,
        params=EncodingParams(11_025, BitDepth.EIGHT, 0.2),
        stored_bytes=100,
        distortion=0.0,
        frames=20,
    )
    zone = Zone(
        pitches=(_UNREACHABLE_PITCH,),
        layer=FIRST_LAYER,
        representative_key=SampleKey(_UNREACHABLE_PITCH, 100),
        weight=1.0,
        chosen=option,
        hull=(option,),
    )
    plan = GroupedInstrumentPlan(
        instrument_id="x",
        budget=BudgetBreakdown(storage=storage, instruments=ONE_SLOT, module_bytes=64 * 1024, sample_bytes=63 * 1024),
        velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
        layers=VelocityLayers((VelocityBand(0, MIDI_MAX_VELOCITY),)),
        zones=(zone,),
        total_bytes=100,
        objective=0.0,
        reserve=_UNCHARGED,
        energy_exponent=0.5,
        reduction=reduction,
    )
    audio = {SampleKey(_UNREACHABLE_PITCH, 100): piano_note(60, 100, seed=60 * 200 + 100)}
    with pytest.raises(ValueError, match="outside the IT key range"):
        build_module(plan, recordings(audio, SR), [], export_context)


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
    recordings: Recordings,
) -> Callable[..., tuple[InstrumentPlan, TrackerModule]]:
    """Optimize a periodic pad whose budget only a loop fits, so the stored sample is attack + one loop."""

    def _looped_build(hold_s: float = 3.0) -> tuple[InstrumentPlan, TrackerModule]:
        audio = {SampleKey(60, 100): periodic_tone(dur=3.0)}
        material = [NoteEvent(pitch=60, velocity=100, duration_s=hold_s, count=1)]
        instrument = InstrumentSpec(
            id="pad",
            budget_kb=64.0,
            samples=[SourceSample(file=Path("60.wav"), pitch=60, velocity=100)],
            material=material,
        )
        settings = optimize_settings(sweep=sweep(rates=(_LOOP_RATE,), depth=BitDepth.SIXTEEN, dither=False))
        plan = optimize_instrument(instrument, recordings(audio, SR), settings)
        return plan, build_module(plan, recordings(audio, SR), material, export_context)

    return _looped_build


def test_looped_plan_carries_loop_points_into_the_module(
    looped_build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    plan, module = looped_build()
    assert plan.pitches[0].chosen.params.loop_index is not None  # 3 s stored whole overruns the budget
    sample = routed_voices(module.song).samples[0]
    assert sample.loop is not None
    assert 0 <= sample.loop.begin < sample.loop.end <= sample.frames  # the loop lies inside the stored sample
    assert sample.loop.end / sample.rate < 1.0  # storage is attack + a ~0.5 s loop, not the whole 3 s recording


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
    assert len(extracted) == len(routed_voices(module.song).samples)
    for wav, sample in zip(extracted, routed_voices(module.song).samples):
        data, sample_rate = sf.read(str(wav), dtype="float64", always_2d=False)
        assert np.asarray(data).size == sample.frames
        assert sample_rate == sample.rate  # the tagged rate is the rate the sample really stores
