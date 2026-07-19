from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from numpy.typing import NDArray

from optisample.dsp.surrogate import EncodingParams
from optisample.io.it_writer import NOTE_CUT, ITModule, write_it, write_it_module
from optisample.io.render import openmpt123_available, render_module
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.export import build_grouped_it_module, build_it_module, c5speed_for_pitch
from optisample.optimize.grouping import GroupedInstrumentPlan, Zone, ZoneOption, optimize_instrument_grouped
from optisample.optimize.operating_points import SweepGrid
from optisample.optimize.orchestrate import BudgetBreakdown, InstrumentPlan, OptimizeSettings, optimize_instrument
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap
from optisample.synth import NoteSpec, render_sample

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

SR = 44_100
GRID = SweepGrid(rates=(44_100, 11_025), depths=(16, 8))
PITCHES = (60, 67)
VELOCITIES = (50, 100)


def note(pitch: int, velocity: int, dur: float = 0.6) -> NDArray[np.float64]:
    return render_sample(
        "piano", NoteSpec(pitch, velocity, 0.0, dur, SR), np.random.default_rng(pitch * 200 + velocity)
    )


def demo_audio() -> dict[tuple[int, int], NDArray[np.float64]]:
    return {(p, v): note(p, v) for p in PITCHES for v in VELOCITIES}


def demo_material() -> list[NoteEvent]:
    return [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=8),
        NoteEvent(pitch=60, velocity=50, duration_s=0.5, count=3),
        NoteEvent(pitch=67, velocity=100, duration_s=0.4, count=6),
    ]


def demo_instrument(budget_kb: float = 64.0) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{p}_{v}.wav"), pitch=p, velocity=v) for p in PITCHES for v in VELOCITIES]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=demo_material())


def build(material: list[NoteEvent] | None = None) -> tuple[InstrumentPlan, ITModule]:
    audio = demo_audio()
    plan = optimize_instrument(demo_instrument(), audio, SR, OptimizeSettings(grid=GRID))
    module = build_it_module(plan, audio, SR, material if material is not None else demo_material())
    return plan, module


def test_one_sample_per_planned_pitch_with_identity_note_map() -> None:
    plan, module = build()
    assert len(module.samples) == len(plan.pitches)
    assert len(module.instruments) == 1
    note_map = module.instruments[0].note_map
    for index, pitch_plan in enumerate(plan.pitches):
        assert note_map[pitch_plan.pitch] == (pitch_plan.pitch, index + 1)  # key -> (identity note, 1-based sample)


def test_stored_samples_match_the_chosen_operating_points() -> None:
    plan, module = build()
    for pitch_plan, sample in zip(plan.pitches, module.samples):
        assert sample.depth_bits == pitch_plan.chosen.params.depth_bits
        assert sample.frames == pitch_plan.chosen.frames
        rate = pitch_plan.chosen.params.target_rate
        assert sample.c5speed == round(rate * 2.0 ** ((60 - pitch_plan.pitch) / 12.0))


def test_c5speed_makes_the_root_key_play_natural() -> None:
    assert c5speed_for_pitch(11_025, 60) == 11_025  # C-5 itself: no transpose
    assert c5speed_for_pitch(11_025, 48) == 22_050  # an octave down the keyboard doubles the rate
    assert c5speed_for_pitch(11_025, 72) == 5_512  # an octave up halves it


def test_sample_bytes_equal_the_budgeted_amount() -> None:
    plan, module = build()
    sample_bytes = sum(sample.frames * (sample.depth_bits // 8) + 80 for sample in module.samples)
    assert sample_bytes == plan.used_bytes  # PCM + 80-B headers is exactly what the solver budgeted


def test_pattern_applies_velocity_volume_map_and_cuts_each_note() -> None:
    plan, module = build()
    cells = [cell for pattern in module.patterns for _, _, cell in pattern.cells]
    notes = [(c.note, c.volume) for c in cells if c.note is not None and c.note != NOTE_CUT]
    cuts = [c for c in cells if c.note == NOTE_CUT]
    assert [n for n, _ in notes] == [60, 60, 67]  # material order preserved
    assert notes[0][1] == plan.velocity_map.volume(100)  # loudest velocity -> full volume
    assert notes[1][1] == plan.velocity_map.volume(50)  # quieter event uses the mapped volume
    assert notes[0][1] > notes[1][1]
    assert len(cuts) == 3  # every event is released


def test_long_material_spills_into_multiple_ordered_patterns() -> None:
    _, module = build([NoteEvent(pitch=60, velocity=100, duration_s=0.5) for _ in range(60)])
    assert len(module.patterns) >= 2  # 60 events * 5 rows > 200-row cap
    assert module.orders == tuple(range(len(module.patterns)))
    assert all(1 <= pattern.rows <= 200 for pattern in module.patterns)


def test_build_is_deterministic() -> None:
    _, module_a = build()
    _, module_b = build()
    assert write_it_module(module_a) == write_it_module(module_b)


def test_pitch_out_of_it_range_raises() -> None:
    audio = {(120, 100): note(60, 100)}
    inst = InstrumentSpec(
        id="x",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("p.wav"), pitch=120, velocity=100)],
        material=[NoteEvent(pitch=120, velocity=100, duration_s=0.4)],
    )
    plan = optimize_instrument(inst, audio, SR, OptimizeSettings(grid=GRID))
    with pytest.raises(ValueError, match="outside the IT key range"):
        build_it_module(plan, audio, SR, inst.material or [])


def test_written_file_round_trips_through_xmodits(tmp_path: Path) -> None:
    xmodits = pytest.importorskip("xmodits")
    _, module = build()
    it_path = tmp_path / "piano.it"
    write_it(it_path, module)
    dest = tmp_path / "out"
    dest.mkdir()
    xmodits.dump(str(it_path), str(dest), format="wav")
    extracted = sorted(dest.glob("*.wav"))
    assert len(extracted) == len(module.samples)
    for path, sample in zip(extracted, module.samples):
        data, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
        assert np.asarray(data).size == sample.frames
        assert sample_rate == sample.c5speed  # xmodits tags the WAV with the sample's C5Speed


# --- grouped export (one repitched sample per zone) ----------------------------------------------

# A single cheap operating point and a tight budget, so the two keys are forced into one shared zone.
GRID_G = SweepGrid(rates=(11_025,), depths=(8,), dither=False)


def grouped_build(budget_kb: float = 8.0) -> tuple[GroupedInstrumentPlan, ITModule]:
    audio = demo_audio()
    plan = optimize_instrument_grouped(demo_instrument(budget_kb), audio, SR, OptimizeSettings(grid=GRID_G))
    module = build_grouped_it_module(plan, audio, SR, demo_material())
    return plan, module


def test_grouped_module_shares_one_sample_across_a_merged_zone() -> None:
    plan, module = grouped_build()
    assert len(plan.zones) == 1  # the tight budget merged both keys into one zone
    assert len(module.samples) == 1  # ... served by a single stored sample
    note_map = module.instruments[0].note_map
    for pitch in plan.zones[0].pitches:
        assert note_map[pitch] == (pitch, 1)  # every covered key -> (its own note, the shared sample)


def test_grouped_sample_c5speed_tracks_the_representative() -> None:
    plan, module = grouped_build()
    zone = plan.zones[0]
    rate = zone.chosen.params.target_rate
    assert module.samples[0].c5speed == round(rate * 2.0 ** ((60 - zone.representative) / 12.0))


def test_grouped_sample_bytes_equal_the_budgeted_amount() -> None:
    plan, module = grouped_build()
    sample_bytes = sum(sample.frames * (sample.depth_bits // 8) + 80 for sample in module.samples)
    assert sample_bytes == plan.used_bytes


def test_grouped_build_is_deterministic() -> None:
    _, module_a = grouped_build()
    _, module_b = grouped_build()
    assert write_it_module(module_a) == write_it_module(module_b)


def test_grouped_pitch_out_of_it_range_raises() -> None:
    option = ZoneOption(
        representative=120, params=EncodingParams(11_025, 8, 0.2), stored_bytes=100, distortion=0.0, frames=20
    )
    zone = Zone(
        pitches=(120,), representative=120, representative_velocity=100, weight=1.0, chosen=option, hull=(option,)
    )
    plan = GroupedInstrumentPlan(
        instrument_id="x",
        budget=BudgetBreakdown(module_bytes=64 * 1024, sample_bytes=64 * 1024 - 746),
        velocity_map=VelocityVolumeMap(tuple(64 for _ in range(128)), (VelocityAnchor(100, -10.0, 64),)),
        zones=(zone,),
        total_bytes=100,
        objective=0.0,
    )
    with pytest.raises(ValueError, match="outside the IT key range"):
        build_grouped_it_module(plan, {(120, 100): note(60, 100)}, SR, [])


def test_grouped_round_trips_through_xmodits(tmp_path: Path) -> None:
    xmodits = pytest.importorskip("xmodits")
    _, module = grouped_build()
    it_path = tmp_path / "grouped.it"
    write_it(it_path, module)
    dest = tmp_path / "out"
    dest.mkdir()
    xmodits.dump(str(it_path), str(dest), format="wav")
    assert len(sorted(dest.glob("*.wav"))) == len(module.samples)


@requires_openmpt
def test_grouped_module_renders_through_openmpt() -> None:
    _, module = grouped_build()  # both keys share one repitched sample
    audio, rate = render_module(module)
    assert rate == 48_000
    assert audio.ndim == 1 and audio.size > 0
    assert float(np.max(np.abs(audio))) > 0.0  # the repitched zone actually sounds in the real engine
