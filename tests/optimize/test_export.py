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
from optisample.io.it_writer import NOTE_CUT, ITModule, write_it, write_it_module
from optisample.io.render import openmpt123_available, render_module
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.export import ExportContext, build_module, c5speed_for_pitch
from optisample.optimize.grouping import optimize_instrument_grouped
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import BudgetBreakdown, GroupedInstrumentPlan, InstrumentPlan, Zone, ZoneOption
from optisample.optimize.velocity_map import VelocityAnchor, VelocityVolumeMap

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

SR = 44_100
PITCHES = (60, 67)
VELOCITIES = (50, 100)


@pytest.fixture
def demo_audio(piano_note: Callable[..., NDArray[np.float64]]) -> dict[tuple[int, int], NDArray[np.float64]]:
    """The 2x2 demo audio grid (piano notes are fixture-independent test-signal data)."""
    return {(p, v): piano_note(p, v, seed=p * 200 + v) for p in PITCHES for v in VELOCITIES}


def demo_material() -> list[NoteEvent]:
    return [
        NoteEvent(pitch=60, velocity=100, duration_s=0.5, count=8),
        NoteEvent(pitch=60, velocity=50, duration_s=0.5, count=3),
        NoteEvent(pitch=67, velocity=100, duration_s=0.4, count=6),
    ]


def demo_instrument(budget_kb: float = 64.0) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{p}_{v}.wav"), pitch=p, velocity=v) for p in PITCHES for v in VELOCITIES]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=demo_material())


@pytest.fixture
def build(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    export_context: ExportContext,
    demo_audio: dict[tuple[int, int], NDArray[np.float64]],
) -> Callable[..., tuple[InstrumentPlan, ITModule]]:
    """Optimize the demo instrument at a 2x2 grid and export it to an IT module."""

    def _build(material: list[NoteEvent] | None = None) -> tuple[InstrumentPlan, ITModule]:
        settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depths=(16, 8)))
        plan = optimize_instrument(demo_instrument(), demo_audio, SR, settings)
        module = build_module(
            plan, demo_audio, SR, material if material is not None else demo_material(), export_context
        )
        return plan, module

    return _build


def test_one_sample_per_planned_pitch_with_identity_note_map(
    build: Callable[..., tuple[InstrumentPlan, ITModule]],
) -> None:
    plan, module = build()
    assert len(module.samples) == len(plan.pitches)
    assert len(module.instruments) == 1
    note_map = module.instruments[0].note_map
    for index, pitch_plan in enumerate(plan.pitches):
        assert note_map[pitch_plan.pitch] == (pitch_plan.pitch, index + 1)  # key -> (identity note, 1-based sample)


def test_stored_samples_match_the_chosen_operating_points(
    build: Callable[..., tuple[InstrumentPlan, ITModule]],
) -> None:
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


@pytest.mark.parametrize("builder", ["build", "grouped_build"])
def test_sample_bytes_equal_the_budgeted_amount(builder: str, request: pytest.FixtureRequest) -> None:
    plan, module = request.getfixturevalue(builder)()
    sample_bytes = sum(sample.frames * (sample.depth_bits // 8) + 80 for sample in module.samples)
    assert sample_bytes == plan.used_bytes  # PCM + 80-B headers is exactly what the solver budgeted


def test_pattern_applies_velocity_volume_map_and_cuts_each_note(
    build: Callable[..., tuple[InstrumentPlan, ITModule]],
) -> None:
    plan, module = build()
    cells = [cell for pattern in module.patterns for _, _, cell in pattern.cells]
    notes = [(c.note, c.volume) for c in cells if c.note is not None and c.note != NOTE_CUT]
    cuts = [c for c in cells if c.note == NOTE_CUT]
    assert [n for n, _ in notes] == [60, 60, 67]  # material order preserved
    assert notes[0][1] == plan.velocity_map.volume(100)  # loudest velocity -> full volume
    assert notes[1][1] == plan.velocity_map.volume(50)  # quieter event uses the mapped volume
    assert notes[0][1] > notes[1][1]
    assert len(cuts) == 3  # every event is released


def test_long_material_spills_into_multiple_ordered_patterns(
    build: Callable[..., tuple[InstrumentPlan, ITModule]],
) -> None:
    _, module = build([NoteEvent(pitch=60, velocity=100, duration_s=0.5) for _ in range(60)])
    assert len(module.patterns) >= 2  # 60 events * 5 rows > 200-row cap
    assert module.orders == tuple(range(len(module.patterns)))
    assert all(1 <= pattern.rows <= 200 for pattern in module.patterns)


@pytest.mark.parametrize("builder", ["build", "grouped_build"])
def test_build_is_deterministic(builder: str, request: pytest.FixtureRequest) -> None:
    run = request.getfixturevalue(builder)
    _, module_a = run()
    _, module_b = run()
    assert write_it_module(module_a) == write_it_module(module_b)


def test_pitch_out_of_it_range_raises(
    export_context: ExportContext,
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    piano_note: Callable[..., NDArray[np.float64]],
) -> None:
    audio = {(120, 100): piano_note(60, 100, seed=60 * 200 + 100)}
    inst = InstrumentSpec(
        id="x",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("p.wav"), pitch=120, velocity=100)],
        material=[NoteEvent(pitch=120, velocity=100, duration_s=0.4)],
    )
    plan = optimize_instrument(inst, audio, SR, optimize_settings(sweep=sweep(rates=(44_100, 11_025), depths=(16, 8))))
    with pytest.raises(ValueError, match="outside the IT key range"):
        build_module(plan, audio, SR, inst.material or [], export_context)


@pytest.mark.parametrize("builder", ["build", "grouped_build"])
def test_written_file_round_trips_through_xmodits(builder: str, tmp_path: Path, request: pytest.FixtureRequest) -> None:
    xmodits = pytest.importorskip("xmodits")
    _, module = request.getfixturevalue(builder)()
    it_path = tmp_path / "module.it"
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


@pytest.fixture
def grouped_build(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    export_context: ExportContext,
    demo_audio: dict[tuple[int, int], NDArray[np.float64]],
) -> Callable[..., tuple[GroupedInstrumentPlan, ITModule]]:
    """A tight-budget grouped build: one cheap operating point forces both keys into one shared zone."""

    def _grouped_build(budget_kb: float = 8.0) -> tuple[GroupedInstrumentPlan, ITModule]:
        settings = optimize_settings(sweep=sweep(rates=(11_025,), depths=(8,), dither=False))
        plan = optimize_instrument_grouped(demo_instrument(budget_kb), demo_audio, SR, settings)
        module = build_module(plan, demo_audio, SR, demo_material(), export_context)
        return plan, module

    return _grouped_build


def test_grouped_module_shares_one_sample_across_a_merged_zone(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, ITModule]],
) -> None:
    plan, module = grouped_build()
    assert len(plan.zones) == 1  # the tight budget merged both keys into one zone
    assert len(module.samples) == 1  # ... served by a single stored sample
    note_map = module.instruments[0].note_map
    for pitch in plan.zones[0].pitches:
        assert note_map[pitch] == (pitch, 1)  # every covered key -> (its own note, the shared sample)


def test_grouped_sample_c5speed_tracks_the_representative(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, ITModule]],
) -> None:
    plan, module = grouped_build()
    zone = plan.zones[0]
    rate = zone.chosen.params.target_rate
    assert module.samples[0].c5speed == round(rate * 2.0 ** ((60 - zone.representative) / 12.0))


def test_grouped_pitch_out_of_it_range_raises(
    export_context: ExportContext, piano_note: Callable[..., NDArray[np.float64]]
) -> None:
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
        build_module(plan, {(120, 100): piano_note(60, 100, seed=60 * 200 + 100)}, SR, [], export_context)


@requires_openmpt
def test_grouped_module_renders_through_openmpt(
    grouped_build: Callable[..., tuple[GroupedInstrumentPlan, ITModule]], render_config: RenderConfig
) -> None:
    _, module = grouped_build()  # both keys share one repitched sample
    audio, rate = render_module(module, render_config)
    assert rate == render_config.sample_rate
    assert audio.ndim == 1 and audio.size > 0
    assert float(np.max(np.abs(audio))) > 0.0  # the repitched zone actually sounds in the real engine


# --- looping (P6) --------------------------------------------------------------------------------

# 245 Hz has an exact 180-frame period at 44.1 kHz, so a whole-period loop plays it back in tune.


def tone(freq: float = 245.0, dur: float = 3.0) -> NDArray[np.float64]:
    t = np.arange(int(dur * SR), dtype=np.float64) / SR
    return 0.7 * np.sin(2.0 * np.pi * freq * t) + 0.2 * np.sin(2.0 * np.pi * 2.0 * freq * t)


@pytest.fixture
def looped_build(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    export_context: ExportContext,
) -> Callable[..., tuple[InstrumentPlan, ITModule]]:
    """Optimize a periodic pad with looping forced on, so the stored sample is attack + a short loop."""

    def _looped_build(hold_s: float = 3.0) -> tuple[InstrumentPlan, ITModule]:
        audio = {(60, 100): tone(dur=3.0)}
        samples = [SourceSample(file=Path("60.wav"), pitch=60, velocity=100)]
        material = [NoteEvent(pitch=60, velocity=100, duration_s=hold_s, count=1)]
        inst = InstrumentSpec(id="pad", budget_kb=64.0, samples=samples, material=material)
        settings = optimize_settings(sweep=sweep(rates=(22_050,), depths=(16,), dither=False, loops=(True,)))
        plan = optimize_instrument(inst, audio, SR, settings)
        return plan, build_module(plan, audio, SR, material, export_context)

    return _looped_build


def test_looped_plan_carries_loop_points_into_the_module(
    looped_build: Callable[..., tuple[InstrumentPlan, ITModule]],
) -> None:
    plan, module = looped_build()
    assert plan.pitches[0].chosen.params.loop is True
    begin, end = module.samples[0].loop  # type: ignore[misc]
    assert 0 <= begin < end <= module.samples[0].frames  # the loop lies inside the stored sample
    assert end < 22_050  # storage is attack + a ~0.5 s loop, not the whole 3 s recording


@requires_openmpt
def test_looped_note_sustains_in_openmpt_past_the_stored_length(
    looped_build: Callable[..., tuple[InstrumentPlan, ITModule]], render_config: RenderConfig
) -> None:
    _, module = looped_build(hold_s=3.0)  # the stored sample is ~0.1 s; the note is held 3 s
    audio, rate = render_module(module, render_config)
    assert rate == render_config.sample_rate
    tail = audio[-rate:]  # the final second, long after a non-looping sample would have fallen silent
    assert float(np.sqrt(np.mean(tail**2))) > 0.05  # the loop keeps the note sounding
