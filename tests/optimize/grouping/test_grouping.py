from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.io.audio import write_wav
from optisample.io.tracker.target import export_target
from optisample.metrics.composite import build_composite
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping import (
    build_zone_options,
    optimize_instrument_grouped,
    run_instrument_grouped,
    zone_hull,
)
from optisample.optimize.grouping.cost_model import _zone_trim
from optisample.optimize.orchestrate import optimize_instrument, prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, ZoneOption
from optisample.optimize.tasks import Event, PitchTask
from optisample.synth import NoteSpec, render_sample

SR = 44_100
PITCHES = (60, 62, 64)

_CONFIG = load_config()
_COMPOSITE = build_composite(_CONFIG.metrics)


def _grid(**overrides: object) -> SweepConfig:
    return SweepConfig.model_validate({**_CONFIG.sweep.model_dump(), **overrides})


def _settings(sweep: SweepConfig) -> OptimizeSettings:
    return OptimizeSettings(
        sweep=sweep,
        reduce=_CONFIG.reduce,
        encode=_CONFIG.encode,
        composite=_COMPOSITE,
        velocity=_CONFIG.velocity,
        method=_CONFIG.optimize.method,
        target=export_target(_CONFIG.tracker),
    )


GRID = _grid(rates=(44_100, 11_025), depths=(16, 8), dither=False)
GRID_TINY = _grid(rates=(11_025,), depths=(8,), dither=False)


def _note(pitch: int, velocity: int, dur: float) -> NDArray[np.float64]:
    return render_sample(
        "piano",
        NoteSpec(pitch, velocity, 0.0, dur, SR),
        np.random.default_rng(pitch * 137 + velocity),
        _CONFIG.synth,
    )


def _audio() -> dict[tuple[int, int], NDArray[np.float64]]:
    return {(pitch, 100): _note(pitch, 100, 0.6) for pitch in PITCHES}


def _material() -> list[NoteEvent]:
    return [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=4) for pitch in PITCHES]


def _instrument(budget_kb: float) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in PITCHES]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=_material())


# Module-scoped fixtures: the zone-option build (an encode + composite score per representative,
# encoding and covered pitch) is the expensive step, so compute the shared ones exactly once.


@pytest.fixture(scope="module")
def audio() -> dict[tuple[int, int], NDArray[np.float64]]:
    return _audio()


@pytest.fixture(scope="module")
def options48(
    audio: dict[tuple[int, int], NDArray[np.float64]],
) -> tuple[list[PitchTask], dict[tuple[int, int], tuple[ZoneOption, ...]]]:
    _, context, tasks = prepare_run(_instrument(48.0), audio, SR, _settings(GRID))
    return tasks, build_zone_options(tasks, context)


@pytest.fixture(scope="module")
def plan48(audio: dict[tuple[int, int], NDArray[np.float64]]) -> GroupedInstrumentPlan:
    return optimize_instrument_grouped(_instrument(48.0), audio, SR, _settings(GRID))


# --- zone cost model -----------------------------------------------------------------------------


def test_zone_trim_scales_with_upward_transpose() -> None:
    silence = np.zeros(4, dtype=np.float64)
    tasks = [
        PitchTask(60, 1.0, 100, silence, (Event(100, 0.5, 1.0, silence),)),
        PitchTask(72, 1.0, 100, silence, (Event(100, 0.5, 1.0, silence),)),
    ]
    assert _zone_trim(tasks, 60) == pytest.approx(1.0)  # bottom rep must stretch an octave up (2x length)
    assert _zone_trim(tasks, 72) == pytest.approx(0.5)  # top rep plays the low key slower -> its own length


def test_build_zone_options_covers_every_range_and_every_member_as_representative(
    options48: tuple[list[PitchTask], dict[tuple[int, int], tuple[ZoneOption, ...]]],
) -> None:
    tasks, options = options48
    count = len(tasks)
    assert set(options) == {(i, j) for i in range(count) for j in range(i + 1, count + 1)}
    for (i, j), zone_options in options.items():
        assert {option.representative for option in zone_options} == {task.pitch for task in tasks[i:j]}


def test_zone_hull_is_a_monotone_frontier(
    options48: tuple[list[PitchTask], dict[tuple[int, int], tuple[ZoneOption, ...]]],
) -> None:
    tasks, options = options48
    hull = zone_hull(options[(0, len(tasks))])  # the widest zone: all keys from one representative
    assert len(hull) >= 1
    assert all(a.stored_bytes < b.stored_bytes for a, b in zip(hull, hull[1:]))  # ascending bytes
    assert all(a.distortion > b.distortion for a, b in zip(hull, hull[1:]))  # descending distortion


# --- end-to-end behaviour ------------------------------------------------------------------------


def test_grouping_is_never_worse_than_ungrouped_at_a_feasible_budget(
    audio: dict[tuple[int, int], NDArray[np.float64]], plan48: GroupedInstrumentPlan
) -> None:
    ungrouped = optimize_instrument(_instrument(48.0), audio, SR, _settings(GRID))
    assert plan48.objective <= ungrouped.objective + 1e-9  # singletons are always in the search space
    assert plan48.used_bytes <= plan48.sample_budget_bytes


def test_generous_budget_keeps_one_sample_per_key(plan48: GroupedInstrumentPlan) -> None:
    assert len(plan48.zones) == len(PITCHES)  # no reason to merge when every key fits
    assert all(len(zone.pitches) == 1 and zone.representative == zone.pitches[0] for zone in plan48.zones)


def test_zones_partition_all_pitches_and_bytes_add_up(plan48: GroupedInstrumentPlan) -> None:
    covered = [pitch for zone in plan48.zones for pitch in zone.pitches]
    assert covered == list(PITCHES)  # ascending, contiguous, no gaps or overlaps
    assert all(zone.representative in zone.pitches for zone in plan48.zones)
    assert plan48.used_bytes == sum(zone.chosen.stored_bytes for zone in plan48.zones)


def test_grouping_is_feasible_where_ungrouped_is_not(audio: dict[tuple[int, int], NDArray[np.float64]]) -> None:
    settings = _settings(GRID_TINY)
    inst = _instrument(10.0)  # room for one shared sample, not for three separate ones
    with pytest.raises(BudgetInfeasibleError):
        optimize_instrument(inst, audio, SR, settings)
    grouped = optimize_instrument_grouped(inst, audio, SR, settings)
    assert len(grouped.zones) < len(PITCHES)  # keys had to be merged to fit
    assert set(grouped.pitches) == set(PITCHES)  # but every key is still covered
    assert grouped.used_bytes <= grouped.sample_budget_bytes


def test_grouping_raises_when_even_one_merged_zone_overflows(audio: dict[tuple[int, int], NDArray[np.float64]]) -> None:
    with pytest.raises(BudgetInfeasibleError):
        optimize_instrument_grouped(_instrument(1.0), audio, SR, _settings(GRID_TINY))


def test_grouping_handles_the_sustained_archetype() -> None:
    pitches = (60, 62, 64)
    audio = {
        (pitch, 100): render_sample(
            "sustained", NoteSpec(pitch, 100, 0.0, 0.6, SR), np.random.default_rng(pitch), _CONFIG.synth
        )
        for pitch in pitches
    }
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in pitches]
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=3) for pitch in pitches]
    inst = InstrumentSpec(id="strings", budget_kb=10.0, samples=samples, material=material)
    grouped = optimize_instrument_grouped(inst, audio, SR, _settings(GRID_TINY))
    assert set(grouped.pitches) == set(pitches)  # the pad archetype groups the same way the piano does
    assert len(grouped.zones) < len(pitches)  # tight budget forces at least one merge
    assert grouped.used_bytes <= grouped.sample_budget_bytes


def test_run_instrument_grouped_reads_wavs_from_disk(tmp_path: Path) -> None:
    samples = []
    for pitch in PITCHES:
        path = tmp_path / f"p{pitch}.wav"
        write_wav(path, _note(pitch, 100, 0.6), SR)
        samples.append(SourceSample(file=path, pitch=pitch, velocity=100))
    inst = InstrumentSpec(id="piano", budget_kb=48.0, samples=samples, material=_material())
    grouped = run_instrument_grouped(inst, _settings(GRID_TINY))
    assert set(grouped.pitches) == set(PITCHES)
    assert grouped.used_bytes <= grouped.sample_budget_bytes
