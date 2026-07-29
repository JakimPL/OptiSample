from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.audio import write_wav
from optisample.io.tracker.target import export_target
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.dp import BudgetInfeasibleError
from optisample.optimize.grouping import build_zone_options, zone_hull
from optisample.optimize.grouping.cost_model import (
    ZoneSegment,
    _Candidate,
    _capped_ranges,
    _GridCache,
    _zone_delta,
    _zone_demand,
    _zone_trim,
    _ZoneOptions,
    candidate_zones,
    store_requests,
    zone_shortlists,
    zone_starts,
)
from optisample.optimize.grouping.optimize import (
    optimize_instrument_grouped,
    run_instrument_grouped,
)
from optisample.optimize.grouping.stores import dither, score_request, score_stores
from optisample.optimize.orchestrate import optimize_instrument, prepare_run
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import GroupedInstrumentPlan, ZoneOption
from optisample.optimize.reduce.bandwidth import ClipDemand
from optisample.optimize.reduce.keys import SampleKey
from optisample.optimize.tasks import EvalContext, Event, PitchTask
from optisample.parallel import IN_PROCESS
from optisample.progress import NO_PROGRESS
from optisample.synth import NoteSpec, synthesize

SR = 44_100
PITCHES = (60, 62, 64)
_OCTAVE = 12
_SHARED = 2  # workers, enough to score the representatives apart without asking the machine for every core
_PER_KEY_BYTES = 1_000  # the share of the budget one key brings, which a zone's keys pool behind its sample

_CONFIG = load_config()


def _grid(**overrides: object) -> SweepConfig:
    return SweepConfig.model_validate({**_CONFIG.optimize.sweep.model_dump(), **overrides})


def _reduce(**sections: Mapping[str, object]) -> ReduceConfig:
    """The bundled reduction with the named sections' fields overridden, re-validated."""
    raw: dict[str, dict[str, object]] = _CONFIG.reduce.model_dump()
    return ReduceConfig.model_validate({name: {**fields, **sections.get(name, {})} for name, fields in raw.items()})


def _settings(sweep: SweepConfig, **sections: Mapping[str, object]) -> OptimizeSettings:
    return OptimizeSettings(
        sweep=sweep,
        reduce=_reduce(**sections),
        layers=_CONFIG.optimize.layers,
        encode=_CONFIG.codec.encode,
        metrics=_CONFIG.analysis.metrics,
        velocity=_CONFIG.optimize.velocity,
        method=_CONFIG.optimize.budget.method,
        energy_exponent=_CONFIG.optimize.budget.energy_exponent,
        max_samples=_CONFIG.optimize.budget.max_samples,
        target=export_target(_CONFIG.export.tracker),
    )


def _one_layer(tasks: Sequence[PitchTask]) -> tuple[ZoneSegment, ...]:
    """The whole keyboard as a single velocity layer -- the axis grouping alone segments over."""
    return (ZoneSegment(tuple(tasks), _PER_KEY_BYTES),)


GRID = _grid(rates=(44_100, 11_025), depths=(16, 8), dither=False)
GRID_TINY = _grid(rates=(11_025,), depths=(8,), dither=False)
GRID_DITHERED = _grid(rates=(11_025,), depths=(8,), dither=True)  # a grid whose encodes draw noise


def _note(pitch: int, velocity: int, dur: float) -> NDArray[np.float64]:
    return synthesize(
        "piano",
        NoteSpec(pitch, velocity, 0.0, dur, SR),
        np.random.default_rng(pitch * 137 + velocity),
        _CONFIG.synth,
    )


def _audio() -> dict[SampleKey, NDArray[np.float64]]:
    return {SampleKey(pitch, 100): _note(pitch, 100, 0.6) for pitch in PITCHES}


def _material() -> list[NoteEvent]:
    return [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=4) for pitch in PITCHES]


def _instrument(budget_kb: float) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in PITCHES]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=_material())


# Module-scoped fixtures: the zone-option build (an encode + composite score per representative,
# encoding and covered pitch) is the expensive step, so compute the shared ones exactly once.


@pytest.fixture(scope="module")
def audio() -> dict[SampleKey, NDArray[np.float64]]:
    return _audio()


@pytest.fixture(scope="module")
def options48(
    audio: dict[SampleKey, NDArray[np.float64]],
) -> tuple[list[PitchTask], dict[tuple[int, int], tuple[ZoneOption, ...]]]:
    inputs = prepare_run(_instrument(48.0), audio, SR, _settings(GRID))
    segments = _one_layer(inputs.tasks)
    (options,) = build_zone_options(segments, inputs.context, workers=IN_PROCESS, progress=NO_PROGRESS)
    return list(inputs.tasks), options


@pytest.fixture(scope="module")
def plan48(audio: dict[SampleKey, NDArray[np.float64]]) -> GroupedInstrumentPlan:
    return optimize_instrument_grouped(_instrument(48.0), audio, SR, _settings(GRID))


@pytest.fixture(scope="module")
def dithered(audio: dict[SampleKey, NDArray[np.float64]]) -> tuple[list[PitchTask], EvalContext]:
    """A run whose encodes draw dither, so a shared stream and a per-identity one tell apart."""
    inputs = prepare_run(_instrument(48.0), audio, SR, _settings(GRID_DITHERED))
    return list(inputs.tasks), inputs.context


# --- zone cost model -----------------------------------------------------------------------------


def _task(pitch: int, silence: NDArray[np.float64]) -> PitchTask:
    """A one-event task at ``pitch``, the minimum ``_zone_trim`` reads."""
    key = SampleKey(pitch, 100)
    return PitchTask(pitch, 1.0, key, silence, (key,), (Event(key, 100, 64, 0.5, 1.0, silence, 1.0),))


def test_zone_trim_scales_with_upward_transpose() -> None:
    silence = np.zeros(4, dtype=np.float64)
    tasks = [
        _task(60, silence),
        _task(72, silence),
    ]
    assert _zone_trim(tasks, 60) == pytest.approx(1.0)  # bottom rep must stretch an octave up (2x length)
    assert _zone_trim(tasks, 72) == pytest.approx(0.5)  # top rep plays the low key slower -> its own length


def test_zone_delta_is_the_widest_upward_transpose() -> None:
    silence = np.zeros(4, dtype=np.float64)
    tasks = [_task(60, silence), _task(72, silence)]
    assert _zone_delta(tasks, 60) == _OCTAVE  # the bottom rep has to survive being played an octave up
    assert _zone_delta(tasks, 72) == 0  # the top rep only ever plays downward, which asks no bandwidth


def test_a_zone_asks_its_representative_for_the_reach_of_every_key_it_covers() -> None:
    silence = np.zeros(4, dtype=np.float64)
    tasks = [_task(pitch, silence) for pitch in (60, 67, 72)]
    assert _zone_demand(tasks, 60, _PER_KEY_BYTES) == ClipDemand(
        trim_s=1.0,
        delta_semitones=_OCTAVE,
        byte_target=3 * _PER_KEY_BYTES,
    )


def test_the_span_cap_leaves_the_zones_one_recording_can_reach_across() -> None:
    silence = np.zeros(4, dtype=np.float64)
    tasks = [_task(pitch, silence) for pitch in (60, 62, 67, 76)]
    assert list(_capped_ranges(tasks, 7)) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (3, 4)]


def test_every_key_stays_its_own_zone_however_tight_the_cap() -> None:
    silence = np.zeros(4, dtype=np.float64)
    tasks = [_task(pitch, silence) for pitch in (60, 72, 84)]
    assert list(_capped_ranges(tasks, 1)) == [(0, 1), (1, 2), (2, 3)]


def test_zone_starts_reports_where_the_zones_ending_at_each_key_begin() -> None:
    options: _ZoneOptions = {(0, 1): (), (0, 2): (), (1, 2): (), (2, 3): ()}
    assert zone_starts(options, 3) == [(), (0,), (0, 1), (2,)]


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


# --- what the candidate zones share ---------------------------------------------------------------


def _zones(tasks: list[PitchTask], context: EvalContext) -> list[_Candidate]:
    return candidate_zones(_one_layer(tasks), context.grouping.max_zone_semitones)


def test_an_encodings_dither_follows_its_identity_rather_than_when_it_is_drawn(
    dithered: tuple[list[PitchTask], EvalContext],
) -> None:
    """The property the whole staging rests on: a stored sample scores the same wherever it is reached."""
    tasks, context = dithered
    demand = _zone_demand(tasks[:1], tasks[0].pitch, _PER_KEY_BYTES)
    params = _GridCache(context).shortlist(tasks[0], demand)[0]
    stored, other = tasks[0].representative_key, tasks[1].representative_key
    assert dither(context.seed, stored, params).random() == dither(context.seed, stored, params).random()
    assert dither(context.seed, stored, params).random() != dither(context.seed, other, params).random()


def test_zones_holding_a_representative_the_same_length_read_back_one_priced_grid(
    dithered: tuple[list[PitchTask], EvalContext],
) -> None:
    """The stored length settles the pricing; a zone's transpose and reach only rank what it priced."""
    tasks, context = dithered
    cache = _GridCache(context)
    wide = _zone_demand(tasks[:2], tasks[0].pitch, _PER_KEY_BYTES)
    alone = replace(wide, delta_semitones=0, byte_target=_PER_KEY_BYTES)  # another ask, same length

    cache.shortlist(tasks[0], wide)
    cache.shortlist(tasks[0], alone)
    assert len(cache.grids) == 1


def test_pooling_the_shortlists_states_each_stored_sample_once(
    dithered: tuple[list[PitchTask], EvalContext],
) -> None:
    """Every candidate zone asks a recording for something; each distinct ask is stated once."""
    tasks, context = dithered
    shortlists = zone_shortlists(tasks, _zones(tasks, context), context, NO_PROGRESS)
    requests = store_requests(tasks, shortlists)

    stored = [(request.stored_key, encoding.params) for request in requests for encoding in request.encodings]
    assert len(stored) == len(set(stored))
    assert {key for key, _ in stored} == {stored_key for _, stored_key in shortlists}


def test_a_stored_sample_is_asked_about_exactly_the_keys_some_zone_routes_to_it(
    dithered: tuple[list[PitchTask], EvalContext],
) -> None:
    tasks, context = dithered
    shortlists = zone_shortlists(tasks, _zones(tasks, context), context, NO_PROGRESS)

    wanted: dict[tuple[SampleKey, object], set[int]] = {}
    for (span, stored_key), shortlist in shortlists.items():
        for params in shortlist:
            wanted.setdefault((stored_key, params), set()).update(range(span[0], span[1]))

    for request in store_requests(tasks, shortlists):
        for encoding in request.encodings:
            assert set(encoding.positions) == wanted[(request.stored_key, encoding.params)]


def test_scoring_a_representative_answers_for_every_encoding_asked_of_it(
    dithered: tuple[list[PitchTask], EvalContext],
) -> None:
    tasks, context = dithered
    request = store_requests(tasks, zone_shortlists(tasks, _zones(tasks, context), context, NO_PROGRESS))[0]
    scores = score_request(request, context)

    assert set(scores) == {encoding.params for encoding in request.encodings}
    for encoding in request.encodings:
        assert set(scores[encoding.params].distortions) == set(encoding.positions)


def test_scoring_shared_across_processes_reads_the_same_as_scoring_in_one(
    dithered: tuple[list[PitchTask], EvalContext],
) -> None:
    """A stored sample belongs to one representative and draws its own dither, so scheduling cannot move it."""
    tasks, context = dithered
    requests = store_requests(tasks, zone_shortlists(tasks, _zones(tasks, context), context, NO_PROGRESS))
    alone = score_stores(requests, context, workers=IN_PROCESS, progress=NO_PROGRESS)
    shared = score_stores(requests, context, workers=_SHARED, progress=NO_PROGRESS)
    assert {key: score.distortions for key, score in shared.items()} == {
        key: score.distortions for key, score in alone.items()
    }


def test_zone_options_read_the_same_however_the_scoring_was_shared_out(
    audio: dict[SampleKey, NDArray[np.float64]],
) -> None:
    inputs = prepare_run(_instrument(48.0), audio, SR, _settings(GRID_DITHERED))
    segments = _one_layer(inputs.tasks)
    alone = build_zone_options(segments, inputs.context, workers=IN_PROCESS, progress=NO_PROGRESS)
    shared = build_zone_options(segments, inputs.context, workers=_SHARED, progress=NO_PROGRESS)
    assert shared == alone


# --- end-to-end behaviour ------------------------------------------------------------------------


def test_grouping_is_never_worse_than_ungrouped_at_a_feasible_budget(
    audio: dict[SampleKey, NDArray[np.float64]], plan48: GroupedInstrumentPlan
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


def test_every_stored_sample_states_its_share_of_the_objective(plan48: GroupedInstrumentPlan) -> None:
    """One reading every consumer can add up, whichever scale the strategy searched its hulls at."""
    assert sum(unit.objective_share for unit in plan48.sample_units()) == pytest.approx(plan48.objective)


def test_a_span_cap_below_the_key_spacing_leaves_every_key_its_own_zone(
    audio: dict[SampleKey, NDArray[np.float64]],
) -> None:
    """The allocation steps between the ranges the cap left, and still covers the keyboard."""
    settings = _settings(GRID_TINY, grouping={"max_zone_semitones": 1})
    grouped = optimize_instrument_grouped(_instrument(48.0), audio, SR, settings)
    assert grouped.pitches == PITCHES  # the keys sit two semitones apart, so no pair may merge
    assert all(len(zone.pitches) == 1 for zone in grouped.zones)


def test_grouping_is_feasible_where_ungrouped_is_not(audio: dict[SampleKey, NDArray[np.float64]]) -> None:
    settings = _settings(GRID_TINY)
    inst = _instrument(10.0)  # room for one shared sample, not for three separate ones
    with pytest.raises(BudgetInfeasibleError):
        optimize_instrument(inst, audio, SR, settings)
    grouped = optimize_instrument_grouped(inst, audio, SR, settings)
    assert len(grouped.zones) < len(PITCHES)  # keys had to be merged to fit
    assert set(grouped.pitches) == set(PITCHES)  # but every key is still covered
    assert grouped.used_bytes <= grouped.sample_budget_bytes


def test_grouping_raises_when_even_one_merged_zone_overflows(audio: dict[SampleKey, NDArray[np.float64]]) -> None:
    with pytest.raises(BudgetInfeasibleError):
        optimize_instrument_grouped(_instrument(1.0), audio, SR, _settings(GRID_TINY))


def test_grouping_handles_the_sustained_archetype() -> None:
    pitches = (60, 62, 64)
    audio = {
        SampleKey(pitch, 100): synthesize(
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
