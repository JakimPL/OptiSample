import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts import DumpSettings, dump_instrument, dump_project
from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.audio import read_wav, write_wav
from optisample.io.render import openmpt123_available
from optisample.io.tracker.target import export_target
from optisample.keys import SampleKey
from optisample.model import (
    InstrumentSpec,
    Manifest,
    NoteEvent,
    ProjectSpec,
    SourceSample,
)
from optisample.optimize.orchestrate.audio import LoadedInstrument
from optisample.optimize.orchestrate.looping import run_loops
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.trim import NO_SCREEN
from optisample.optimize.tasks import AudioMap, StoredRecordings
from trackmod.trackers.it.instrument_file import ITInstrumentFile

Recordings = Callable[..., StoredRecordings]

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

SR = 44_100
PITCHES = (60, 62, 64)
CONTAINER = "piano.bank"
MANIFEST = "bank.json"

AudioFactory = Callable[..., dict[SampleKey, NDArray[np.float64]]]

_CONFIG = load_config()
_STORED_CEILING_HZ = 5_000.0  # the band these fixtures store, which lands them on the 11 kHz rung


def _narrow_band_reduce() -> ReduceConfig:
    """The bundled reduction, storing only the band ``_STORED_CEILING_HZ`` names.

    That ceiling is what puts these fixtures' samples on the 11 kHz rung, so the budgets they assert
    against stay at the scale of a handful of stored samples.
    """
    raw = _CONFIG.reduce.model_dump()
    return ReduceConfig.model_validate({**raw, "bandwidth": {**raw["bandwidth"], "ceiling_hz": _STORED_CEILING_HZ}})


def _settings() -> OptimizeSettings:
    grid = SweepConfig.model_validate(
        {**_CONFIG.optimize.sweep.model_dump(), "rates": (11_025,), "depth": 8, "dither": False}
    )
    return OptimizeSettings(
        loop=_CONFIG.loop,
        sweep=grid,
        reduce=_narrow_band_reduce(),
        layers=_CONFIG.optimize.layers,
        encode=_CONFIG.encode,
        metrics=_CONFIG.analysis.metrics,
        velocity=_CONFIG.optimize.velocity,
        method=_CONFIG.optimize.budget.method,
        energy_exponent=_CONFIG.optimize.budget.energy_exponent,
        max_samples=_CONFIG.optimize.budget.max_samples,
        target=export_target(_CONFIG.export.tracker),
    )


NO_RENDER = DumpSettings(
    optimize=_settings(), render=_CONFIG.export.render, playback=_CONFIG.export.playback, render_ground_truth=False
)


@pytest.fixture(scope="session")
def demo_audio_map(piano_note: Callable[..., NDArray[np.float64]]) -> AudioFactory:
    """Factory: the demo piano audio grid over ``pitches`` (default the full material set)."""

    def _audio(pitches: tuple[int, ...] = PITCHES) -> dict[SampleKey, NDArray[np.float64]]:
        return {SampleKey(pitch, 100): piano_note(pitch, 100, 0.6, seed=pitch * 137 + 100) for pitch in pitches}

    return _audio


def _instrument(budget_kb: float, pitches: tuple[int, ...] = PITCHES) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in pitches]
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=4) for pitch in pitches]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=material)


def _settled_recordings(instrument: InstrumentSpec, audio: AudioMap) -> StoredRecordings:
    """The recordings a dump run encodes from, with each one's loop settled the way the stage would."""
    loaded = LoadedInstrument(instrument=instrument, audio=dict(audio), sample_rate=SR, screen=NO_SCREEN)
    return run_loops(loaded, NO_RENDER.optimize).recordings


@pytest.fixture(scope="module")
def generous(tmp_path_factory: pytest.TempPathFactory, demo_audio_map: AudioFactory) -> Path:
    out = tmp_path_factory.mktemp("generous")
    dump_instrument(_instrument(48.0), _settled_recordings(_instrument(48.0), demo_audio_map()), out, NO_RENDER)
    return out


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _bank(directory: Path) -> dict[str, Any]:
    """The manifest one strategy's bank states, read out of the archive it travels in."""
    with zipfile.ZipFile(directory / CONTAINER) as archive:
        return json.loads(archive.read(MANIFEST))


def _held(directory: Path) -> list[str]:
    """Every instrument one strategy's bank holds, in the order the manifest names them."""
    with zipfile.ZipFile(directory / CONTAINER) as archive:
        return [entry for entry in archive.namelist() if entry != MANIFEST]


# --- tree + contents -----------------------------------------------------------------------------


def test_dump_writes_both_strategy_subtrees(generous: Path) -> None:
    for name in ("ungrouped", "grouped"):
        base = generous / name
        assert (base / "piano.bank").is_file()
        assert (base / "report.txt").is_file()
        assert (base / "plan.json").is_file()
        assert (base / "reduction.json").is_file()
        assert (base / "metrics.json").is_file()
        assert list((base / "samples").glob("*.wav"))  # at least one stored sample
        assert list((base / "instruments").glob("*.iti"))
        assert list((base / "compare").glob("*/*_ref.wav"))
        assert list((base / "compare").glob("*/*_render.wav"))


def test_one_instrument_file_is_written_for_each_instrument_the_module_numbers(generous: Path) -> None:
    """A consumer loads a voice by file, so every instrument the plan was written as has one of its own."""
    for name in ("ungrouped", "grouped"):
        plan = _load(generous / name / "plan.json")
        written = sorted((generous / name / "instruments").iterdir())
        assert len(written) == len(plan["instruments"])
        assert {path.suffix for path in written} == {".iti"}


def test_a_written_instrument_loads_back_as_the_voice_the_plan_states(generous: Path) -> None:
    """The file stands alone: it names the instrument the plan wrote and carries the samples it owns."""
    plan = _load(generous / "grouped" / "plan.json")
    written = sorted((generous / "grouped" / "instruments").iterdir())
    for record, path in zip(plan["instruments"], written):
        unit = ITInstrumentFile.load(path).unit
        assert unit.instrument.name == record["name"]
        assert len(unit.samples) == record["samples"]


def test_the_bank_holds_the_instruments_the_tree_spreads_beside_it(generous: Path) -> None:
    """One description of a bank is written twice, so a voice read either way is the same file."""
    for name in ("ungrouped", "grouped"):
        base = generous / name
        with zipfile.ZipFile(base / CONTAINER) as archive:
            for entry in _held(base):
                assert archive.read(entry) == (base / entry).read_bytes()


def test_an_instrument_file_is_named_by_the_keys_and_the_dynamics_it_answers(generous: Path) -> None:
    plan = _load(generous / "grouped" / "plan.json")
    (record,) = plan["instruments"]
    (written,) = (generous / "grouped" / "instruments").iterdir()
    assert written.stem == f"p{record['lowest_pitch']:03d}-p{record['highest_pitch']:03d}_{record['band']}"


def test_the_bank_names_the_instruments_it_carries(generous: Path) -> None:
    """A player is handed the bank alone, so every entry it names is stored in the bank itself."""
    for name in ("ungrouped", "grouped"):
        base = generous / name
        bank = _bank(base)
        held = _held(base)
        assert bank["version"] == 2
        assert bank["name"] == "piano"
        assert len(bank["layers"]) == len(_load(base / "plan.json")["instruments"])
        assert [layer["source"]["file"] for layer in bank["layers"]] == held


def test_the_bank_hands_every_dynamic_to_the_band_the_plan_stored_it_in(generous: Path) -> None:
    plan = _load(generous / "grouped" / "plan.json")
    bank = _bank(generous / "grouped")
    stored = [(record["lowest_velocity"], record["highest_velocity"]) for record in plan["instruments"]]
    selected = [(layer["select"]["velocity"]["low"], layer["select"]["velocity"]["high"]) for layer in bank["layers"]]
    assert selected == stored


def test_metrics_objective_reproduces_plan_objective(generous: Path) -> None:
    for name in ("ungrouped", "grouped"):
        plan = _load(generous / name / "plan.json")
        metrics = _load(generous / name / "metrics.json")
        assert metrics["objective"] == pytest.approx(plan["objective"])  # the surrogate scores are the objective
        assert metrics["plan_objective"] == pytest.approx(plan["objective"])


def test_ungrouped_dumps_one_sample_per_pitch(generous: Path) -> None:
    plan = _load(generous / "ungrouped" / "plan.json")
    samples = list((generous / "ungrouped" / "samples").glob("*.wav"))
    assert len(samples) == len(plan["pitches"]) == len(PITCHES)
    # Each covered pitch gets an A/B pair.
    assert len(list((generous / "ungrouped" / "compare").glob("*/*_ref.wav"))) == len(PITCHES)


def test_grouped_samples_match_zone_count(generous: Path) -> None:
    plan = _load(generous / "grouped" / "plan.json")
    samples = list((generous / "grouped" / "samples").glob("*.wav"))
    assert len(samples) == len(plan["zones"])
    covered = [pitch for zone in plan["zones"] for pitch in zone["pitches"]]
    assert covered == list(PITCHES)  # zones partition every key, in order


def test_plan_json_records_budget_and_params(generous: Path) -> None:
    plan = _load(generous / "ungrouped" / "plan.json")
    assert plan["strategy"] == "ungrouped"
    assert plan["budget"]["used_bytes"] <= plan["budget"]["sample_budget_bytes"]
    first = plan["pitches"][0]
    assert {"pitch", "note", "target_rate", "depth_bits", "stored_bytes", "distortion"} <= set(first)


def test_reduction_json_states_what_the_pre_pass_left(generous: Path) -> None:
    reduction = _load(generous / "ungrouped" / "reduction.json")
    assert reduction["kept_recordings"] == len(reduction["recordings"]) == len(PITCHES)
    assert reduction["scored_classes"] <= reduction["played_notes"]
    assert [grid["pitch"] for grid in reduction["grids"]] == list(PITCHES)
    assert all(grid["stored"]["target_rate"] >= grid["useful_rate_hz"] for grid in reduction["grids"])
    assert all(grid["swept"] > 0 for grid in reduction["grids"])


def test_the_report_states_the_reduction_alongside_the_allocation(generous: Path) -> None:
    for name in ("ungrouped", "grouped"):
        report = (generous / name / "report.txt").read_text()
        assert "Reduction (pre-optimization)" in report
        assert "swept per key" in report


def test_the_map_a_layer_carries_has_anchors_and_the_full_table(generous: Path) -> None:
    """A layer reads a note's dynamic through the map stored beside the waveforms it was measured on."""
    for layer in _bank(generous / "ungrouped")["layers"]:
        vmap = layer["velocity_map"]
        assert len(vmap["volumes"]) == 128
        assert vmap["anchors"]
        assert 0 <= vmap["reference_volume"] <= 64

    assert _load(generous / "ungrouped" / "plan.json")["velocity_map"]["volumes"] == vmap["volumes"]


def test_metrics_per_note_has_breakdown_and_diagnostics(generous: Path) -> None:
    metrics = _load(generous / "grouped" / "metrics.json")
    note = metrics["notes"][0]
    assert note["events"]
    event = note["events"][0]
    assert "mrstft" in event["breakdown"]
    assert "si_sdr_db" in event["diagnostics"]
    assert note["render_source"] == "surrogate"  # ground-truth rendering was disabled


# --- surrogate vs ground-truth render ------------------------------------------------------------


def test_no_render_skips_the_render_directory(generous: Path) -> None:
    assert not (generous / "ungrouped" / "render").exists()


@requires_openmpt
def test_ground_truth_render_produces_real_audio(tmp_path: Path, demo_audio_map: AudioFactory) -> None:
    out = tmp_path / "gt"
    instrument = _instrument(48.0, (60, 62))
    result = dump_instrument(
        instrument,
        _settled_recordings(instrument, demo_audio_map((60, 62))),
        out,
        # render_ground_truth defaults True
        DumpSettings(optimize=_settings(), render=_CONFIG.export.render, playback=_CONFIG.export.playback),
    )
    assert all(plan.rendered for plan in result.plans if plan.feasible)
    module_wav = out / "grouped" / "render" / "module.wav"
    assert module_wav.is_file()
    audio, rate = read_wav(module_wav)
    assert rate == _CONFIG.export.render.sample_rate and float(np.max(np.abs(audio))) > 0.0
    metrics = _load(out / "grouped" / "metrics.json")
    assert metrics["notes"][0]["render_source"] == "openmpt123"


# --- feasibility ---------------------------------------------------------------------------------


def test_tight_budget_marks_ungrouped_infeasible_but_dumps_grouped(
    tmp_path: Path,
    demo_audio_map: AudioFactory,
    recordings: Recordings,
) -> None:
    out = tmp_path / "tight"
    result = dump_instrument(_instrument(10.0), recordings(demo_audio_map(), SR), out, NO_RENDER)
    by_name = {plan.name: plan for plan in result.plans}
    assert by_name["ungrouped"].feasible is False
    assert (out / "ungrouped" / "INFEASIBLE.txt").is_file()
    assert not (out / "ungrouped" / CONTAINER).exists()
    assert by_name["grouped"].feasible is True
    assert (out / "grouped" / CONTAINER).is_file()


def test_impossible_budget_marks_both_infeasible(
    tmp_path: Path, demo_audio_map: AudioFactory, recordings: Recordings
) -> None:
    out = tmp_path / "impossible"
    result = dump_instrument(_instrument(1.0), recordings(demo_audio_map(), SR), out, NO_RENDER)
    assert all(not plan.feasible for plan in result.plans)
    assert all(plan.reason for plan in result.plans)


# --- strategy selection + determinism ------------------------------------------------------------


def test_strategy_flags_restrict_which_plans_run(
    tmp_path: Path, demo_audio_map: AudioFactory, recordings: Recordings
) -> None:
    out = tmp_path / "grouped-only"
    settings = DumpSettings(
        optimize=_settings(),
        render=_CONFIG.export.render,
        playback=_CONFIG.export.playback,
        render_ground_truth=False,
        ungrouped=False,
    )
    result = dump_instrument(_instrument(48.0), recordings(demo_audio_map(), SR), out, settings)
    assert [plan.name for plan in result.plans] == ["grouped"]
    assert not (out / "ungrouped").exists()


def test_dump_is_deterministic(tmp_path: Path, demo_audio_map: AudioFactory, recordings: Recordings) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    dump_instrument(_instrument(48.0), recordings(demo_audio_map(), SR), first, NO_RENDER)
    dump_instrument(_instrument(48.0), recordings(demo_audio_map(), SR), second, NO_RENDER)
    assert (first / "grouped" / CONTAINER).read_bytes() == (second / "grouped" / CONTAINER).read_bytes()
    # The stored-sample WAVs decode identically (only libsndfile's PEAK-chunk timestamp differs on disk).
    audio_a, _ = read_wav(next((first / "grouped" / "samples").glob("*.wav")))
    audio_b, _ = read_wav(next((second / "grouped" / "samples").glob("*.wav")))
    assert np.array_equal(audio_a, audio_b)


# --- project-level -------------------------------------------------------------------------------


def test_dump_project_reads_wavs_and_writes_per_instrument(
    tmp_path: Path, piano_note: Callable[..., NDArray[np.float64]]
) -> None:
    data_dir = tmp_path / "wavs"
    data_dir.mkdir()
    samples = []
    for pitch in PITCHES:
        path = data_dir / f"p{pitch}.wav"
        write_wav(path, piano_note(pitch, 100, 0.6, seed=pitch * 137 + 100), SR)
        samples.append(SourceSample(file=path, pitch=pitch, velocity=100))
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=2) for pitch in PITCHES]
    instrument = InstrumentSpec(id="piano", budget_kb=48.0, samples=samples, material=material)
    manifest = Manifest(project=ProjectSpec(name="demo"), instruments=[instrument])

    results = dump_project(manifest, tmp_path / "artifacts", NO_RENDER)
    assert len(results) == 1
    assert (tmp_path / "artifacts" / "piano" / "grouped" / CONTAINER).is_file()


# --- velocity layers -----------------------------------------------------------------------------

_DYNAMICS = (40, 110)  # two dynamics per key, so a velocity split has a timbre difference to buy


@pytest.fixture(scope="module")
def layered(tmp_path_factory: pytest.TempPathFactory, piano_note: Callable[..., NDArray[np.float64]]) -> Path:
    """A generous dump of an instrument every key of which is played softly and loudly."""
    audio = {
        SampleKey(pitch, velocity): piano_note(pitch, velocity, 0.6, seed=pitch * 137 + velocity)
        for pitch in PITCHES
        for velocity in _DYNAMICS
    }
    samples = [
        SourceSample(file=Path(f"{pitch}_{velocity}.wav"), pitch=pitch, velocity=velocity)
        for pitch in PITCHES
        for velocity in _DYNAMICS
    ]
    material = [
        NoteEvent(pitch=pitch, velocity=velocity, duration_s=0.5, count=4)
        for pitch in PITCHES
        for velocity in _DYNAMICS
    ]
    instrument = InstrumentSpec(id="piano", budget_kb=192.0, samples=samples, material=material)
    out = tmp_path_factory.mktemp("layered")
    dump_instrument(instrument, _settled_recordings(instrument, audio), out, NO_RENDER)
    return out


def test_a_layered_plan_files_its_ab_pairs_under_the_band_that_played_them(layered: Path) -> None:
    plan = _load(layered / "grouped" / "plan.json")
    metrics = _load(layered / "grouped" / "metrics.json")
    bands = sorted({note["layer"] for note in metrics["notes"]})
    assert len(bands) > 1  # the material earned a velocity split
    assert sorted(folder.name for folder in (layered / "grouped" / "compare").iterdir()) == bands
    for band in bands:
        assert list((layered / "grouped" / "compare" / band).glob("*_ref.wav"))

    assert len(metrics["notes"]) == sum(len(zone["pitches"]) for zone in plan["zones"])


def test_a_layered_plan_scores_each_key_once_per_band_it_is_played_in(layered: Path) -> None:
    metrics = _load(layered / "grouped" / "metrics.json")
    covered = [(note["layer"], note["pitch"]) for note in metrics["notes"]]
    assert len(covered) == len(set(covered))  # one record per (layer, pitch), never a silent overwrite
    assert metrics["objective"] == pytest.approx(_load(layered / "grouped" / "plan.json")["objective"], rel=1e-4)


def test_a_layered_plan_writes_one_instrument_file_per_band(layered: Path) -> None:
    """Each band is a voice of its own, so a consumer picks its dynamics by picking a file."""
    plan = _load(layered / "grouped" / "plan.json")
    written = sorted(path.stem for path in (layered / "grouped" / "instruments").iterdir())
    assert written == sorted(
        f"p{record['lowest_pitch']:03d}-p{record['highest_pitch']:03d}_{record['band']}"
        for record in plan["instruments"]
    )
    assert len({stem.split("_")[-1] for stem in written}) > 1  # the material earned a velocity split


def test_a_layered_bank_states_a_layer_per_band_and_tiles_the_dynamics(layered: Path) -> None:
    """A note of any dynamic reaches the instrument the allocation stored its band in."""
    bank = _bank(layered / "grouped")
    bands = [layer["select"]["velocity"] for layer in bank["layers"]]
    assert len(bands) > 1  # the material earned a velocity split
    for velocity in range(128):
        assert len([band for band in bands if band["low"] <= velocity <= band["high"]]) == 1


def test_an_ungrouped_plan_stays_a_single_full_range_layer(layered: Path) -> None:
    metrics = _load(layered / "ungrouped" / "metrics.json")
    assert {note["layer"] for note in metrics["notes"]} == {"v000-v127"}
