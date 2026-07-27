import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts import DumpSettings, dump_instrument, dump_project
from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.io.audio import read_wav, write_wav
from optisample.io.render import openmpt123_available
from optisample.io.tracker.target import export_target
from optisample.model import (
    InstrumentSpec,
    Manifest,
    NoteEvent,
    ProjectSpec,
    SourceSample,
)
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.reduce.keys import SampleKey

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

SR = 44_100
PITCHES = (60, 62, 64)

AudioFactory = Callable[..., dict[SampleKey, NDArray[np.float64]]]

_CONFIG = load_config()


def _settings() -> OptimizeSettings:
    grid = SweepConfig.model_validate(
        {**_CONFIG.sweep.model_dump(), "rates": (11_025,), "depths": (8,), "dither": False}
    )
    return OptimizeSettings(
        sweep=grid,
        reduce=_CONFIG.reduce,
        layers=_CONFIG.layers,
        encode=_CONFIG.encode,
        metrics=_CONFIG.metrics,
        velocity=_CONFIG.velocity,
        method=_CONFIG.optimize.method,
        target=export_target(_CONFIG.tracker),
    )


NO_RENDER = DumpSettings(
    optimize=_settings(), render=_CONFIG.render, playback=_CONFIG.playback, render_ground_truth=False
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


@pytest.fixture(scope="module")
def generous(tmp_path_factory: pytest.TempPathFactory, demo_audio_map: AudioFactory) -> Path:
    out = tmp_path_factory.mktemp("generous")
    dump_instrument(_instrument(48.0), demo_audio_map(), SR, out, NO_RENDER)
    return out


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


# --- tree + contents -----------------------------------------------------------------------------


def test_dump_writes_both_strategy_subtrees(generous: Path) -> None:
    for name in ("ungrouped", "grouped"):
        base = generous / name
        assert (base / "module.it").is_file()
        assert (base / "report.txt").is_file()
        assert (base / "plan.json").is_file()
        assert (base / "velocity_map.json").is_file()
        assert (base / "reduction.json").is_file()
        assert (base / "metrics.json").is_file()
        assert list((base / "samples").glob("*.wav"))  # at least one stored sample
        assert list((base / "compare").glob("*_ref.wav"))
        assert list((base / "compare").glob("*_render.wav"))


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
    assert len(list((generous / "ungrouped" / "compare").glob("*_ref.wav"))) == len(PITCHES)


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
    assert all(len(grid["shortlist"]) <= reduction["grid_size"] for grid in reduction["grids"])


def test_the_report_states_the_reduction_alongside_the_allocation(generous: Path) -> None:
    for name in ("ungrouped", "grouped"):
        report = (generous / name / "report.txt").read_text()
        assert "Reduction (pre-optimization)" in report
        assert "shortlisted per key" in report


def test_velocity_map_json_has_anchors_and_full_table(generous: Path) -> None:
    vmap = _load(generous / "ungrouped" / "velocity_map.json")
    assert len(vmap["volumes"]) == 128
    assert vmap["anchors"]
    assert 0 <= vmap["reference_volume"] <= 64


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
    result = dump_instrument(
        _instrument(48.0, (60, 62)),
        demo_audio_map((60, 62)),
        SR,
        out,
        # render_ground_truth defaults True
        DumpSettings(optimize=_settings(), render=_CONFIG.render, playback=_CONFIG.playback),
    )
    assert all(plan.rendered for plan in result.plans if plan.feasible)
    module_wav = out / "grouped" / "render" / "module.wav"
    assert module_wav.is_file()
    audio, rate = read_wav(module_wav)
    assert rate == _CONFIG.render.sample_rate and float(np.max(np.abs(audio))) > 0.0
    metrics = _load(out / "grouped" / "metrics.json")
    assert metrics["notes"][0]["render_source"] == "openmpt123"


# --- feasibility ---------------------------------------------------------------------------------


def test_tight_budget_marks_ungrouped_infeasible_but_dumps_grouped(
    tmp_path: Path, demo_audio_map: AudioFactory
) -> None:
    out = tmp_path / "tight"
    result = dump_instrument(_instrument(10.0), demo_audio_map(), SR, out, NO_RENDER)
    by_name = {plan.name: plan for plan in result.plans}
    assert by_name["ungrouped"].feasible is False
    assert (out / "ungrouped" / "INFEASIBLE.txt").is_file()
    assert not (out / "ungrouped" / "module.it").exists()
    assert by_name["grouped"].feasible is True
    assert (out / "grouped" / "module.it").is_file()


def test_impossible_budget_marks_both_infeasible(tmp_path: Path, demo_audio_map: AudioFactory) -> None:
    out = tmp_path / "impossible"
    result = dump_instrument(_instrument(1.0), demo_audio_map(), SR, out, NO_RENDER)
    assert all(not plan.feasible for plan in result.plans)
    assert all(plan.reason for plan in result.plans)


# --- strategy selection + determinism ------------------------------------------------------------


def test_strategy_flags_restrict_which_plans_run(tmp_path: Path, demo_audio_map: AudioFactory) -> None:
    out = tmp_path / "grouped-only"
    settings = DumpSettings(
        optimize=_settings(),
        render=_CONFIG.render,
        playback=_CONFIG.playback,
        render_ground_truth=False,
        ungrouped=False,
    )
    result = dump_instrument(_instrument(48.0), demo_audio_map(), SR, out, settings)
    assert [plan.name for plan in result.plans] == ["grouped"]
    assert not (out / "ungrouped").exists()


def test_dump_is_deterministic(tmp_path: Path, demo_audio_map: AudioFactory) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    dump_instrument(_instrument(48.0), demo_audio_map(), SR, first, NO_RENDER)
    dump_instrument(_instrument(48.0), demo_audio_map(), SR, second, NO_RENDER)
    module_a = (first / "grouped" / "module.it").read_bytes()
    module_b = (second / "grouped" / "module.it").read_bytes()
    assert module_a == module_b  # the exported module is byte-identical across runs
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
    assert (tmp_path / "artifacts" / "piano" / "grouped" / "module.it").is_file()
