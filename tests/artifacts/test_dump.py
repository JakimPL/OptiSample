from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.artifacts import DumpSettings, dump_instrument, dump_project
from optisample.config import load_config
from optisample.config.optimize import SweepConfig
from optisample.io.audio import read_wav, write_wav
from optisample.io.render import openmpt123_available
from optisample.metrics import build_composite
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample
from optisample.optimize.orchestrate import OptimizeSettings
from optisample.synth import NoteSpec, render_sample

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

SR = 44_100
PITCHES = (60, 62, 64)

# Build cheap swept settings straight from the bundled config (dither off, so the dump re-encode is
# deterministic and fast). Loaded once here since these feed module-level constants and the module-scoped
# ``generous`` fixture that the function-scoped conftest fixtures cannot reach.
_CONFIG = load_config()


def _settings() -> OptimizeSettings:
    grid = SweepConfig.model_validate(
        {**_CONFIG.sweep.model_dump(), "rates": (11_025,), "depths": (8,), "dither": False}
    )
    return OptimizeSettings(
        sweep=grid,
        encode=_CONFIG.encode,
        composite=build_composite(_CONFIG.metrics),
        velocity=_CONFIG.velocity,
        method=_CONFIG.optimize.method,
    )


NO_RENDER = DumpSettings(
    optimize=_settings(), render=_CONFIG.render, playback=_CONFIG.playback, render_ground_truth=False
)


def _note(pitch: int, velocity: int, dur: float) -> NDArray[np.float64]:
    return render_sample(
        "piano",
        NoteSpec(pitch, velocity, 0.0, dur, SR),
        np.random.default_rng(pitch * 137 + velocity),
        _CONFIG.synth,
    )


def _audio(pitches: tuple[int, ...] = PITCHES) -> dict[tuple[int, int], NDArray[np.float64]]:
    return {(pitch, 100): _note(pitch, 100, 0.6) for pitch in pitches}


def _instrument(budget_kb: float, pitches: tuple[int, ...] = PITCHES) -> InstrumentSpec:
    samples = [SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in pitches]
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=4) for pitch in pitches]
    return InstrumentSpec(id="piano", budget_kb=budget_kb, samples=samples, material=material)


@pytest.fixture(scope="module")
def generous(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("generous")
    dump_instrument(_instrument(48.0), _audio(), SR, out, NO_RENDER)
    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# --- tree + contents -----------------------------------------------------------------------------


def test_dump_writes_both_strategy_subtrees(generous: Path) -> None:
    for name in ("ungrouped", "grouped"):
        base = generous / name
        assert (base / "module.it").is_file()
        assert (base / "report.txt").is_file()
        assert (base / "plan.json").is_file()
        assert (base / "velocity_map.json").is_file()
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
def test_ground_truth_render_produces_real_audio(tmp_path: Path) -> None:
    out = tmp_path / "gt"
    result = dump_instrument(
        _instrument(48.0, (60, 62)),
        _audio((60, 62)),
        SR,
        out,
        # render_ground_truth defaults True
        DumpSettings(optimize=_settings(), render=_CONFIG.render, playback=_CONFIG.playback),
    )
    assert all(plan.rendered for plan in result.plans if plan.feasible)
    module_wav = out / "grouped" / "render" / "module.wav"
    assert module_wav.is_file()
    audio, rate = read_wav(module_wav)
    assert rate == 48_000 and float(np.max(np.abs(audio))) > 0.0
    metrics = _load(out / "grouped" / "metrics.json")
    assert metrics["notes"][0]["render_source"] == "openmpt123"


# --- feasibility ---------------------------------------------------------------------------------


def test_tight_budget_marks_ungrouped_infeasible_but_dumps_grouped(tmp_path: Path) -> None:
    out = tmp_path / "tight"
    result = dump_instrument(_instrument(10.0), _audio(), SR, out, NO_RENDER)
    by_name = {plan.name: plan for plan in result.plans}
    assert by_name["ungrouped"].feasible is False
    assert (out / "ungrouped" / "INFEASIBLE.txt").is_file()
    assert not (out / "ungrouped" / "module.it").exists()
    assert by_name["grouped"].feasible is True
    assert (out / "grouped" / "module.it").is_file()


def test_impossible_budget_marks_both_infeasible(tmp_path: Path) -> None:
    out = tmp_path / "impossible"
    result = dump_instrument(_instrument(1.0), _audio(), SR, out, NO_RENDER)
    assert all(not plan.feasible for plan in result.plans)
    assert all(plan.reason for plan in result.plans)


# --- strategy selection + determinism ------------------------------------------------------------


def test_strategy_flags_restrict_which_plans_run(tmp_path: Path) -> None:
    out = tmp_path / "grouped-only"
    settings = DumpSettings(
        optimize=_settings(),
        render=_CONFIG.render,
        playback=_CONFIG.playback,
        render_ground_truth=False,
        ungrouped=False,
    )
    result = dump_instrument(_instrument(48.0), _audio(), SR, out, settings)
    assert [plan.name for plan in result.plans] == ["grouped"]
    assert not (out / "ungrouped").exists()


def test_dump_is_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    dump_instrument(_instrument(48.0), _audio(), SR, first, NO_RENDER)
    dump_instrument(_instrument(48.0), _audio(), SR, second, NO_RENDER)
    module_a = (first / "grouped" / "module.it").read_bytes()
    module_b = (second / "grouped" / "module.it").read_bytes()
    assert module_a == module_b  # the exported module is byte-identical across runs
    # The stored-sample WAVs decode identically (only libsndfile's PEAK-chunk timestamp differs on disk).
    audio_a, _ = read_wav(next((first / "grouped" / "samples").glob("*.wav")))
    audio_b, _ = read_wav(next((second / "grouped" / "samples").glob("*.wav")))
    assert np.array_equal(audio_a, audio_b)


# --- project-level -------------------------------------------------------------------------------


def test_dump_project_reads_wavs_and_writes_per_instrument(tmp_path: Path) -> None:
    data_dir = tmp_path / "wavs"
    data_dir.mkdir()
    samples = []
    for pitch in PITCHES:
        path = data_dir / f"p{pitch}.wav"
        write_wav(path, _note(pitch, 100, 0.6), SR)
        samples.append(SourceSample(file=path, pitch=pitch, velocity=100))
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=0.5, count=2) for pitch in PITCHES]
    instrument = InstrumentSpec(id="piano", budget_kb=48.0, samples=samples, material=material)
    manifest = Manifest(project=ProjectSpec(name="demo"), instruments=[instrument])

    results = dump_project(manifest, tmp_path / "artifacts", NO_RENDER)
    assert len(results) == 1
    assert (tmp_path / "artifacts" / "piano" / "grouped" / "module.it").is_file()
