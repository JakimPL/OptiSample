from __future__ import annotations

from pathlib import Path

from optisample.artifacts.context import DumpContext, DumpResult, DumpSettings, PlanArtifacts
from optisample.config import OptiConfig
from optisample.optimize.orchestrate import OptimizeSettings


def test_dump_settings_defaults_run_both_strategies_and_render(
    tiny_settings: OptimizeSettings, config: OptiConfig
) -> None:
    settings = DumpSettings(optimize=tiny_settings, render=config.render, playback=config.playback)
    assert settings.render_ground_truth is True  # ground-truth render is on by default
    assert settings.grouped is True and settings.ungrouped is True


def test_dump_context_indexes_tasks_by_material_pitch(dump_context: DumpContext) -> None:
    assert set(dump_context.tasks_by_pitch) == {60, 62, 64}  # one task per distinct material pitch
    assert dump_context.sample_rate == 44_100
    assert len(dump_context.material) == 3


def test_plan_artifacts_and_result_carry_the_outcome() -> None:
    artifact = PlanArtifacts(
        name="ungrouped",
        reason=None,
        rendered=False,
        objective=1.25,
        used_bytes=4096,
        elapsed_s=0.5,
    )
    result = DumpResult(instrument_id="piano", directory=Path("/tmp/out"), plans=(artifact,))
    assert result.instrument_id == "piano"
    assert [plan.name for plan in result.plans] == ["ungrouped"]
    assert result.plans[0].objective == 1.25 and result.plans[0].used_bytes == 4096
