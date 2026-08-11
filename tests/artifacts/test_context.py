from __future__ import annotations

from pathlib import Path

from optisample.artifacts.context import NO_INSTRUMENTS, DumpContext, DumpResult, PlanArtifacts
from optisample.optimize.layers.bands import UNSPLIT, VelocityBand, VelocityLayers


def test_dump_context_indexes_tasks_by_layer_and_material_pitch(dump_context: DumpContext) -> None:
    tasks = dump_context.layer_tasks(UNSPLIT)
    assert set(tasks) == {(0, 60), (0, 62), (0, 64)}  # one task per distinct material pitch of the one layer
    assert dump_context.sample_rate == 44_100
    assert len(dump_context.material) == 3


def test_dump_context_answers_the_prepared_tasks_for_the_full_range_layer(dump_context: DumpContext) -> None:
    tasks = dump_context.layer_tasks(UNSPLIT)
    prepared = {task.pitch: task for task in dump_context.inputs.tasks}
    for (_, pitch), task in tasks.items():
        expected = prepared[pitch]
        assert (task.weight, task.representative_key, task.candidates) == (
            expected.weight,
            expected.representative_key,
            expected.candidates,
        )
        assert [event.identity for event in task.events] == [event.identity for event in expected.events]


def test_dump_context_gives_a_split_only_the_keys_each_band_plays(dump_context: DumpContext) -> None:
    """The demo plays every note at velocity 100, so a split below it leaves its quiet layer empty."""
    split = VelocityLayers((VelocityBand(0, 99), VelocityBand(100, 127)))
    tasks = dump_context.layer_tasks(split)
    assert {layer for layer, _ in tasks} == {1}
    assert {pitch for _, pitch in tasks} == {60, 62, 64}


def test_plan_artifacts_and_result_carry_the_outcome() -> None:
    artifact = PlanArtifacts(
        name="ungrouped",
        reason=None,
        rendered=False,
        objective=1.25,
        used_bytes=4096,
        instruments=NO_INSTRUMENTS,
        elapsed_s=0.5,
    )
    result = DumpResult(instrument_id="piano", directory=Path("/tmp/out"), plans=(artifact,))
    assert result.instrument_id == "piano"
    assert [plan.name for plan in result.plans] == ["ungrouped"]
    assert result.plans[0].objective == 1.25 and result.plans[0].used_bytes == 4096
