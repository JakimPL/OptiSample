from __future__ import annotations

import numpy as np
import pytest
from trackmod import BitDepth

from optisample.calibrate.ranking import (
    RankingGrid,
    clip_renditions,
    reference,
    rendered,
    widened_encodings,
)
from optisample.optimize.orchestrate import RunInputs
from optisample.optimize.tasks import EvalContext, PitchTask


def _priced(inputs: RunInputs, grid: RankingGrid) -> tuple[PitchTask, EvalContext, tuple]:
    task = inputs.tasks[0]
    encodings = widened_encodings(
        inputs.reduction.encodings()[task.pitch],
        grid,
        inputs.context.sweep,
        sample_rate=inputs.context.sample_rate,
    )
    return task, inputs.context, encodings


def test_every_encoding_a_grid_offers_is_priced(priced_run: RunInputs) -> None:
    grid = RankingGrid(depths=(BitDepth.SIXTEEN, BitDepth.EIGHT), rate_steps=1)
    task, context, encodings = _priced(priced_run, grid)

    clip = clip_renditions(task, encodings, context)

    assert tuple(rendition.params for rendition in clip.renditions) == encodings


def test_a_rendition_is_priced_in_the_bytes_a_module_spends_on_it(priced_run: RunInputs) -> None:
    task, context, encodings = _priced(priced_run, RankingGrid(depths=(BitDepth.SIXTEEN, BitDepth.EIGHT), rate_steps=0))

    clip = clip_renditions(task, encodings, context)

    assert all(rendition.stored_bytes > 0 for rendition in clip.renditions)


def test_a_shallower_grid_stores_fewer_bytes(priced_run: RunInputs) -> None:
    task, context, encodings = _priced(priced_run, RankingGrid(depths=(BitDepth.SIXTEEN, BitDepth.EIGHT), rate_steps=0))
    clip = clip_renditions(task, encodings, context)

    by_depth = {
        rendition.params.depth: rendition.stored_bytes
        for rendition in clip.renditions
        if rendition.params.loop_index is None
    }

    assert by_depth[8] < by_depth[16]


def test_a_clip_names_the_recording_and_the_dynamic_it_stands_for(priced_run: RunInputs) -> None:
    task, context, encodings = _priced(priced_run, RankingGrid(depths=(BitDepth.SIXTEEN,), rate_steps=0))

    clip = clip_renditions(task, encodings, context)

    assert (clip.pitch, clip.key, clip.velocity) == (
        task.pitch,
        task.representative_key,
        task.representative_event.velocity,
    )


def test_a_rebuilt_member_is_the_audio_its_distortion_was_read_on(priced_run: RunInputs) -> None:
    task, context, encodings = _priced(priced_run, RankingGrid(depths=(BitDepth.SIXTEEN, BitDepth.EIGHT), rate_steps=0))
    clip = clip_renditions(task, encodings, context)

    twice = [rendered(clip, clip.renditions[0], context) for _ in range(2)]

    assert np.array_equal(twice[0], twice[1])


def test_the_recording_is_held_for_the_length_the_class_is_scored_over(priced_run: RunInputs) -> None:
    task, context, encodings = _priced(priced_run, RankingGrid(depths=(BitDepth.SIXTEEN,), rate_steps=0))
    clip = clip_renditions(task, encodings, context)

    held = reference(clip, context)

    assert held.size == pytest.approx(clip.event.duration_s * context.sample_rate, abs=1)


def test_a_member_is_rendered_over_the_stretch_the_recording_is_judged_on(priced_run: RunInputs) -> None:
    task, context, encodings = _priced(priced_run, RankingGrid(depths=(BitDepth.SIXTEEN,), rate_steps=0))
    clip = clip_renditions(task, encodings, context)

    assert rendered(clip, clip.renditions[0], context).size == reference(clip, context).size
