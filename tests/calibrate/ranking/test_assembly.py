from __future__ import annotations

from dataclasses import replace

from optisample.calibrate.ranking import (
    PairAxis,
    PairQuota,
    RankingSettings,
    assemble_ranking,
)
from optisample.optimize.orchestrate import RunInputs
from optisample.progress import SilentProgress


def test_every_played_pitch_is_priced(priced_run: RunInputs, ranking_settings: RankingSettings) -> None:
    built = assemble_ranking(priced_run, ranking_settings, SilentProgress())

    assert {clip.pitch for clip in built.clips} == {task.pitch for task in priced_run.tasks}


def test_a_set_states_how_many_encodings_it_chose_from(
    priced_run: RunInputs,
    ranking_settings: RankingSettings,
) -> None:
    built = assemble_ranking(priced_run, ranking_settings, SilentProgress())

    assert built.priced == sum(len(clip.renditions) for clip in built.clips)


def test_the_encodings_the_sweep_prices_are_all_in_the_set(
    priced_run: RunInputs,
    ranking_settings: RankingSettings,
) -> None:
    built = assemble_ranking(priced_run, ranking_settings, SilentProgress())

    swept = priced_run.reduction.encodings()
    for clip in built.clips:
        priced = {rendition.params for rendition in clip.renditions}
        assert set(swept[clip.pitch]) <= priced


def test_the_set_asks_about_the_depth_the_sweep_holds_fixed(
    priced_run: RunInputs,
    ranking_settings: RankingSettings,
) -> None:
    built = assemble_ranking(
        priced_run,
        replace(ranking_settings, quota=PairQuota(loop=0, rate=0, depth=4, trade=0)),
        SilentProgress(),
    )

    assert [pair.axis for pair in built.pairs] == [PairAxis.DEPTH] * len(built.pairs)
    assert built.pairs


def test_a_set_carries_what_rebuilds_its_own_audio(
    priced_run: RunInputs,
    ranking_settings: RankingSettings,
) -> None:
    built = assemble_ranking(priced_run, ranking_settings, SilentProgress())

    assert built.context is priced_run.context
