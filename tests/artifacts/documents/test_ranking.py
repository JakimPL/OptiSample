from __future__ import annotations

from collections.abc import Sequence

import pytest

from optisample.artifacts.documents.ranking import (
    RankingSetDocument,
    pair_directory,
    ranking_document,
)
from optisample.artifacts.serialize import json_text
from optisample.calibrate.ranking import (
    ClipRenditions,
    ListeningPair,
    PairAxis,
    Rendition,
    Side,
)
from optisample.dsp.surrogate import UNLOOPED, EncodingParams
from optisample.keys import SampleKey
from optisample.optimize.tasks import Event, PitchTask

_SAMPLE_RATE = 44_100
_SEED = 137
_PRICED = 11


def _rendition(*, rate: int, depth: int, loop: int | None, stored_bytes: int, distortion: float) -> Rendition:
    return Rendition(
        params=EncodingParams(target_rate=rate, depth_bits=depth, trim_s=1.0, loop_index=loop, compress=depth <= 8),
        stored_bytes=stored_bytes,
        distortion=distortion,
    )


@pytest.fixture
def clip(scored_event: Event) -> ClipRenditions:
    key = scored_event.reference_key
    task = PitchTask(60, 1.0, key, scored_event.reference, (key,), (scored_event,))
    return ClipRenditions(task=task, event=scored_event, renditions=())


@pytest.fixture
def pairs(clip: ClipRenditions) -> tuple[ListeningPair, ...]:
    return (
        ListeningPair(
            clip=clip,
            axis=PairAxis.DEPTH,
            first=_rendition(rate=16_000, depth=16, loop=UNLOOPED, stored_bytes=8_000, distortion=1.0),
            second=_rendition(rate=16_000, depth=8, loop=UNLOOPED, stored_bytes=4_000, distortion=1.5),
        ),
        ListeningPair(
            clip=clip,
            axis=PairAxis.LOOP,
            first=_rendition(rate=16_000, depth=16, loop=0, stored_bytes=3_000, distortion=2.0),
            second=_rendition(rate=16_000, depth=16, loop=UNLOOPED, stored_bytes=8_000, distortion=1.2),
        ),
    )


def _document(pairs: Sequence[ListeningPair]) -> RankingSetDocument:
    return ranking_document(
        pairs,
        instrument_id="piano",
        sample_rate=_SAMPLE_RATE,
        seed=_SEED,
        priced_encodings=_PRICED,
    )


def test_a_pair_directory_leads_with_its_place_in_the_set(pairs: Sequence[ListeningPair]) -> None:
    assert pair_directory(pairs[0], 7).startswith("007_")


def test_a_pair_directory_names_the_recording_both_sides_came_from(pairs: Sequence[ListeningPair]) -> None:
    assert pairs[0].clip.key.label in pair_directory(pairs[0], 0)


def test_a_pair_directory_carries_nothing_of_the_question_it_asks(pairs: Sequence[ListeningPair]) -> None:
    directories = [pair_directory(pair, index) for index, pair in enumerate(pairs)]

    assert not any(axis in directory for directory in directories for axis in PairAxis)


def test_the_manifest_decodes_both_sides_of_every_pair(pairs: Sequence[ListeningPair]) -> None:
    document = _document(pairs)

    stated = [(record.first.depth_bits, record.second.depth_bits) for record in document.pairs]
    assert stated == [(16, 8), (16, 16)]


def test_the_manifest_names_which_side_each_member_took(pairs: Sequence[ListeningPair]) -> None:
    document = _document(pairs)

    assert [(record.first.side, record.second.side) for record in document.pairs] == [(Side.A, Side.B)] * len(pairs)


def test_the_manifest_states_the_ranking_the_composite_gives(pairs: Sequence[ListeningPair]) -> None:
    document = _document(pairs)

    assert [record.composite_side for record in document.pairs] == [Side.A, Side.B]


def test_the_manifest_states_how_far_apart_the_composite_puts_a_pair(pairs: Sequence[ListeningPair]) -> None:
    document = _document(pairs)

    assert document.pairs[0].margin == pytest.approx(0.5)


def test_an_unlooped_member_names_no_region(pairs: Sequence[ListeningPair]) -> None:
    document = _document(pairs)

    assert document.pairs[0].first.loop_index is None


def test_the_manifest_states_what_the_pairs_were_chosen_from(pairs: Sequence[ListeningPair]) -> None:
    document = _document(pairs)

    assert (document.priced_encodings, document.seed, document.sample_rate) == (_PRICED, _SEED, _SAMPLE_RATE)


def test_the_manifest_round_trips_through_its_own_json(pairs: Sequence[ListeningPair]) -> None:
    document = _document(pairs)

    assert RankingSetDocument.model_validate_json(json_text(document)) == document
