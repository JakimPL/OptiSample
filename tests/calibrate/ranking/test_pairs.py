from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from optisample.calibrate.ranking import (
    ClipRenditions,
    ListeningPair,
    PairAxis,
    PairQuota,
    Rendition,
    Side,
    listening_pairs,
)
from optisample.dsp.surrogate import UNLOOPED, EncodingParams
from optisample.optimize.tasks import Event, PitchTask

_SEED = 137
_MATCHED = 0.15  # the byte tolerance these fixtures' trade pairs are asked under
_BASE_BYTES = 10_000
_UNREPEATED = 0
_REPEATED = 3
_ASKED = 20
_LOUDNESS = -23.0
_LOOP_OFFERS = 8


@dataclass(frozen=True)
class _Encoding:
    """One member of a clip's grid as a test states it: what it asks for, and what it earned."""

    rate: int
    depth: int
    loop: int | None
    stored_bytes: int
    distortion: float
    loudness: float = _LOUDNESS
    compress: bool = False


def _rendition(encoding: _Encoding) -> Rendition:
    return Rendition(
        params=EncodingParams(
            target_rate=encoding.rate,
            depth_bits=encoding.depth,
            trim_s=2.0,
            loop_index=encoding.loop,
            compress=encoding.compress,
        ),
        stored_bytes=encoding.stored_bytes,
        distortion=encoding.distortion,
        loudness_lufs=encoding.loudness,
    )


def _clip(pitch: int, encodings: Sequence[_Encoding], events: tuple[Event, ...]) -> ClipRenditions:
    """A clip standing in for one priced pitch, its task carrying only what a pair reads off it."""
    task = PitchTask(
        pitch,
        1.0,
        events[0].reference_key,
        events[0].reference,
        (events[0].reference_key,),
        events,
    )
    return ClipRenditions(task=task, event=events[0], renditions=tuple(_rendition(one) for one in encodings))


@pytest.fixture
def event(scored_event: Event) -> Event:
    return scored_event


@pytest.fixture
def single_axis(event: Event) -> ClipRenditions:
    """One clip offering a loop step, a rate step and a depth step off the same stored span."""
    return _clip(
        60,
        (
            _Encoding(16_000, 16, UNLOOPED, _BASE_BYTES, 1.0),
            _Encoding(16_000, 16, 0, _BASE_BYTES // 2, 1.4),
            _Encoding(11_025, 16, UNLOOPED, 6_900, 1.2),
            _Encoding(16_000, 8, UNLOOPED, _BASE_BYTES // 2, 1.6),
        ),
        (event,),
    )


@pytest.fixture
def many_loops(event: Event) -> ClipRenditions:
    """One clip offering enough loop steps for a set to have room to put questions a second time."""
    return _clip(
        60,
        tuple(_Encoding(16_000, 16, index, _BASE_BYTES, 1.0 + index / 10) for index in range(_LOOP_OFFERS)),
        (event,),
    )


def _axes(clips: Sequence[ClipRenditions], quota: PairQuota) -> Counter[PairAxis]:
    pairs = listening_pairs(clips, quota, byte_tolerance=_MATCHED, repeats=_UNREPEATED, seed=_SEED)
    return Counter(pair.axis for pair in pairs)


def _loop_set(clip: ClipRenditions, repeats: int, seed: int) -> tuple[ListeningPair, ...]:
    """A set of loop questions large enough to hold repeats well apart from their sources."""
    return listening_pairs(
        [clip],
        PairQuota(loop=_ASKED, rate=0, depth=0, compress=0, trade=0),
        byte_tolerance=_MATCHED,
        repeats=repeats,
        seed=seed,
    )


def _members(pair: ListeningPair) -> tuple[str, ...]:
    """The two encodings a pair puts, read apart from which side each took."""
    return tuple(sorted((repr(pair.first.params), repr(pair.second.params))))


def test_a_pair_differing_in_one_axis_asks_about_that_axis(single_axis: ClipRenditions) -> None:
    asked = _axes([single_axis], PairQuota(loop=1, rate=1, depth=1, compress=0, trade=0))

    assert asked == Counter({PairAxis.LOOP: 1, PairAxis.RATE: 1, PairAxis.DEPTH: 1})


def test_a_quota_bounds_how_many_of_each_question_a_listener_is_asked(single_axis: ClipRenditions) -> None:
    asked = _axes([single_axis], PairQuota(loop=1, rate=0, depth=1, compress=0, trade=0))

    assert asked == Counter({PairAxis.LOOP: 1, PairAxis.DEPTH: 1})


def test_two_degradations_of_nearly_equal_size_are_asked_as_a_trade(single_axis: ClipRenditions) -> None:
    asked = _axes([single_axis], PairQuota(loop=0, rate=0, depth=0, compress=0, trade=4))

    assert asked[PairAxis.TRADE] > 0


def test_a_trade_pairs_two_encodings_within_the_tolerance_of_one_another(single_axis: ClipRenditions) -> None:
    pairs = listening_pairs(
        [single_axis],
        PairQuota(loop=0, rate=0, depth=0, compress=0, trade=4),
        byte_tolerance=_MATCHED,
        repeats=_UNREPEATED,
        seed=_SEED,
    )

    for pair in pairs:
        larger = max(pair.first.stored_bytes, pair.second.stored_bytes)
        assert abs(pair.first.stored_bytes - pair.second.stored_bytes) <= _MATCHED * larger


def test_encodings_of_different_size_along_several_axes_are_left_unasked(event: Event) -> None:
    lopsided = _clip(
        60,
        (
            _Encoding(16_000, 16, UNLOOPED, _BASE_BYTES, 1.0),
            _Encoding(8_000, 8, 0, 900, 3.0),
        ),
        (event,),
    )

    assert _axes([lopsided], PairQuota(loop=4, rate=4, depth=4, compress=0, trade=4)) == Counter()


def test_the_composite_names_the_side_it_calls_closer(single_axis: ClipRenditions) -> None:
    pairs = listening_pairs(
        [single_axis],
        PairQuota(loop=0, rate=1, depth=0, compress=0, trade=0),
        byte_tolerance=_MATCHED,
        repeats=_UNREPEATED,
        seed=_SEED,
    )

    pair = pairs[0]
    closer = pair.first if pair.composite_side is Side.A else pair.second
    assert closer.distortion == min(pair.first.distortion, pair.second.distortion)


def test_a_pair_states_how_far_apart_the_composite_puts_its_two_sides(single_axis: ClipRenditions) -> None:
    pairs = listening_pairs(
        [single_axis],
        PairQuota(loop=1, rate=0, depth=0, compress=0, trade=0),
        byte_tolerance=_MATCHED,
        repeats=_UNREPEATED,
        seed=_SEED,
    )

    assert pairs[0].margin == pytest.approx(abs(pairs[0].first.distortion - pairs[0].second.distortion))


def test_one_seed_reproduces_the_whole_set(single_axis: ClipRenditions) -> None:
    quota = PairQuota(loop=1, rate=1, depth=1, compress=0, trade=2)
    sides = [
        [
            (pair.axis, pair.first.params, pair.second.params)
            for pair in listening_pairs([single_axis], quota, byte_tolerance=_MATCHED, repeats=_UNREPEATED, seed=_SEED)
        ]
        for _ in range(2)
    ]

    assert sides[0] == sides[1]


def test_another_seed_settles_the_sides_differently(many_loops: ClipRenditions) -> None:
    quota = PairQuota(loop=_LOOP_OFFERS, rate=0, depth=0, compress=0, trade=0)
    firsts = [
        [
            pair.first.params.loop_index
            for pair in listening_pairs([many_loops], quota, byte_tolerance=_MATCHED, repeats=_UNREPEATED, seed=seed)
        ]
        for seed in (_SEED, _SEED + 1)
    ]

    assert firsts[0] != firsts[1]


def test_a_quota_past_what_the_clips_offer_asks_every_pair_there_is(single_axis: ClipRenditions) -> None:
    asked = _axes([single_axis], PairQuota(loop=99, rate=99, depth=99, compress=0, trade=99))

    assert asked[PairAxis.LOOP] == 1


def test_the_listening_is_spread_over_the_clips_offering_it(event: Event) -> None:
    clips = [
        _clip(
            pitch,
            (
                _Encoding(16_000, 16, UNLOOPED, _BASE_BYTES, 1.0),
                _Encoding(16_000, 16, 0, _BASE_BYTES, 1.0 + pitch / 100),
                _Encoding(16_000, 16, 1, _BASE_BYTES, 1.0 + pitch / 50),
            ),
            (event,),
        )
        for pitch in (60, 62, 64)
    ]
    pairs = listening_pairs(
        clips,
        PairQuota(loop=3, rate=0, depth=0, compress=0, trade=0),
        byte_tolerance=_MATCHED,
        repeats=_UNREPEATED,
        seed=_SEED,
    )

    assert len({pair.clip.pitch for pair in pairs}) == len(clips)


def test_a_quota_asking_nothing_writes_no_listening(single_axis: ClipRenditions) -> None:
    assert (
        listening_pairs(
            [single_axis], PairQuota(0, 0, 0, 0, 0), byte_tolerance=_MATCHED, repeats=_UNREPEATED, seed=_SEED
        )
        == ()
    )


def test_a_quota_states_the_listening_it_costs() -> None:
    assert PairQuota(loop=1, rate=2, depth=3, compress=0, trade=4).total == 10


def test_a_repeated_question_is_put_a_second_time(many_loops: ClipRenditions) -> None:
    pairs = _loop_set(many_loops, _REPEATED, _SEED)

    assert len(pairs) == len({pair.question_id for pair in pairs}) + _REPEATED


def test_a_repeat_stands_well_after_the_question_it_repeats(many_loops: ClipRenditions) -> None:
    pairs = _loop_set(many_loops, _REPEATED, _SEED)

    met: dict[int, int] = {}
    gaps = []
    for place, pair in enumerate(pairs):
        if pair.question_id in met:
            gaps.append(place - met[pair.question_id])

        met[pair.question_id] = place

    assert min(gaps) >= _ASKED // 2


def test_the_two_occurrences_of_a_question_put_the_same_two_encodings(many_loops: ClipRenditions) -> None:
    put: dict[int, set[tuple[str, ...]]] = {}
    for pair in _loop_set(many_loops, _REPEATED, _SEED):
        put.setdefault(pair.question_id, set()).add(_members(pair))

    assert [len(occurrences) for occurrences in put.values()] == [1] * len(put)


def test_a_repeat_draws_its_sides_afresh(many_loops: ClipRenditions) -> None:
    swapped = []
    for seed in range(_SEED, _SEED + 4):
        leading: dict[int, str] = {}
        for pair in _loop_set(many_loops, _REPEATED, seed):
            first = repr(pair.first.params)
            if pair.question_id in leading:
                swapped.append(leading[pair.question_id] != first)

            leading[pair.question_id] = first

    assert any(swapped)


def test_a_short_set_puts_as_many_questions_again_as_it_has_room_for(single_axis: ClipRenditions) -> None:
    pairs = listening_pairs(
        [single_axis],
        PairQuota(loop=1, rate=1, depth=0, compress=0, trade=0),
        byte_tolerance=_MATCHED,
        repeats=9,
        seed=_SEED,
    )

    assert len(pairs) == 3


def test_two_encodings_moving_the_grid_and_the_dynamics_at_once_are_left_unasked(event: Event) -> None:
    tied = _clip(
        60,
        (
            _Encoding(16_000, 16, UNLOOPED, _BASE_BYTES, 1.0, compress=False),
            _Encoding(16_000, 8, UNLOOPED, _BASE_BYTES // 2, 1.6, compress=True),
        ),
        (event,),
    )

    assert _axes([tied], PairQuota(loop=4, rate=4, depth=4, compress=4, trade=4)) == Counter()


def test_a_pair_moving_the_compressor_alone_asks_about_the_dynamics(event: Event) -> None:
    dynamics = _clip(
        60,
        (
            _Encoding(16_000, 8, UNLOOPED, _BASE_BYTES, 1.0, compress=False),
            _Encoding(16_000, 8, UNLOOPED, _BASE_BYTES, 1.6, compress=True),
        ),
        (event,),
    )

    asked = _axes([dynamics], PairQuota(loop=0, rate=0, depth=0, compress=1, trade=0))

    assert asked == Counter({PairAxis.COMPRESS: 1})


def test_a_depth_pair_holds_the_compressor_where_it_was(event: Event) -> None:
    grid = _clip(
        60,
        (
            _Encoding(16_000, 16, UNLOOPED, _BASE_BYTES, 1.0, compress=False),
            _Encoding(16_000, 8, UNLOOPED, _BASE_BYTES // 2, 1.6, compress=False),
        ),
        (event,),
    )
    pairs = listening_pairs(
        [grid],
        PairQuota(loop=0, rate=0, depth=1, compress=0, trade=0),
        byte_tolerance=_MATCHED,
        repeats=_UNREPEATED,
        seed=_SEED,
    )

    assert pairs[0].first.params.compress == pairs[0].second.params.compress


def test_a_pair_states_how_much_louder_one_side_plays(event: Event) -> None:
    clip = _clip(
        60,
        (
            _Encoding(16_000, 16, UNLOOPED, _BASE_BYTES, 1.0, loudness=-22.0),
            _Encoding(16_000, 16, 0, _BASE_BYTES, 1.4, loudness=-24.0),
        ),
        (event,),
    )
    pairs = listening_pairs(
        [clip],
        PairQuota(loop=1, rate=0, depth=0, compress=0, trade=0),
        byte_tolerance=_MATCHED,
        repeats=_UNREPEATED,
        seed=_SEED,
    )

    assert abs(pairs[0].loudness_delta_lu) == pytest.approx(2.0)
