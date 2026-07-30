from __future__ import annotations

from collections.abc import Callable

import pytest

from optisample.config.tracker import TrackerFormat
from optisample.dsp.surrogate import EncodingParams
from optisample.io.tracker.target import ExportTarget
from optisample.keys import SampleKey
from optisample.music import sounded_note
from optisample.optimize.export.coverage import covered_routing, key_coverage, played_keys
from optisample.optimize.plans import FIRST_LAYER, SampleUnit
from trackmod.core.instruments.keymap import KeyAssignment, Keymap, routed_keymap
from trackmod.core.notes.pitch import Note

_LOWEST_PLAYED = 60
_HIGHEST_PLAYED = 72
_SECOND_SAMPLE = 1
_BOTTOM_KEY = 12  # the lowest key either format numbers, which is MIDI's own C-0
_RECORDED_VELOCITY = 100


def _routing(target: ExportTarget, *pitches: int) -> dict[Note, KeyAssignment]:
    """One sample per pitch, each rooted at its own recording, as a plan's own units would route them."""
    return {
        target.key(pitch): KeyAssignment(sample=sample, note=sounded_note(target.key(pitch), target.key(pitch)))
        for sample, pitch in enumerate(pitches)
    }


def _numbered_keys(target: ExportTarget) -> list[Note]:
    return [target.key(pitch) for pitch in range(target.min_pitch, target.max_pitch + 1)]


def _unit(pitch: int, keys: tuple[int, ...]) -> SampleUnit:
    return SampleUnit(
        label=f"p{pitch}",
        representative_key=SampleKey(pitch, _RECORDED_VELOCITY),
        layer=FIRST_LAYER,
        keys=keys,
        params=EncodingParams(target_rate=22_050, depth_bits=16),
        frames=0,
        stored_bytes=0,
        distortion=0.0,
        objective_share=0.0,
        hull_size=0,
        weight=0.0,
    )


def _per_sample_offsets(covered: dict[Note, KeyAssignment]) -> dict[int, set[int]]:
    """Every interval each sample is sounded at, which a format tuning it once needs to hold at one."""
    offsets: dict[int, set[int]] = {}
    for key, assignment in covered.items():
        offsets.setdefault(assignment.sample, set()).add(key.value - assignment.note.value)

    return offsets


def test_every_key_the_format_numbers_is_answered(target: ExportTarget) -> None:
    """A note reaching a key the material never played sounds the recording nearest it."""
    covered = covered_routing(_routing(target, _LOWEST_PLAYED, _HIGHEST_PLAYED), target)
    assert set(covered) == set(_numbered_keys(target))


def test_the_keys_the_material_played_keep_their_own_recording(target: ExportTarget) -> None:
    routing = _routing(target, _LOWEST_PLAYED, _HIGHEST_PLAYED)
    covered = covered_routing(routing, target)
    assert all(covered[key] == assignment for key, assignment in routing.items())


def test_the_lowest_recording_reaches_the_bottom_key_and_the_highest_the_top(target: ExportTarget) -> None:
    covered = covered_routing(_routing(target, _LOWEST_PLAYED, _HIGHEST_PLAYED), target)
    assert covered[target.key(target.min_pitch)].sample == 0
    assert covered[target.key(target.max_pitch)].sample == _SECOND_SAMPLE


def test_a_gap_is_split_between_its_neighbours_with_the_tie_going_down(target: ExportTarget) -> None:
    """Equally distant recordings hand the key to the lower one, which is the transposition priced for."""
    covered = covered_routing(_routing(target, _LOWEST_PLAYED, _HIGHEST_PLAYED), target)
    midpoint = (_LOWEST_PLAYED + _HIGHEST_PLAYED) // 2
    assert covered[target.key(midpoint)].sample == 0
    assert covered[target.key(midpoint + 1)].sample == _SECOND_SAMPLE


def test_every_key_of_one_sample_sounds_it_at_a_single_offset(target: ExportTarget) -> None:
    """FastTracker 2 tunes a sample once for every key reaching it, so a filled key holds that offset."""
    covered = covered_routing(_routing(target, _LOWEST_PLAYED, _HIGHEST_PLAYED), target)
    offsets = _per_sample_offsets(covered)
    assert len(offsets) == 2  # both recordings answer keys, so both offsets are measured
    assert all(len(spread) == 1 for spread in offsets.values())


def test_a_routing_answering_nothing_stays_empty(target: ExportTarget) -> None:
    """An instrument the allocation stored nothing for has no recording to reach a key with."""
    assert covered_routing({}, target) == {}


def test_a_key_whose_sounded_note_leaves_the_tracker_range_stays_silent(target: ExportTarget) -> None:
    """Reaching far above one deep recording asks for a note no tracker numbers, so that key waits."""
    covered = covered_routing(_routing(target, _BOTTOM_KEY), target)
    assert target.key(_BOTTOM_KEY) in covered
    assert target.key(target.max_pitch) not in covered


@pytest.mark.parametrize("tracker_format", list(TrackerFormat))
def test_each_format_is_answered_over_its_own_keyboard(
    retarget: Callable[[TrackerFormat], ExportTarget], tracker_format: TrackerFormat
) -> None:
    """The two formats number different stretches of keyboard, and each is filled over its own."""
    target = retarget(tracker_format)
    covered = covered_routing(_routing(target, _LOWEST_PLAYED, _HIGHEST_PLAYED), target)
    assert len(covered) == target.max_pitch - target.min_pitch + 1


def test_a_key_beyond_one_recording_is_answered_by_a_further_one_that_reaches_it(target: ExportTarget) -> None:
    """A recording too far for a tracker to name hands the key on rather than leaving it silent."""
    covered = covered_routing(_routing(target, _BOTTOM_KEY, _HIGHEST_PLAYED), target)
    top = covered[target.key(target.max_pitch)]
    assert top.sample == _SECOND_SAMPLE  # the deep recording cannot name the top key, so the high one does
    assert covered[target.key(_BOTTOM_KEY)].sample == 0


def _keymap(target: ExportTarget, *pitches: int) -> Keymap:
    return routed_keymap(covered_routing(_routing(target, *pitches), target))


def test_coverage_reports_the_keyboard_the_written_keymaps_answer(target: ExportTarget) -> None:
    coverage = key_coverage([_keymap(target, _LOWEST_PLAYED, _HIGHEST_PLAYED)], target, played=2)
    assert coverage.numbered == target.max_pitch - target.min_pitch + 1
    assert coverage.answered == coverage.numbered
    assert coverage.filled == coverage.numbered - coverage.played


def test_coverage_states_what_the_least_covered_instrument_answers(target: ExportTarget) -> None:
    """A reader takes the figure as true of every written instrument, so the smallest one sets it."""
    whole = _keymap(target, _LOWEST_PLAYED, _HIGHEST_PLAYED)
    partial = _keymap(target, _BOTTOM_KEY)
    coverage = key_coverage([whole, partial], target, played=2)
    assert coverage.answered < target.max_pitch - target.min_pitch + 1


def test_an_instrument_writing_nothing_answers_no_key(target: ExportTarget) -> None:
    assert key_coverage([], target, played=0).answered == 0


def test_played_keys_counts_each_key_once_across_the_layers_reaching_it() -> None:
    """A key stored in two velocity bands is one key of keyboard, played twice over."""
    units = (_unit(60, (60, 61)), _unit(61, (60, 61)), _unit(72, (72,)))
    assert played_keys(units) == 3
