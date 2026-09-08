from colorsys import rgb_to_hsv

import pytest

from notebooks.utils import clusters
from optisample.cluster.partition import partition, sweep
from optisample.cluster.representative import grouping
from optisample.cluster.selection import selection
from optisample.config import OptiConfig
from optisample.config.cluster import Representative
from optisample.keys import SampleKey
from tests.notebooks.conftest import Clustered

_BLOCKS = 4  # readings a space is assembled from, each laying down a span of its own
_WHOLE = 1.0  # what the shares of the spread come to between them
_QUIETEST = 0.25  # the saturation the softest take is drawn at
_LOUDEST = 1.0  # the saturation the hardest-struck take fills
_CHANNEL = 255  # what one color channel is spelled out of in a hex triplet
_HEX_TRIPLET = 7  # characters a color is spelled in, the hash and the three channels
_MIDDLE_C = 60
_TOP_VELOCITY = 127


def _hsv(color: str) -> tuple[float, float, float]:
    """A drawn color read back as the hue, the saturation and the value a key turned it to."""
    channels = (int(color[place : place + 2], 16) / _CHANNEL for place in (1, 3, 5))
    return rgb_to_hsv(*channels)


def test_a_group_is_named_after_the_key_the_take_standing_for_it_plays() -> None:
    """One spelling, so a legend entry, a table cell and a dropdown option all name a group by its key."""
    assert clusters.group_name(SampleKey(pitch=_MIDDLE_C, velocity=100)) == "p060_v100"
    assert clusters.group_name(SampleKey(pitch=9, velocity=7)) == "p009_v007"


def test_group_names_sort_into_keyboard_order() -> None:
    """The pitch leads zero-padded, so a sorted legend reads up the keys rather than by the digits."""
    keys = [SampleKey(pitch=pitch, velocity=64) for pitch in (9, 60, 108)]
    names = [clusters.group_name(key) for key in keys]

    assert sorted(names) == names


def test_a_groups_color_turns_with_its_pitch_and_fills_with_its_velocity() -> None:
    """The keyboard read as color: where a group sits sets the hue and how hard it was struck fills it in."""
    low = _hsv(clusters.group_color(SampleKey(pitch=24, velocity=_TOP_VELOCITY)))
    high = _hsv(clusters.group_color(SampleKey(pitch=96, velocity=_TOP_VELOCITY)))
    soft = _hsv(clusters.group_color(SampleKey(pitch=24, velocity=0)))

    assert low[0] < high[0]
    assert low[1] == pytest.approx(_LOUDEST, abs=1e-2)
    assert soft[1] == pytest.approx(_QUIETEST, abs=1e-2)
    assert soft[0] == pytest.approx(low[0], abs=1e-2)


def test_the_softest_take_keeps_a_color_of_its_own() -> None:
    """The saturation opens a quarter of the way up, which is what tells quiet groups apart from each other."""
    quiet = [clusters.group_color(SampleKey(pitch=pitch, velocity=1)) for pitch in (24, 60, 96)]

    assert len(set(quiet)) == len(quiet)
    assert all(len(color) == _HEX_TRIPLET and color.startswith("#") for color in quiet)


def test_one_key_is_drawn_the_same_color_wherever_it_is_read() -> None:
    """The color is read off the key alone, so a stage, a cut and a dataset all draw that key alike."""
    assert clusters.group_color(SampleKey(pitch=_MIDDLE_C, velocity=100)) == clusters.group_color(
        SampleKey(pitch=_MIDDLE_C, velocity=100)
    )


def test_two_sets_holding_a_take_of_one_name_are_told_apart(gathered: Clustered) -> None:
    """A file stem is unique inside its own set alone, so the name a table uses across sets leads with the set."""
    recordings = gathered.described.recordings
    named = [clusters.sample_name(recording) for recording in recordings]

    assert len({recording.label for recording in recordings}) < len(recordings)
    assert len(set(named)) == len(recordings)
    assert all(name.startswith(recording.source.label) for name, recording in zip(named, recordings, strict=True))


def test_a_title_states_every_set_one_space_was_gathered_from(gathered: Clustered) -> None:
    """A spinner, a picture and a summary line all say which sets they answer for, in the order they were read."""
    sources = gathered.described.corpus.sources
    label = clusters.sources_label(sources)

    assert clusters.sources_label(sources[:1]) == sources[0].label
    assert all(source.label in label for source in sources)
    assert label.index(sources[0].label) < label.index(sources[1].label)


def test_groups_stood_for_by_one_key_are_told_apart_by_their_names(clustered: Clustered) -> None:
    """A corpus can place two renditions of a note apart, and a panel still needs one word per group."""
    doubled = clustered.groups + clustered.groups
    named = clusters.named_groups(clustered.described, doubled)

    assert len({group.name for group in named}) == len(doubled)
    assert [group.name for group in named[: len(clustered.groups)]] == [group.name for group in clustered.named]
    assert all(group.name.endswith("#2") for group in named[len(clustered.groups) :])


def test_a_named_group_carries_the_cut_it_was_read_off(clustered: Clustered) -> None:
    """Naming is what the panels add to a cut, so the group behind a name is the one the space settled."""
    assert all(named.group is group for named, group in zip(clustered.named, clustered.groups, strict=True))
    assert clusters.group_colors(clustered.named) == {group.name: group.color for group in clustered.named}


def test_a_group_is_named_and_colored_by_the_take_standing_for_it(clustered: Clustered) -> None:
    """The word and the color both come off the representative, so a legend says what a reader will hear."""
    for named in clustered.named:
        key = clustered.described.recordings[named.group.representative].key

        assert named.name == clusters.group_name(key)
        assert named.color == clusters.group_color(key)


def test_every_recording_gets_one_row_naming_the_group_it_fell_into(clustered: Clustered) -> None:
    """A point row stands for one take, so what is hovered and what is listed are the same reading."""
    rows = clusters.point_rows(clustered.described, clustered.named, clustered.distances)

    assert len(rows) == clustered.described.size
    assert {str(row["group"]) for row in rows} == {group.name for group in clustered.named}
    for row, recording in zip(rows, clustered.described.recordings, strict=True):
        assert row["sample"] == recording.label
        assert row["set"] == recording.source.label
        assert row["pitch"] == recording.key.pitch
        assert row["playing_s"] == pytest.approx(recording.weight)


def test_a_gathered_space_reads_every_point_back_to_the_set_it_came_from(gathered: Clustered) -> None:
    """One field holds several sets, so every point carries the column a reader colors and filters them by."""
    rows = clusters.point_rows(gathered.described, gathered.named, gathered.distances)

    assert len(rows) == gathered.described.size
    assert {str(row["set"]) for row in rows} == {source.label for source in gathered.described.corpus.sources}


def test_the_take_standing_for_a_group_is_the_one_its_row_names_a_representative(clustered: Clustered) -> None:
    """Exactly the members the rule picks read as representatives, so the picture and the tables agree."""
    rows = clusters.point_rows(clustered.described, clustered.named, clustered.distances)
    marked = {index for index, row in enumerate(rows) if row["role"] == "representative"}

    assert marked == {group.representative for group in clustered.groups}


def test_a_representative_stands_no_distance_from_itself(clustered: Clustered) -> None:
    """The distance a row states is read in the space the group was cut in, so its own take reads zero."""
    rows = clusters.point_rows(clustered.described, clustered.named, clustered.distances)

    for group in clustered.groups:
        assert rows[group.medoid]["to_medoid"] == pytest.approx(0.0)
        assert rows[group.representative]["to_representative"] == pytest.approx(0.0)


def test_a_group_row_spans_the_pitches_and_the_playing_time_its_members_carry(clustered: Clustered) -> None:
    """A group reads as what it gathered, so the sizes and the playing times sum back to the corpus."""
    rows = clusters.group_rows(clustered.described, clustered.named)

    assert len(rows) == len(clustered.groups)
    assert sum(int(row["size"]) for row in rows) == clustered.described.size
    assert sum(float(row["playing_s"]) for row in rows) == pytest.approx(clustered.described.corpus.weights.sum())
    for row, group in zip(rows, clustered.groups, strict=True):
        pitches = [clustered.described.recordings[member].key.pitch for member in group.members.tolist()]
        assert row["pitch_lo"] == min(pitches)
        assert row["pitch_hi"] == max(pitches)
        assert row["stands_for_it"] == clusters.sample_name(clustered.described.recordings[group.representative])


def test_a_group_counts_the_sets_its_members_were_gathered_from(gathered: Clustered) -> None:
    """A group holding takes of two sets is a sound they share, which is what one shared space is read for."""
    rows = clusters.group_rows(gathered.described, gathered.named)
    recordings = gathered.described.recordings

    for row, group in zip(rows, gathered.groups, strict=True):
        assert row["sets"] == len({recordings[member].source for member in group.members.tolist()})

    assert max(int(row["sets"]) for row in rows) == len(gathered.described.corpus.sources)


def test_members_are_listed_from_the_medoid_outwards(clustered: Clustered) -> None:
    """Ordering by that distance puts the take standing for the group first and the one it covers least last."""
    rows = clusters.point_rows(clustered.described, clustered.named, clustered.distances)
    for named in clustered.named:
        listed = clusters.member_rows(rows, named)

        assert len(listed) == named.group.size
        assert listed[0]["sample"] == rows[named.group.medoid]["sample"]
        assert [float(row["to_medoid"]) for row in listed] == sorted(float(row["to_medoid"]) for row in listed)


def test_the_blocks_shares_of_the_spread_come_to_the_whole(clustered: Clustered) -> None:
    """Every column of the space belongs to one block, so what the blocks account for is all of it."""
    rows = clusters.block_rows(clustered.space)

    assert len(rows) == _BLOCKS
    assert sum(int(row["columns"]) for row in rows) == clustered.space.dimensions
    assert sum(float(row["share"]) for row in rows) == pytest.approx(_WHOLE)


def test_a_block_given_no_say_holds_no_spread(clustered: Clustered, config: OptiConfig) -> None:
    """A weight of zero leaves a block's columns standing at one point, which is the whole of its say."""
    silent = clustered.described.space(config.cluster.space.model_copy(update={"envelope_weight": 0.0}))
    rows = {str(row["block"]): row for row in clusters.block_rows(silent)}

    assert rows["envelope"]["spread"] == pytest.approx(0.0)
    assert rows["envelope"]["share"] == pytest.approx(0.0)


def test_each_group_reads_a_spread_in_each_block(clustered: Clustered) -> None:
    """One row per group and block, which is what says which reading a group was drawn together by."""
    rows = clusters.group_block_rows(clustered.space, clustered.named)

    assert len(rows) == len(clustered.groups) * _BLOCKS
    assert all(float(row["spread"]) >= 0.0 for row in rows)


def test_the_sweep_marks_the_count_on_screen(clustered: Clustered, config: OptiConfig) -> None:
    """A reader picks a count off the curve, so the one being shown is the one marked on it."""
    climbed = sweep(clustered.space.coordinates, config.cluster.partition)
    chosen = len(clustered.groups)
    rows = clusters.sweep_rows(climbed, chosen=chosen)

    assert [int(row["groups"]) for row in rows] == [found.groups for found in climbed]
    assert [int(row["groups"]) for row in rows if bool(row["on_screen"])] == [chosen]


def test_the_anchors_state_which_depths_a_take_reached_and_which_the_space_reads(
    clustered: Clustered, config: OptiConfig
) -> None:
    """A take short of a depth its corpus shares reads as the row holding what it last had."""
    depths = config.cluster.descriptor.anchor_depths_db
    rows = clusters.anchor_rows(clustered.described.descriptors[0], depths, clustered.space)

    assert [float(row["depth_db"]) for row in rows] == list(depths)
    assert sum(bool(row["in_space"]) for row in rows) == int(clustered.space.depths.sum())
    assert all(float(row["time_s"]) > 0.0 for row in rows)


def test_one_take_states_its_own_readings_on_a_single_line(clustered: Clustered) -> None:
    """The examine panel reads a take by one row, so every number it shows comes off one descriptor."""
    rows = clusters.descriptor_rows(clustered.described.descriptors[0], clustered.described.recordings[0])

    assert len(rows) == 1
    assert rows[0]["sample"] == clusters.sample_name(clustered.described.recordings[0])
    assert rows[0]["columns"] == clustered.described.descriptors[0].columns


def test_a_take_reads_its_distance_to_every_group_and_owns_exactly_one(clustered: Clustered) -> None:
    """The company a take nearly kept reads beside its own, which is where a cut is finely balanced."""
    rows = clusters.reach_rows(0, clustered.described, clustered.named, clustered.distances)

    assert len(rows) == len(clustered.groups)
    assert sum(bool(row["own"]) for row in rows) == 1
    owned = next(row for row in rows if bool(row["own"]))
    assert owned["to_nearest"] == pytest.approx(0.0)


def test_a_reach_row_names_the_take_standing_for_each_group_by_its_own_set(gathered: Clustered) -> None:
    """The company a take nearly kept is named across sets, so the take it would answer to is unambiguous."""
    rows = clusters.reach_rows(0, gathered.described, gathered.named, gathered.distances)

    for row, named in zip(rows, gathered.named, strict=True):
        assert row["stands_for_it"] == clusters.sample_name(gathered.described.recordings[named.group.representative])


def test_a_pick_names_the_set_its_dataset_is_written_out_of(gathered: Clustered) -> None:
    """A selection cut across sets is written one dataset apiece, so each row says which one it lands in."""
    chosen = selection(gathered.described.corpus, gathered.groups)
    rows = clusters.pick_rows(chosen, gathered.named)

    for row, pick in zip(rows, chosen.picks, strict=True):
        assert row["set"] == pick.source.label
        assert row["sample"] == pick.recording.label

    assert {str(row["set"]) for row in rows} <= {source.label for source in gathered.described.corpus.sources}


def test_a_chosen_take_is_listed_beside_the_share_of_the_corpus_it_stands_for(clustered: Clustered) -> None:
    """A written selection reads as one row per take, so what is about to be written is what is shown."""
    chosen = selection(clustered.described.corpus, clustered.groups)
    rows = clusters.pick_rows(chosen, clustered.named)

    assert len(rows) == len(clustered.groups)
    assert sum(int(row["members"]) for row in rows) == clustered.described.size
    assert sum(float(row["playing_s"]) for row in rows) == pytest.approx(clustered.described.corpus.weights.sum())
    for row, named in zip(rows, clustered.named, strict=True):
        standing = clustered.described.recordings[named.group.representative]

        assert row["group"] == named.name
        assert row["sample"] == standing.label
        assert row["set"] == standing.source.label


def test_a_selection_read_against_another_cut_is_rejected(clustered: Clustered) -> None:
    """A pick reads under its own group's word, so a count of picks other than of groups states a mismatch."""
    chosen = selection(clustered.described.corpus, clustered.groups)

    with pytest.raises(ValueError):
        clusters.pick_rows(chosen, clustered.named[:-1])


def test_a_representative_rule_names_the_member_the_rows_stand_by(clustered: Clustered) -> None:
    """Naming another rule moves which take a group is stood for by, and every row follows it."""
    leaning = clustered.cutting.model_copy(update={"representative": Representative.WEIGHTED_MEDOID})
    regrouped = grouping(
        clustered.space.coordinates,
        partition(clustered.space.coordinates, groups=len(clustered.groups), config=leaning).labels,
        clustered.described.readings,
        config=leaning,
    )
    rows = clusters.group_rows(clustered.described, clusters.named_groups(clustered.described, regrouped))

    for row, group in zip(rows, regrouped, strict=True):
        assert row["stands_for_it"] == clusters.sample_name(clustered.described.recordings[group.weighted_medoid])
