import pytest

from notebooks.utils import clusters
from optisample.cluster.partition import sweep
from optisample.config import OptiConfig
from optisample.config.cluster import Representative
from tests.notebooks.conftest import Clustered

_BLOCKS = 4  # readings a space is assembled from, each laying down a span of its own
_WHOLE = 1.0  # what the shares of the spread come to between them


def test_a_group_is_spelled_the_same_way_in_every_panel() -> None:
    """One spelling, so a legend entry, a table cell and a hover all name the same group the same word."""
    assert clusters.group_name(0) == "g00"
    assert clusters.group_name(12) == "g12"


def test_every_recording_gets_one_row_naming_the_group_it_fell_into(clustered: Clustered) -> None:
    """A point row stands for one take, so what is hovered and what is listed are the same reading."""
    rows = clusters.point_rows(clustered.described, clustered.groups, clustered.distances, rule=clustered.rule)

    assert len(rows) == clustered.described.size
    assert {str(row["group"]) for row in rows} == {clusters.group_name(group.label) for group in clustered.groups}
    for row, recording in zip(rows, clustered.described.recordings, strict=True):
        assert row["sample"] == recording.label
        assert row["pitch"] == recording.key.pitch
        assert row["playing_s"] == pytest.approx(recording.weight)


def test_the_take_standing_for_a_group_is_the_one_its_row_names_a_representative(clustered: Clustered) -> None:
    """Exactly the members the rule picks read as representatives, so the picture and the tables agree."""
    rows = clusters.point_rows(clustered.described, clustered.groups, clustered.distances, rule=clustered.rule)
    marked = {index for index, row in enumerate(rows) if row["role"] == "representative"}

    assert marked == {group.representative(clustered.rule) for group in clustered.groups}


def test_a_representative_stands_no_distance_from_itself(clustered: Clustered) -> None:
    """The distance a row states is read in the space the group was cut in, so its own take reads zero."""
    rows = clusters.point_rows(clustered.described, clustered.groups, clustered.distances, rule=clustered.rule)

    for group in clustered.groups:
        assert rows[group.medoid]["to_medoid"] == pytest.approx(0.0)
        assert rows[group.representative(clustered.rule)]["to_representative"] == pytest.approx(0.0)


def test_a_group_row_spans_the_pitches_and_the_playing_time_its_members_carry(clustered: Clustered) -> None:
    """A group reads as what it gathered, so the sizes and the playing times sum back to the corpus."""
    rows = clusters.group_rows(clustered.described, clustered.groups, rule=clustered.rule)

    assert len(rows) == len(clustered.groups)
    assert sum(int(row["size"]) for row in rows) == clustered.described.size
    assert sum(float(row["playing_s"]) for row in rows) == pytest.approx(clustered.described.weights.sum())
    for row, group in zip(rows, clustered.groups, strict=True):
        pitches = [clustered.described.recordings[member].key.pitch for member in group.members.tolist()]
        assert row["pitch_lo"] == min(pitches)
        assert row["pitch_hi"] == max(pitches)
        assert row["stands_for_it"] == clustered.described.recordings[group.representative(clustered.rule)].label


def test_members_are_listed_from_the_medoid_outwards(clustered: Clustered) -> None:
    """Ordering by that distance puts the take standing for the group first and the one it covers least last."""
    rows = clusters.point_rows(clustered.described, clustered.groups, clustered.distances, rule=clustered.rule)
    for group in clustered.groups:
        listed = clusters.member_rows(rows, group)

        assert len(listed) == group.size
        assert listed[0]["sample"] == rows[group.medoid]["sample"]
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
    rows = clusters.group_block_rows(clustered.space, clustered.groups)

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
    assert rows[0]["sample"] == clustered.described.recordings[0].label
    assert rows[0]["columns"] == clustered.described.descriptors[0].columns


def test_a_take_reads_its_distance_to_every_group_and_owns_exactly_one(clustered: Clustered) -> None:
    """The company a take nearly kept reads beside its own, which is where a cut is finely balanced."""
    rows = clusters.reach_rows(0, clustered.described, clustered.groups, clustered.distances, rule=clustered.rule)

    assert len(rows) == len(clustered.groups)
    assert sum(bool(row["own"]) for row in rows) == 1
    owned = next(row for row in rows if bool(row["own"]))
    assert owned["to_nearest"] == pytest.approx(0.0)


def test_a_representative_rule_names_the_member_the_rows_stand_by(clustered: Clustered) -> None:
    """Naming another rule moves which take a group is stood for by, and every row follows it."""
    weighted = clusters.group_rows(clustered.described, clustered.groups, rule=Representative.WEIGHTED_MEDOID)

    for row, group in zip(weighted, clustered.groups, strict=True):
        assert row["stands_for_it"] == clustered.described.recordings[group.weighted_medoid].label
