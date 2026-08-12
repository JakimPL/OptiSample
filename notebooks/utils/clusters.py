from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

from notebooks.utils.views import Row
from optisample.cluster.corpus import DescribedCorpus
from optisample.cluster.descriptor import SampleDescriptor
from optisample.cluster.partition import Partition
from optisample.cluster.representative import Group
from optisample.cluster.selection import Selection
from optisample.cluster.space import Block, Coordinates, SampleSpace, mean_pairwise_square
from optisample.cluster.stages import StageRecording
from optisample.config.cluster import Representative

_LOG_BASE: Final = 10.0  # what the anchor times are stated in logs of, which reads them back as seconds
_NO_SPREAD: Final = 0.0  # the spread a set of readings standing at one point holds
_REPRESENTATIVE: Final = "representative"  # what the member a group is stood for by is listed as
_MEMBER: Final = "member"  # what every other recording of a group is listed as


def group_name(label: int) -> str:
    """How a group is spelled wherever it is read -- in a table cell, in a legend and in a hover.

    Spelled rather than numbered so a scatter reads the column as a set of names and draws one colour and
    one legend entry per group, and so the same group answers to the same word in every panel.
    """
    return f"g{label:02d}"


def _group_of(groups: Sequence[Group], samples: int) -> tuple[int, ...]:
    """Which group each recording of the space fell into, as an index into ``groups``."""
    placed = [0] * samples
    for index, group in enumerate(groups):
        for member in group.members.tolist():
            placed[member] = index

    return tuple(placed)


def point_rows(
    described: DescribedCorpus,
    groups: Sequence[Group],
    distances: Coordinates,
    *,
    rule: Representative,
) -> list[Row]:
    """One row per recording: where it sits, which group it fell into, and how far it stands from that group.

    These are the rows the scatter hovers and the member tables list, so a point picked off the picture and
    a line read out of a table say the same things about the same take.
    """
    placed = _group_of(groups, described.size)
    return [_point_row(described, index, groups[placed[index]], reach, rule) for index, reach in enumerate(distances)]


def _point_row(
    described: DescribedCorpus,
    index: int,
    group: Group,
    reach: Coordinates,
    rule: Representative,
) -> Row:
    """One recording as the line every panel reads it through."""
    recording = described.recordings[index]
    descriptor = described.descriptors[index]
    representative = group.representative(rule)
    return {
        "sample": recording.label,
        "note": recording.note,
        "pitch": recording.key.pitch,
        "velocity": recording.key.velocity,
        "group": group_name(group.label),
        "role": _REPRESENTATIVE if index == representative else _MEMBER,
        "to_medoid": float(reach[group.medoid]),
        "to_representative": float(reach[representative]),
        "dur_s": recording.duration_s,
        "playing_s": recording.weight,
        "rate": recording.sample_rate,
        "depths": int(descriptor.reached.sum()),
    }


def group_rows(described: DescribedCorpus, groups: Sequence[Group], *, rule: Representative) -> list[Row]:
    """One row per group: how big it is, what it spans, and which takes stand for it.

    ``spread_mean`` and ``spread_max`` say how far the group reaches from its medoid, so a representative
    is read beside the amount of sound it is being asked to cover.
    """
    recordings = described.recordings
    return [
        {
            "group": group_name(group.label),
            "size": group.size,
            "pitch_lo": min(recordings[member].key.pitch for member in group.members.tolist()),
            "pitch_hi": max(recordings[member].key.pitch for member in group.members.tolist()),
            "vel_lo": min(recordings[member].key.velocity for member in group.members.tolist()),
            "vel_hi": max(recordings[member].key.velocity for member in group.members.tolist()),
            "playing_s": sum(recordings[member].weight for member in group.members.tolist()),
            "spread_mean": group.spread_mean,
            "spread_max": group.spread_max,
            "stands_for_it": recordings[group.representative(rule)].label,
            "medoid": recordings[group.medoid].label,
            "weighted_medoid": recordings[group.weighted_medoid].label,
            "nearest_centroid": recordings[group.nearest_centroid].label,
            "farthest": recordings[group.farthest].label,
        }
        for group in groups
    ]


def pick_rows(chosen: Selection) -> list[Row]:
    """One row per chosen take: what it plays, and how much of the corpus it was chosen to stand for.

    ``members`` and ``playing_s`` are what the group behind a pick holds, so a written selection is read
    beside the share of the material each of its recordings answers for.
    """
    return [
        {
            "group": group_name(pick.group),
            "sample": pick.recording.label,
            "note": pick.recording.note,
            "pitch": pick.recording.key.pitch,
            "velocity": pick.recording.key.velocity,
            "members": pick.members,
            "playing_s": pick.playing_s,
            "dur_s": pick.recording.duration_s,
        }
        for pick in chosen.picks
    ]


def member_rows(rows: Sequence[Row], group: Group) -> list[Row]:
    """The recordings of one group, the closest to its medoid first.

    Ordering by that distance puts the take standing for the group at the top and the one it covers least
    at the bottom, which is where a group about to be split shows itself.
    """
    members = [rows[member] for member in group.members.tolist()]
    return sorted(members, key=lambda row: float(row["to_medoid"]))


def block_rows(space: SampleSpace) -> list[Row]:
    """One row per block: how many columns it laid down, the say it was given, and the spread it holds.

    ``share`` is how much of the whole space's spread that block accounts for as it stands, weight and all,
    which says which reading the groups on screen were genuinely drawn by.
    """
    spreads = {block: mean_pairwise_square(space.block(block)) for block in Block}
    whole = sum(spreads.values())
    return [
        {
            "block": block.value,
            "weight": space.spans[block].weight,
            "columns": space.spans[block].columns,
            "spread": spreads[block],
            "share": spreads[block] / whole if whole > _NO_SPREAD else _NO_SPREAD,
        }
        for block in Block
    ]


def group_block_rows(space: SampleSpace, groups: Sequence[Group]) -> list[Row]:
    """How tightly each group holds together in each block, one row per group and block.

    A group whose members sit close in the onset block and spread out in the envelope block is a set of
    takes sharing an attack while their contours differ, which is what says where the grouping came from.
    """
    return [
        {
            "group": group_name(group.label),
            "block": block.value,
            "spread": _block_spread(space.block(block), group),
        }
        for group in groups
        for block in Block
    ]


def _block_spread(columns: Coordinates, group: Group) -> float:
    """The mean distance from a group's members to their middle, read inside one block alone."""
    held = columns[group.members]
    return float(np.linalg.norm(held - held.mean(axis=0), axis=1).mean())


def sweep_rows(partitions: Sequence[Partition], *, chosen: int) -> list[Row]:
    """What every group count the sweep climbed is worth, with the count now on screen marked.

    A silhouette runs from -1 to 1 and says how much closer a recording sits to its own group than to the
    nearest other one, so the count to read a corpus at is the one topping this column.
    """
    return [
        {
            "groups": found.groups,
            "silhouette": found.silhouette,
            "on_screen": found.groups == chosen,
        }
        for found in partitions
    ]


def reach_rows(
    place: int,
    described: DescribedCorpus,
    groups: Sequence[Group],
    distances: Coordinates,
    *,
    rule: Representative,
) -> list[Row]:
    """How far one recording stands from every group, so the company it nearly kept reads beside its own.

    A take sitting almost as close to a neighbouring group as to its own is where a cut is finely balanced,
    which is what says whether one more group would tell this corpus apart better.
    """
    reach = distances[place]
    return [
        {
            "group": group_name(group.label),
            "own": place in group.members.tolist(),
            "to_representative": float(reach[group.representative(rule)]),
            "to_nearest": float(min(reach[member] for member in group.members.tolist())),
            "stands_for_it": described.recordings[group.representative(rule)].label,
        }
        for group in groups
    ]


def descriptor_rows(descriptor: SampleDescriptor, recording: StageRecording) -> list[Row]:
    """One recording's own readings, as the single line the examine panel states it by."""
    envelope = descriptor.envelope
    return [
        {
            "sample": recording.label,
            "columns": descriptor.columns,
            "depths_reached": int(descriptor.reached.sum()),
            "attack_s": envelope.attack_s,
            "settle_s": envelope.settle_s,
            "decay_db_per_s": envelope.decay_db_per_s,
            "curvature_db": envelope.curvature_db,
            "change_db_per_s": descriptor.movement.change_db_per_s,
            "travel_db_per_s": descriptor.movement.travel_db_per_s,
        }
    ]


def anchor_rows(descriptor: SampleDescriptor, depths_db: Sequence[float], space: SampleSpace) -> list[Row]:
    """One row per fall depth: when this recording got there, and whether the corpus reads that depth.

    ``reached`` says the recording itself arrived at the depth, and ``in_space`` that enough of the corpus
    did for the depth to stand as a column, so a take short of a depth other takes share reads as the row
    holding what it last had.
    """
    return [
        {
            "depth_db": depth,
            "reached": bool(descriptor.reached[index]),
            "in_space": bool(space.depths[index]),
            "time_s": float(_LOG_BASE ** descriptor.envelope.anchor_log_s[index]),
        }
        for index, depth in enumerate(depths_db)
    ]
