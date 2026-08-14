from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from typing import Final

import numpy as np

from notebooks.utils.views import Row
from optisample.cluster.corpus import DescribedCorpus
from optisample.cluster.descriptor import SampleDescriptor
from optisample.cluster.partition import Partition
from optisample.cluster.representative import Group
from optisample.cluster.selection import Selection
from optisample.cluster.space import (
    Block,
    Coordinates,
    SampleSpace,
    mean_pairwise_square,
)
from optisample.cluster.stages import RecordingSource, StageRecording
from optisample.keys import SampleKey
from optisample.music import MIDI_HIGHEST_PITCH, MIDI_MAX_VELOCITY

_LOG_BASE: Final = 10.0  # what the anchor times are stated in logs of, which reads them back as seconds
_NO_SPREAD: Final = 0.0  # the spread a set of readings standing at one point holds
_REPRESENTATIVE: Final = "representative"  # what the member a group is stood for by is listed as
_MEMBER: Final = "member"  # what every other recording of a group is listed as
_HUE_SPAN: Final = 1.0  # of the colour wheel the keyboard is laid across, which holds its two ends apart
_QUIETEST: Final = 0.25  # the saturation the softest take is drawn at, so a quiet group is still a colour
_LOUDEST: Final = 1.0  # the saturation the hardest-struck take fills, the velocity spanning what lies between
_BRIGHTNESS: Final = 0.9  # the value a group is drawn at, one step under white so a light page holds it
_CHANNEL: Final = 255  # what one colour channel is spelled out of in a hex triplet
_ONE_HOLDER: Final = 1  # groups a key stands for while its name still needs no number


def group_name(key: SampleKey) -> str:
    """How a group is spelled wherever it is read -- in a table cell, in a legend and in a dropdown.

    A group goes by the key the take standing for it plays, so the word every panel names it by is the
    note and the velocity a reader is about to hear. The pitch leads, zero-padded, so a sorted list of
    groups reads in keyboard order.
    """
    return f"p{key.pitch:03d}_v{key.velocity:03d}"


def sample_name(recording: StageRecording) -> str:
    """How a table names one recording across sets: the set it came from, then the file holding it.

    A file stem is unique inside its own set alone, so two sets of one run can each hold a take spelled the
    same way. Leading with the set is what keeps them apart wherever a row points at a recording it is not
    itself about -- the take standing for a group, the one a reach row measures to -- and reads that take
    back to the dataset it is played and written out of.
    """
    return f"{recording.source.label} · {recording.label}"


def sources_label(sources: Sequence[RecordingSource]) -> str:
    """How a title names the sets one space was gathered from, in the order their recordings stand.

    A picture, a spinner and a summary line all state which sets they answer for, so a space holding two
    of them says so wherever it is read.
    """
    return " + ".join(source.label for source in sources)


def group_colour(key: SampleKey) -> str:
    """The colour a group is drawn in, turned and filled by the key the take standing for it plays.

    Pitch turns the hue and velocity fills the saturation, so a picture carries the keyboard in its
    colours: where a group sits on the keys and how hard its takes were struck are both read off the field
    at a glance, and one key is drawn the same colour in every picture, at every stage and across datasets.
    The saturation opens a quarter of the way up, which leaves the softest takes a colour of their own to
    be told apart by, and the hue is laid across most of the wheel, which keeps the top of the keyboard
    clear of the bottom.
    """
    hue = _HUE_SPAN * key.pitch / MIDI_HIGHEST_PITCH
    saturation = _QUIETEST + (_LOUDEST - _QUIETEST) * key.velocity / MIDI_MAX_VELOCITY
    channels = hsv_to_rgb(hue, saturation, _BRIGHTNESS)
    return "#" + "".join(f"{round(channel * _CHANNEL):02x}" for channel in channels)


@dataclass(frozen=True)
class NamedGroup:
    """One cut group as the panels read it: the group itself, the word it goes by and the colour it holds.

    ``name`` and ``colour`` are both read off the take standing for the group, so a legend entry, a table
    cell, a dropdown option and a ring over the field all name one group by the key a reader is about to
    hear, and moving the cut moves every one of them together.
    """

    group: Group
    name: str
    colour: str


def named_groups(described: DescribedCorpus, groups: Sequence[Group]) -> tuple[NamedGroup, ...]:
    """Every group beside the word it goes by and the colour it is drawn in, in the order they were cut.

    Two groups can be stood for by takes of one key -- a corpus holding several renditions of a note
    places them apart, and a cut can hand each its own group -- so the second holder of a name onward
    carries the count of how many have held it. That keeps one word per group wherever a panel lists them,
    which is what a dropdown and a legend read a group back by.
    """
    named: list[NamedGroup] = []
    holders: Counter[str] = Counter()
    for group in groups:
        key = described.recordings[group.representative].key
        spelling = group_name(key)
        holders[spelling] += 1
        held = holders[spelling]
        named.append(
            NamedGroup(
                group=group,
                name=spelling if held <= _ONE_HOLDER else f"{spelling}#{held}",
                colour=group_colour(key),
            )
        )

    return tuple(named)


def group_colours(groups: Sequence[NamedGroup]) -> dict[str, str]:
    """The colour each group is drawn in, under the word it goes by, as one picture reads them all."""
    return {named.name: named.colour for named in groups}


def _group_of(groups: Sequence[NamedGroup], samples: int) -> tuple[int, ...]:
    """Which group each recording of the space fell into, as an index into ``groups``."""
    placed = [0] * samples
    for index, named in enumerate(groups):
        for member in named.group.members.tolist():
            placed[member] = index

    return tuple(placed)


def point_rows(described: DescribedCorpus, groups: Sequence[NamedGroup], distances: Coordinates) -> list[Row]:
    """One row per recording: where it sits, which group it fell into, and how far it stands from that group.

    These are the rows the scatter hovers and the member tables list, so a point picked off the picture and
    a line read out of a table say the same things about the same take. ``set`` names the source the take
    came from, which is what colours a gathered space by the sets it holds and tells two of them apart on a
    field they now share.
    """
    placed = _group_of(groups, described.size)
    return [_point_row(described, index, groups[placed[index]], reach) for index, reach in enumerate(distances)]


def _point_row(described: DescribedCorpus, index: int, named: NamedGroup, reach: Coordinates) -> Row:
    """One recording as the line every panel reads it through."""
    recording = described.recordings[index]
    descriptor = described.descriptors[index]
    group = named.group
    representative = group.representative
    return {
        "sample": recording.label,
        "set": recording.source.label,
        "note": recording.note,
        "pitch": recording.key.pitch,
        "velocity": recording.key.velocity,
        "group": named.name,
        "role": _REPRESENTATIVE if index == representative else _MEMBER,
        "to_medoid": float(reach[group.medoid]),
        "to_representative": float(reach[representative]),
        "dur_s": recording.duration_s,
        "playing_s": recording.weight,
        "rate": recording.sample_rate,
        "depths": int(descriptor.reached.sum()),
    }


def group_rows(described: DescribedCorpus, groups: Sequence[NamedGroup]) -> list[Row]:
    """One row per group: how big it is, what it spans, and which takes stand for it.

    ``spread_mean`` and ``spread_max`` say how far the group reaches from its medoid, so a representative
    is read beside the amount of sound it is being asked to cover. ``sets`` counts the sources its members
    were gathered from, which is what says a sound several sets hold in common.
    """
    return [_group_row(described.recordings, named) for named in groups]


def _group_row(recordings: Sequence[StageRecording], named: NamedGroup) -> Row:
    """One group as the line the tables read it through, its members gathered once for every reading."""
    group = named.group
    members = [recordings[member] for member in group.members.tolist()]
    return {
        "group": named.name,
        "size": group.size,
        "sets": len({member.source for member in members}),
        "pitch_lo": min(member.key.pitch for member in members),
        "pitch_hi": max(member.key.pitch for member in members),
        "vel_lo": min(member.key.velocity for member in members),
        "vel_hi": max(member.key.velocity for member in members),
        "playing_s": sum(member.weight for member in members),
        "spread_mean": group.spread_mean,
        "spread_max": group.spread_max,
        "stands_for_it": sample_name(recordings[group.representative]),
        "medoid": sample_name(recordings[group.medoid]),
        "weighted_medoid": sample_name(recordings[group.weighted_medoid]),
        "nearest_centroid": sample_name(recordings[group.nearest_centroid]),
        "farthest": sample_name(recordings[group.farthest]),
    }


def pick_rows(chosen: Selection, groups: Sequence[NamedGroup]) -> list[Row]:
    """One row per chosen take: what it plays, and how much of the corpus it was chosen to stand for.

    ``members`` and ``playing_s`` are what the group behind a pick holds, so a written selection is read
    beside the share of the material each of its recordings answers for. A selection is chosen group by
    group in the order they were cut, so every pick reads under the word its own group goes by. ``set``
    names the source the take is cut of, which is the dataset writing the selection lands it in.

    Raises:
        ValueError: when ``chosen`` and ``groups`` state different counts of groups.
    """
    return [
        {
            "group": named.name,
            "sample": pick.recording.label,
            "set": pick.source.label,
            "note": pick.recording.note,
            "pitch": pick.recording.key.pitch,
            "velocity": pick.recording.key.velocity,
            "members": pick.members,
            "playing_s": pick.playing_s,
            "dur_s": pick.recording.duration_s,
        }
        for pick, named in zip(chosen.picks, groups, strict=True)
    ]


def member_rows(rows: Sequence[Row], named: NamedGroup) -> list[Row]:
    """The recordings of one group, the closest to its medoid first.

    Ordering by that distance puts the take standing for the group at the top and the one it covers least
    at the bottom, which is where a group about to be split shows itself.
    """
    members = [rows[member] for member in named.group.members.tolist()]
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


def group_block_rows(space: SampleSpace, groups: Sequence[NamedGroup]) -> list[Row]:
    """How tightly each group holds together in each block, one row per group and block.

    A group whose members sit close in the onset block and spread out in the envelope block is a set of
    takes sharing an attack while their contours differ, which is what says where the grouping came from.
    """
    return [
        {
            "group": named.name,
            "block": block.value,
            "spread": _block_spread(space.block(block), named.group),
        }
        for named in groups
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
    groups: Sequence[NamedGroup],
    distances: Coordinates,
) -> list[Row]:
    """How far one recording stands from every group, so the company it nearly kept reads beside its own.

    A take sitting almost as close to a neighbouring group as to its own is where a cut is finely balanced,
    which is what says whether one more group would tell this corpus apart better.
    """
    reach = distances[place]
    return [
        {
            "group": named.name,
            "own": place in named.group.members.tolist(),
            "to_representative": float(reach[named.group.representative]),
            "to_nearest": float(min(reach[member] for member in named.group.members.tolist())),
            "stands_for_it": sample_name(described.recordings[named.group.representative]),
        }
        for named in groups
    ]


def descriptor_rows(descriptor: SampleDescriptor, recording: StageRecording) -> list[Row]:
    """One recording's own readings, as the single line the examine panel states it by."""
    envelope = descriptor.envelope
    return [
        {
            "sample": sample_name(recording),
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
