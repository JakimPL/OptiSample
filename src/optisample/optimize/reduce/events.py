from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import floor, log
from typing import Final

from optisample.config.reduce import EventsConfig
from optisample.keys import SampleKey, nearest_key
from optisample.model import NoteEvent
from optisample.optimize.velocity_map import VelocityVolumeMap

EventIdentity = tuple[SampleKey, int, float]

_GRID_ANCHOR_S: Final = 1.0  # the grid's fixed point; one second sits on an edge at every ratio
_EXACT: Final = 1.0  # a ratio of one puts an edge at every duration, so each length is scored as played


@dataclass(frozen=True)
class MergedEvent:
    """A class of notes at one pitch that reconstruct identically, with their playing time added up.

    ``reference_key`` is the recording every note in the class is compared against, ``volume`` the note
    volume they all render at, and ``duration_s`` the length they are all scored over. ``velocity``
    labels the class by the loudest note in it, which the artifacts report as the dynamic it stands for.
    """

    reference_key: SampleKey
    velocity: int
    volume: int
    duration_s: float
    weight: float

    @property
    def identity(self) -> EventIdentity:
        """What decides two notes reconstruct identically: reference, volume, and scored length.

        The one home for that rule, so the classes merging collapses the material into are the very
        classes scoring reads back and remembers a reconstruction under.
        """
        return self.reference_key, self.volume, self.duration_s

    def absorbing(self, other: MergedEvent) -> MergedEvent:
        """This class with ``other`` folded in: their playing time summed, under the louder label."""
        return replace(self, velocity=max(self.velocity, other.velocity), weight=self.weight + other.weight)


def bucket_duration_s(duration_s: float, ratio: float) -> float:
    """The geometric grid edge at or above ``duration_s``, which is the length a note is scored over.

    Edges are powers of ``ratio`` around one second, so every note within a fixed proportion of a length
    shares one edge and the whole span is scored once. Rounding up keeps the scored note at least as
    long as the note played, so a sample trimmed to it still covers the note in full. A ratio of one
    leaves every duration on its own edge, scoring each length exactly as played.
    """
    if ratio <= _EXACT:
        return duration_s

    below = _GRID_ANCHOR_S * ratio ** floor(log(duration_s / _GRID_ANCHOR_S, ratio))
    return below if below >= duration_s else below * ratio


def _classify(
    event: NoteEvent,
    available: Sequence[SampleKey],
    velocity_map: VelocityVolumeMap,
    ratio: float,
) -> MergedEvent:
    """The one-note class ``event`` opens: what it is compared against, rendered at, and scored over."""
    return MergedEvent(
        reference_key=nearest_key(available, event.velocity),
        velocity=event.velocity,
        volume=velocity_map.volume(event.velocity),
        duration_s=bucket_duration_s(event.duration_s, ratio),
        weight=event.weight,
    )


def merge_events(
    events: Sequence[NoteEvent],
    available: Sequence[SampleKey],
    velocity_map: VelocityVolumeMap,
    config: EventsConfig,
) -> list[MergedEvent]:
    """Collapse the notes at one pitch into the classes that reconstruct identically, weights summed.

    Scoring reads a note's velocity twice, and both readings are many-to-one:
    :func:`~optisample.keys.nearest_key` picks the recording it is compared against, and
    the velocity map picks the volume it renders at. Notes agreeing on both, over the same scored length,
    therefore earn the same fidelity report, so one scored class stands for all of them and carries their
    combined playing time as its weight -- an exact reduction, leaving the weighted mean it feeds
    untouched. :func:`bucket_duration_s` widens each class further, to a proportion of length.

    Classes come back in the order the material first plays them.
    """
    classes: dict[EventIdentity, MergedEvent] = {}
    for event in events:
        opened = _classify(event, available, velocity_map, config.duration_bucket_ratio)
        found = classes.get(opened.identity)
        classes[opened.identity] = found.absorbing(opened) if found is not None else opened

    return list(classes.values())
