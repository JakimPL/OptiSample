from enum import StrEnum, unique
from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel


@unique
class DedupeKey(StrEnum):
    """Which identity fields decide that two recordings are renditions of the same thing.

    A coarser key collapses more recordings into one survivor, shrinking what the optimizer explores;
    a finer key keeps timbral variants apart so they can compete as representatives. The three members
    are the only meaningful shapes, so an invalid combination is unrepresentable.
    """

    PITCH = "pitch"
    PITCH_VELOCITY = "pitch_velocity"
    PITCH_VELOCITY_CC = "pitch_velocity_cc"

    @property
    def includes_velocity(self) -> bool:
        return self is not DedupeKey.PITCH

    @property
    def includes_cc(self) -> bool:
        return self is DedupeKey.PITCH_VELOCITY_CC


@unique
class Representatives(StrEnum):
    """Which surviving recordings at a pitch may become the sample the plan stores for it."""

    NEAREST_LOUDEST = "nearest_loudest"
    ALL = "all"


class DedupeConfig(ConfigModel):
    """How the recorded grid collapses to one representative recording per key.

    ``transposition_headroom_semitones`` is the upward transpose a survivor must still cover in full, so
    a recording kept here remains long enough when pitch-zone grouping routes a higher key to it.
    ``cc_quantum`` is the bucket width that turns continuous controller averages into an identity.
    """

    key: DedupeKey
    cc_quantum: Annotated[float, Field(gt=0.0)]
    transposition_headroom_semitones: Annotated[int, Field(ge=0)]
    representatives: Representatives


class EventsConfig(ConfigModel):
    """How the material's note events collapse before they are scored.

    ``duration_bucket_ratio`` sets geometric bucket edges that each note's held duration rounds up to,
    so one scored event stands for every note of a similar length. A ratio of 1.0 keeps every distinct
    duration, which is the exact-per-note behaviour.
    """

    duration_bucket_ratio: Annotated[float, Field(ge=1.0)]


class BandwidthConfig(ConfigModel):
    """How the stored rate/depth shortlist is derived before the expensive sweep runs.

    ``candidates`` is how many rate-distortion vertices survive per clip; a value at or above the full
    grid size keeps every encoding the sweep would have tried. ``ceiling_hz`` is the highest output
    frequency worth carrying, which bounds the stored bandwidth once playback transposition is applied.
    """

    candidates: Annotated[int, Field(ge=1)]
    ceiling_hz: Annotated[float, Field(gt=0.0)]


class ZoneConfig(ConfigModel):
    """How pitch-zone grouping bounds its own search.

    ``max_zone_semitones`` caps how wide a contiguous zone may be, which is what keeps the number of
    candidate ranges linear in the keyboard span. ``memoize`` reuses a scored
    ``(representative, encoding, key)`` reconstruction across every zone that contains it, and seeds
    each encode's dither from that same identity so the result is independent of evaluation order.
    """

    max_zone_semitones: Annotated[int, Field(ge=1)]
    memoize: bool


class ReduceConfig(ConfigModel):
    """Every pre-optimization reduction: what survives ingest, and how small the search space starts."""

    dedupe: DedupeConfig
    events: EventsConfig
    bandwidth: BandwidthConfig
    grouping: ZoneConfig
