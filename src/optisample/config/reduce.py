from enum import StrEnum, unique
from typing import Annotated, Final, Self

from pydantic import Field, model_validator

from optisample.config.base import ConfigModel
from optisample.config.stage import StageConfig

NO_GROUPING: Final = 0  # the ``max_zone_semitones`` width that leaves every key standing on its own


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


class TrimConfig(ConfigModel):
    """How much of a recording is worth keeping: how long it may run, and where its content ends.

    ``max_length_s`` bounds every kept recording, so one running far past the material it serves is cut to
    a span the allocation can afford to store. ``tail_floor`` is the amplitude a decay must still reach for
    the recording to count as carrying it, which is where the kept span ends. ``silence_floor`` is the peak
    a recording has to reach somewhere to enter the dataset at all, so a take that never sounded is left
    out; it sits at or above ``tail_floor``, which leaves every kept recording holding content.
    """

    max_length_s: Annotated[float, Field(gt=0.0)]
    tail_floor: Annotated[float, Field(gt=0.0)]
    silence_floor: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def _floors_ordered(self) -> Self:
        """Hold the two floors in the order that leaves a kept recording with content to trim to.

        Raises:
            ValueError: when ``silence_floor`` sits under ``tail_floor``, which would admit a recording
                whose every frame the tail trim then cuts.
        """
        if self.silence_floor < self.tail_floor:
            raise ValueError(f"silence_floor {self.silence_floor} must be at least tail_floor {self.tail_floor}")

        return self


class EventsConfig(ConfigModel):
    """How the material's note events collapse before they are scored.

    ``duration_bucket_ratio`` sets geometric bucket edges that each note's held duration rounds up to,
    so one scored event stands for every note of a similar length. A ratio of 1.0 keeps every distinct
    duration, which is the exact-per-note behaviour.
    """

    duration_bucket_ratio: Annotated[float, Field(ge=1.0)]


class BandwidthConfig(ConfigModel):
    """How the rate a sample is stored at follows from the recording's own band, and what a narrow one costs.

    ``ceiling_hz`` is the highest output frequency worth carrying, which bounds the stored bandwidth once
    playback transposition is applied. ``content_floor_db`` and ``content_band_hz`` measure the band a
    recording itself occupies: how far under its loudest band content still counts, read off a spectrum
    averaged into bands that wide. Together they name the rate a clip asks to be stored at
    (:func:`~optisample.optimize.reduce.bandwidth.useful_rate_hz`), and the ladder's lowest rung reaching
    it is what the sample is kept at, so ``content_floor_db`` decides how much band the run stores.

    ``discard_floor_db`` reads the same spectrum at a deeper floor, naming the band a recording still
    reaches rather than the band the allocation is obliged to carry
    (:func:`~optisample.optimize.reduce.bandwidth.clip_reach_hz`). The gap between the two floors is the
    content a settled rung leaves out, and ``discard_penalty`` is the distortion charged for each octave
    of it (:func:`~optisample.optimize.reduce.bandwidth.discard_surcharge`). The composite metrics score
    a narrowed sample as close to its reference, so the penalty is where a run states what that
    narrowing is worth by ear; the allocation then buys a wider rung wherever the charge exceeds what
    the same bytes buy elsewhere, and pays for it by storing fewer and wider samples. A penalty of 0.0
    prices a rung on the metrics alone.
    """

    ceiling_hz: Annotated[float, Field(gt=0.0)]
    content_floor_db: Annotated[float, Field(gt=0.0)]
    content_band_hz: Annotated[float, Field(gt=0.0)]
    discard_floor_db: Annotated[float, Field(gt=0.0)]
    discard_penalty: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def _floors_ordered(self) -> Self:
        """Hold the deeper floor at or under the one that settles the rung, which is what leaves a gap to price.

        Raises:
            ValueError: when ``discard_floor_db`` sits above ``content_floor_db``, which would charge a
                sample for band the settled rung already carries.
        """
        if self.discard_floor_db < self.content_floor_db:
            raise ValueError(
                f"discard_floor_db {self.discard_floor_db} must be at least content_floor_db {self.content_floor_db}"
            )

        return self


class ZoneConfig(ConfigModel):
    """How pitch-zone grouping bounds its own search.

    ``max_zone_semitones`` caps how wide a contiguous zone may be, which is what keeps the number of
    candidate ranges linear in the keyboard span. :data:`NO_GROUPING` keeps every key its own sample,
    which is what a keyboard of unrelated sounds asks for: a percussion map numbers a different
    instrument at each key, so a zone spanning several of them stores one of those sounds in place of
    the rest.
    """

    max_zone_semitones: Annotated[int, Field(ge=NO_GROUPING)]


class ReduceConfig(StageConfig):
    """Every pre-optimization reduction: what survives ingest, and how small the search space starts."""

    dedupe: DedupeConfig
    trim: TrimConfig
    events: EventsConfig
    bandwidth: BandwidthConfig
    grouping: ZoneConfig
