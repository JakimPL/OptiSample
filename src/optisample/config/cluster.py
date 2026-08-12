from __future__ import annotations

from enum import StrEnum, unique
from typing import Annotated, Self

from pydantic import Field, model_validator

from optisample.config.base import ConfigModel
from optisample.config.stage import StageConfig


@unique
class FrequencyBasis(StrEnum):
    """Which frequency axis a recording's timbre is read on, which is what a group of them shares.

    ``RELATIVE`` reads the level each partial of the note's own pitch holds, so a reading states the
    balance a struck string sounds with and two notes an octave apart sharing that balance land together.
    ``ABSOLUTE`` reads the mel bands the composite fidelity scores with, so a group here is a set of
    recordings that compare alike under the metric the optimizer allocates by.
    """

    RELATIVE = "relative"
    ABSOLUTE = "absolute"


@unique
class LinkageMethod(StrEnum):
    """How far apart two groups stand while a hierarchy is built, which is the shape the groups take.

    ``WARD`` joins the pair adding the least spread, which favours groups of a similar size and a round
    shape; ``AVERAGE`` reads the mean distance across a pair and ``COMPLETE`` its widest, which lets a
    long-drawn group hold together and holds a group's diameter down respectively.
    """

    WARD = "ward"
    AVERAGE = "average"
    COMPLETE = "complete"


@unique
class PartitionAlgorithm(StrEnum):
    """Which rule cuts the space into groups: a hierarchy read at one height, or a run of moving centres."""

    HIERARCHICAL = "hierarchical"
    KMEANS = "kmeans"


@unique
class Representative(StrEnum):
    """Which member of a group stands for it, once the group is settled.

    ``MEDOID`` is the recording standing closest to the rest of its group, which is the dominant
    representative: a real take a listener can play. ``WEIGHTED_MEDOID`` reads the same sum with each
    member weighted by the playing time the material gives its key, so the representative is the one
    carrying the music. ``NEAREST_CENTROID`` is the member closest to the group's mean coordinate.
    """

    MEDOID = "medoid"
    WEIGHTED_MEDOID = "weighted_medoid"
    NEAREST_CENTROID = "nearest_centroid"


class DescriptorConfig(ConfigModel):
    """How one recording is read into the blocks a sample space is built from.

    ``frequency_basis`` picks the axis a frame's shape is read on, and the reading is level-free on either:
    the harmonics of a note and its mel bands are both stated past that frame's own mean, so a take at v020
    and the same note at v100 differ by what they sound like alone.

    ``anchor_depths_db`` is where along a note's own decline it is read -- each entry a fall below that
    recording's own peak, so anchor *k* is the moment the note had fallen *k* decibels. Reading a note on
    its own decline is what leaves its length out of the reading: a two-second take and an eight-second
    take of one note reach the same depths and are read at each of them. Depths ascend, which lets a
    recording falling short of one hold what it last reached. ``anchor_span_s`` is the stretch of material
    each anchor averages over, so a reading covers a stretch of the note rather than one frame of it.

    ``harmonics`` is how many partials of the note's own pitch the ``RELATIVE`` basis reads, and
    ``harmonic_band_share`` how far either side of each multiple that partial is looked for, as a share of
    the pitch: a share of 0.5 tiles the spectrum into harmonic-wide bands, so each reads the loudest
    partial near its own multiple and follows the stretch a struck string's partials are spaced by.
    ``harmonic_range_db`` is the range one frame states its harmonics across, under that frame's own
    loudest, which has a partial late in a decay read as fully as the attack was.

    ``cepstral_coefficients`` is how many coefficients of the discrete cosine transform a pooled reading
    keeps past the constant one, which holds the smooth spectral envelope; keeping none reads the bands
    as they stand. ``envelope_nodes`` is how many corners the level curve is fitted with, which is the
    curve a note's departure from a straight decline is measured against.
    """

    frequency_basis: FrequencyBasis
    anchor_depths_db: tuple[Annotated[float, Field(ge=0.0)], ...]
    anchor_span_s: Annotated[float, Field(gt=0.0)]
    harmonics: Annotated[int, Field(ge=1)]
    harmonic_band_share: Annotated[float, Field(gt=0.0, le=0.5)]
    harmonic_range_db: Annotated[float, Field(gt=0.0)]
    cepstral_coefficients: Annotated[int, Field(ge=0)]
    envelope_nodes: Annotated[int, Field(ge=2)]

    @model_validator(mode="after")
    def _depths_ascend(self) -> Self:
        """Hold the fall depths in the order that lets a recording short of one hold what it last reached.

        Raises:
            ValueError: when the depths name none, or when one sits at or under the depth before it.
        """
        if not self.anchor_depths_db:
            raise ValueError("a recording is read at one fall depth at the least")

        if any(later <= earlier for earlier, later in zip(self.anchor_depths_db, self.anchor_depths_db[1:])):
            raise ValueError(f"fall depths ascend, against {self.anchor_depths_db}")

        return self


class SpaceConfig(ConfigModel):
    """How the blocks are scaled and weighed into the one space every group and representative is read in.

    Each block is stated in units of its own -- decibels of spectral shape, decibels a second of decline,
    log-seconds of timing -- so each is z-scored and then scaled to a mean pairwise squared distance of
    one. That leaves the weights below dimensionless and comparable across datasets: a weight of 1.0 gives
    a block the same say as any other at 1.0, whatever it measures.

    ``onset_weight`` covers the attack timbre, which for a struck string is the most telling stretch there
    is; ``sustain_weight`` covers what the note sounds like at each depth of its own decline; and
    ``movement_weight`` covers how far its timbre travels while it rings. ``envelope_weight`` covers the
    contour -- how long the note took to reach each depth and how straight its decline runs -- which is a
    supplementary distinction: 0.0 leaves the space reading timbre alone and 1.0 gives the contour equal
    footing with a timbre block.

    ``min_reached_share`` is the share of the corpus that reaches a fall depth for that depth to be read as
    a column at all, so the matrix is built on the part of a decline the recordings genuinely share.
    """

    onset_weight: Annotated[float, Field(ge=0.0)]
    sustain_weight: Annotated[float, Field(ge=0.0)]
    movement_weight: Annotated[float, Field(ge=0.0)]
    envelope_weight: Annotated[float, Field(ge=0.0)]
    min_reached_share: Annotated[float, Field(gt=0.0, le=1.0)]


class PartitionConfig(ConfigModel):
    """How the space is cut into groups, and which member of each stands for it.

    ``algorithm`` picks the rule and ``linkage`` shapes the hierarchy the ``HIERARCHICAL`` rule reads.
    ``groups`` is how many the space is cut into, and ``max_groups`` how far a sweep climbs while it reports
    what each count is worth, so a reader picks the count off a curve. ``representative`` names the member
    a group is stood for by.

    ``min_duration_s`` is how long the take standing for a group rings for at the least. A representative is
    what a later stage loops, shapes and allocates storage to, so it has to hold enough sound for a listener
    to control; where the member the rule names falls short of that length, the group's own nearest member
    holding it stands instead. A floor of zero leaves the rule's own choice standing.
    """

    algorithm: PartitionAlgorithm
    linkage: LinkageMethod
    groups: Annotated[int, Field(ge=2)]
    max_groups: Annotated[int, Field(ge=2)]
    representative: Representative
    min_duration_s: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def _sweep_reaches_the_cut(self) -> Self:
        """Hold the sweep's ceiling at or above the count the space is cut at, so the curve covers the cut.

        Raises:
            ValueError: when ``max_groups`` sits under ``groups``.
        """
        if self.max_groups < self.groups:
            raise ValueError(f"max_groups {self.max_groups} is at least groups {self.groups}")

        return self


class InstrumentConfig(ConfigModel):
    """How a cut sample space is written as the instruments a tracker loads.

    ``layers`` is how many velocity bands the corpus is cut into before any of it is grouped. A tracker
    keymap names a key and nothing else, so a dynamic is told apart by the instrument it plays through and
    one band is written as one instrument -- which is why the axis is split first and the space is cut
    inside each band.

    ``rate`` and ``depth`` state what a stored carrier keeps. A waveform whose level has moved to the
    envelope is level-flat, so it spends the whole of a shallow grid on timbre, which is what makes a depth
    of eight worth asking for and halves what the same set of recordings costs.
    """

    layers: Annotated[int, Field(ge=1)]
    rate: Annotated[int, Field(gt=0)]
    depth: Annotated[int, Field(gt=0)]


class ClusterConfig(StageConfig):
    """How a set of recordings becomes a space of points, the groups it falls into, and what they are written as."""

    descriptor: DescriptorConfig
    space: SpaceConfig
    partition: PartitionConfig
    instrument: InstrumentConfig
