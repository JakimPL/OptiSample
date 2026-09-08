from typing import Annotated

from pydantic import Field
from trackmod import BitDepth

from optisample.config.base import ConfigModel


class RankingQuotaConfig(ConfigModel):
    """How many pairs of each question a listening set spends a listener's time on.

    The four single-axis questions ask whether the metric orders one degradation correctly; ``trade``
    asks the question a byte budget asks, two degradations of nearly equal size against each other.
    Every pair is two clips to hear and compare, so the five together state the whole cost of the set.
    """

    loop: Annotated[int, Field(ge=0)]
    rate: Annotated[int, Field(ge=0)]
    depth: Annotated[int, Field(ge=0)]
    compress: Annotated[int, Field(ge=0)]
    trade: Annotated[int, Field(ge=0)]


class RankingConfig(ConfigModel):
    """What a listening set is built to: the grid it prices, the questions it asks, and how it is blinded.

    The sweep settles one rate and one depth per clip from the recording's own content, so the encodings
    an allocation chooses among differ in how much of the recording they store and in nothing else.
    ``depths`` and ``rate_steps`` widen that grid along the two axes the settling holds fixed, which is
    what lets a label speak to a quantization step and a bandwidth step as well as to a stored span.

    ``byte_tolerance`` is how near two encodings must come in size for their pair to be worth asking as a
    trade, as a share of the larger: two encodings of different size are ordered by their price as much
    as by their sound. ``min_duration_s`` is how long a note class must play to be asked about, since a
    degradation shows itself over a decay. ``repeats`` is how many of the chosen questions are put a
    second time, which reads a listener's own consistency and gives the agreement a ceiling. ``seed``
    settles the order the pairs are met in, where the repeats fall, and the side each member takes.
    """

    depths: Annotated[tuple[BitDepth, ...], Field(min_length=1)]
    rate_steps: Annotated[int, Field(ge=0)]
    quota: RankingQuotaConfig
    byte_tolerance: Annotated[float, Field(ge=0.0, le=1.0)]
    min_duration_s: Annotated[float, Field(gt=0.0)]
    repeats: Annotated[int, Field(ge=0)]
    seed: int
