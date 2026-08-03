from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel


class RankingQuotaConfig(ConfigModel):
    """How many pairs of each question a listening set spends a listener's time on.

    The three single-axis questions ask whether the metric orders one degradation correctly; ``trade``
    asks the question a byte budget asks, two degradations of nearly equal size against each other.
    Every pair is two clips to hear and compare, so the four together state the whole cost of the set.
    """

    loop: Annotated[int, Field(ge=0)]
    rate: Annotated[int, Field(ge=0)]
    depth: Annotated[int, Field(ge=0)]
    trade: Annotated[int, Field(ge=0)]


class RankingConfig(ConfigModel):
    """What a listening set is built to: the grid it prices, the questions it asks, and how it is blinded.

    The sweep settles one rate and one depth per clip from the recording's own content, so the encodings
    an allocation chooses among differ in how much of the recording they store and in nothing else.
    ``depths`` and ``rate_steps`` widen that grid along the two axes the settling holds fixed, which is
    what lets a label speak to a quantization step and a bandwidth step as well as to a stored span.

    ``byte_tolerance`` is how near two encodings must come in size for their pair to be worth asking as a
    trade, as a share of the larger: two encodings of different size are ordered by their price as much
    as by their sound. ``seed`` settles the order the pairs are met in and the side each member takes.
    """

    depths: Annotated[tuple[int, ...], Field(min_length=1)]
    rate_steps: Annotated[int, Field(ge=0)]
    quota: RankingQuotaConfig
    byte_tolerance: Annotated[float, Field(ge=0.0, le=1.0)]
    seed: int
