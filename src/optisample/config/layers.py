from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel


class LayersConfig(ConfigModel):
    """How many recordings a key may keep for different dynamics, and what earns it another one.

    ``max_layers`` caps the velocity bands an instrument stores -- one written instrument each -- so it is
    both the vocabulary dial and what keeps a plan inside the format's instrument count; ``1`` keeps one
    recording per key. ``nodes`` is how finely the velocity axis is cut before bands are assembled from
    those cells, which sets how many splits the allocation prices and how much scoring each one costs.
    ``min_gain`` is the relative objective improvement an extra layer must buy to be preferred to a plan
    holding fewer, so a layer is spent where it is audible.
    """

    max_layers: Annotated[int, Field(ge=1)]
    nodes: Annotated[int, Field(ge=1)]
    min_gain: Annotated[float, Field(ge=0.0)]
