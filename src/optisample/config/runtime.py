from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel


class RuntimeConfig(ConfigModel):
    """How a run is carried out, as opposed to what it computes.

    ``workers`` is how many processes share the stages that fan out -- narrowing each pitch's stored
    grid, rendering each demo note -- with ``0`` asking for one per core. A value of ``1`` carries the
    whole run in the calling process, which is what a profile reads end to end.
    """

    workers: Annotated[int, Field(ge=0)]
