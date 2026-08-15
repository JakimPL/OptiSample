from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel


class HoldConfig(ConfigModel):
    """The curve a level is held back along once it rises past the threshold.

    ``threshold_db`` sits below whatever peak the level is read against, so material is shaped by how far
    it spreads under its own loudest moment rather than by how hot it happens to stand. Levels reaching
    past the threshold keep ``1 / ratio`` of their excess, arriving at that slope over a bend ``knee_db``
    wide.
    """

    threshold_db: float
    ratio: Annotated[float, Field(ge=1.0)]
    knee_db: Annotated[float, Field(ge=0.0)]


class DynamicsConfig(HoldConfig):
    """Soft-knee compression ahead of the quantizer: where the curve acts, how hard, and how quickly.

    The curve itself is :class:`HoldConfig`, read against the clip's own peak. ``rms_window_s`` is how long
    the level detector averages over and ``gain_smoothing_s`` how long the gain takes to follow it, which
    together keep the reduction tracking the material rather than each sample.
    """

    rms_window_s: Annotated[float, Field(gt=0.0)]
    gain_smoothing_s: Annotated[float, Field(gt=0.0)]
