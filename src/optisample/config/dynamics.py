from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel


class DynamicsConfig(ConfigModel):
    """Soft-knee compression ahead of the quantizer: where it acts, how hard, and how quickly.

    ``threshold_db`` sits below the clip's own peak, so a recording is shaped by how far its material
    spreads under its loudest moment rather than by how hot it happens to have been captured. Levels
    reaching past the threshold keep ``1 / ratio`` of their excess, arriving at that slope over a bend
    ``knee_db`` wide. ``rms_window_s`` is how long the level detector averages over and
    ``gain_smoothing_s`` how long the gain takes to follow it, which together keep the reduction
    tracking the material rather than each sample.
    """

    threshold_db: float
    ratio: Annotated[float, Field(ge=1.0)]
    knee_db: Annotated[float, Field(ge=0.0)]
    rms_window_s: Annotated[float, Field(gt=0.0)]
    gain_smoothing_s: Annotated[float, Field(gt=0.0)]
