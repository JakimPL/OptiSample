from typing import Literal

from optisample.config.base import ConfigModel

Interpolation = Literal["none", "linear", "cubic", "sinc"]


class RenderConfig(ConfigModel):
    """How openmpt123 renders a module file: output rate, interpolation filter, output gain (dB)."""

    sample_rate: int
    interpolation: Interpolation
    gain_db: float


class PlaybackConfig(ConfigModel):
    """The clock a module starts on: ``speed`` ticks per row at ``tempo``, which both formats share."""

    speed: int
    tempo: int
