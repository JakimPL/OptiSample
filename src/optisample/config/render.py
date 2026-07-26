from typing import Literal

from optisample.config.base import ConfigModel

Interpolation = Literal["none", "linear", "cubic", "sinc"]


class RenderConfig(ConfigModel):
    """How openmpt123 renders an ``.IT`` file: output rate, interpolation filter, output gain (dB)."""

    sample_rate: int
    interpolation: Interpolation
    gain_db: float


class PlaybackConfig(ConfigModel):
    """IT global playback: ticks/row via ``speed``+``tempo``, plus global and mix volume."""

    speed: int
    tempo: int
    global_volume: int
    mix_volume: int
