from typing import Literal

from optisample.config.base import ConfigModel

Method = Literal["exact", "lagrangian"]


class SweepConfig(ConfigModel):
    """The encoding axes to sweep for one clip, and how candidate stored rates are derived."""

    rates: tuple[int, ...] | None
    rate_divisors: tuple[int, ...]
    min_rate: int
    depths: tuple[int, ...]
    dither: bool
    noise_shaping: bool
    loops: tuple[bool, ...]
    compress: tuple[bool, ...]


class OptimizeConfig(ConfigModel):
    """Budget-solver settings."""

    method: Method


class VelocityConfig(ConfigModel):
    """Velocity->volume map shaping: how far below the reference a silent velocity is clamped."""

    loudness_floor_lu: float
