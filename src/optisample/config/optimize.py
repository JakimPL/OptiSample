from typing import Annotated, Final, Literal

from pydantic import Field

from optisample.config.base import ConfigModel

Method = Literal["exact", "lagrangian"]

EVERY_SAMPLE: Final = 0  # the ``max_samples`` value that keeps whatever the target format numbers


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
    """Budget-solver settings, and what a note's distortion is worth to the objective.

    ``energy_exponent`` raises each note's own energy to a power and scales its distortion by the result,
    so the objective states the error a listener meets in the mix rather than the error measured against
    the note alone. Full scale weighs 1.0: ``0.0`` prices every note alike, ``0.5`` follows its amplitude,
    ``1.0`` its energy, and ``0.3`` the loudness an ear reports for that energy.

    ``max_samples`` is the most stored samples a plan may keep, which pitch-zone grouping meets by
    storing wider zones; :data:`EVERY_SAMPLE` keeps as many as the target format numbers. The cap is the
    grouped strategy's to honour, the ungrouped one keeping a recording per key it plays.
    """

    method: Method
    energy_exponent: Annotated[float, Field(ge=0.0)]
    max_samples: Annotated[int, Field(ge=EVERY_SAMPLE)]


class VelocityConfig(ConfigModel):
    """Velocity->volume map shaping: how far below the reference a silent velocity is clamped."""

    loudness_floor_lu: float
