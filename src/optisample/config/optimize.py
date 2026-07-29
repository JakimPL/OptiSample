from typing import Annotated, Final, Literal

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.layers import LayersConfig
from optisample.config.stage import StageConfig

Method = Literal["exact", "lagrangian"]

EVERY_SAMPLE: Final = 0  # the ``max_samples`` value that keeps whatever the target format numbers
TRIMMED_ONLY: Final = 0  # the ``loop_choices`` value offering the trimmed sample alone


class SweepConfig(ConfigModel):
    """The encoding axes to sweep for one clip.

    ``rates`` is the ladder of reduced stored rates every clip is offered. A clip is also always offered
    its own rate, so the ladder states the rates worth stepping down to and leaves "store it as recorded"
    to follow from the recording itself.

    ``loop_choices`` is how far into :func:`~optisample.dsp.loop.loop_candidates` the sweep reaches, and
    the trimmed sample is enumerated beside them, so the frontier prices a loop against storing none and
    keeps whichever the objective prefers. :data:`TRIMMED_ONLY` sweeps the trimmed sample by itself.
    """

    rates: Annotated[tuple[int, ...], Field(min_length=1)]
    depths: tuple[int, ...]
    dither: bool
    noise_shaping: bool
    loop_choices: Annotated[int, Field(ge=TRIMMED_ONLY)]
    compress: tuple[bool, ...]


class BudgetConfig(ConfigModel):
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


class OptimizeConfig(StageConfig):
    """The search and what it spends: the encodings tried, the budget solver, layering, and the level map.

    ``sweep`` is the grid one clip is offered, ``budget`` how the allocation spends bytes over it,
    ``layers`` how many dynamics a key may keep, and ``velocity`` the shape of the map from a played
    velocity to the volume a note sounds at.
    """

    sweep: SweepConfig
    budget: BudgetConfig
    layers: LayersConfig
    velocity: VelocityConfig
