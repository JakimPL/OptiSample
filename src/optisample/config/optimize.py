from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from optisample.config.base import ConfigModel
from optisample.config.layers import LayersConfig
from optisample.config.stage import StageConfig
from trackmod.core.samples.depth import BitDepth

Method = Literal["exact", "lagrangian"]

EVERY_SAMPLE: Final = 0  # the ``max_samples`` value that keeps whatever the target format numbers


class SweepConfig(ConfigModel):
    """What one clip may be stored as: the format it is kept at, and how its stored span is bounded.

    ``rates`` is the ladder a stored rate is chosen from, and a clip's own rate joins it, so the ladder
    states the rates worth stepping down to and leaves "store it as recorded" to follow from the recording
    itself. The reduction reads the rung the recording's own content asks for
    (:func:`~optisample.optimize.reduce.bandwidth.stored_format`), so every rung stays available to
    material that reaches it. ``depths`` are the depths a stored sample is offered at, and ``compress`` applies
    dynamics on the way to the quantizer where a depth is shallow enough to hear the headroom it buys.
    Naming both depths puts the eighth bit among the things a byte buys: a shallow copy costs half the
    frames of a deep one, so the objective weighs storing one sample deeply against storing two shallowly
    -- a trade worth making wherever the level a waveform carries is handed to an envelope (``carrier``),
    since a level-flat waveform spends the whole of a shallow grid on timbre.

    What the sweep then prices per sample is the stored span and the rate: keeping the played length
    against keeping the attack plus the loop the loop stage settled, each offered at the settled rung and
    at the ``rate_headroom`` rungs of the ladder above it, which
    :func:`~optisample.optimize.reduce.bandwidth.stored_encodings` enumerates in that order. Offering the
    wider rungs is what puts band among the things a byte buys, so a run with room to spare can store
    fewer samples and keep more of each one's spectrum; what those rungs are worth is
    :attr:`~optisample.config.reduce.BandwidthConfig.discard_penalty`. A headroom of 0 prices the settled
    rung alone.

    ``carrier`` states what a stored sample holds. Set, each waveform is its recording divided by the gain
    the instrument's own volume envelope applies, so the sample keeps the timbre and the envelope carries
    the level -- which is what lets a shallow grid spend itself on sound rather than on a decline the curve
    states anyway. Left unset, each waveform holds the recording as it was played, level and all.

    A carrier travels with the curve it was divided by
    (:attr:`~optisample.dsp.surrogate.sample.StoredSample.level`), so the surrogate renderer
    (:func:`~optisample.dsp.surrogate.render.render`) puts one out at the level the module plays it at and
    every reading taken from a written plan is taken on what a listener hears. The sweep itself prices each
    clip against the curve that clip alone states, so what a plan's own readings stand above the sweep's is
    what one shared envelope costs the keys written under it.
    """

    rates: Annotated[tuple[int, ...], Field(min_length=1)]
    rate_headroom: Annotated[int, Field(ge=0)]
    depths: Annotated[tuple[int, ...], Field(min_length=1)]
    dither: bool
    noise_shaping: bool
    compress: bool
    carrier: bool

    @model_validator(mode="after")
    def _depths_are_storable(self) -> Self:
        """Hold every offered depth to one a tracker sample is written at.

        Raises:
            ValueError: when a depth names a grid no format stores, which no encoding could be written to.
        """
        storable = {int(depth) for depth in BitDepth}
        offered = [depth for depth in self.depths if depth not in storable]
        if offered:
            raise ValueError(f"depths {offered} are stored by no tracker format, against {sorted(storable)}")

        return self


class BudgetConfig(ConfigModel):
    """Budget-solver settings, and what a note's distortion is worth to the objective.

    ``energy_exponent`` raises each note's own energy to a power and scales its distortion by the result,
    so the objective states the error a listener meets in the mix rather than the error measured against
    the note alone. Full scale weighs 1.0: ``0.0`` prices every note alike, ``0.5`` follows its amplitude,
    ``1.0`` its energy, and ``0.3`` the loudness an ear reports for that energy.

    ``max_samples`` is the most stored samples a plan may keep, which pitch-zone grouping meets by
    storing wider zones; :data:`EVERY_SAMPLE` keeps as many as the target format numbers. The cap is the
    grouped strategy's to honour, the ungrouped one keeping a recording per key it plays.

    ``resolution`` is the most byte totals a partition walk resolves a budget into
    (:func:`~optisample.optimize.dp.byte_grid`). The walk holds a row per total it can reach, so this is
    what decides the table it fills and the time it takes, and it holds them steady as the packs grow:
    the granularity follows the budget, and each zone gives up under one step of it. ``None`` walks every
    budget to the byte, which is the exact allocation and costs a table the size of the pack.
    """

    method: Method
    energy_exponent: Annotated[float, Field(ge=0.0)]
    max_samples: Annotated[int, Field(ge=EVERY_SAMPLE)]
    resolution: Annotated[int, Field(gt=0)] | None


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
