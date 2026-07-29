from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.dynamics import DynamicsConfig
from optisample.config.stage import StageConfig


class LoopConfig(ConfigModel):
    """Loop-point detection: search band, periodicity/sustain gates, placement, and seam crossfade.

    ``min_periods`` and ``min_loop_s`` together set the shortest loop that may be stored, which is the
    floor every candidate clears. ``placements`` is how many starts are spread through the steady region
    and ``length_multiples`` the lengths each start is offered, so the two say how many loops
    :func:`~optisample.dsp.loop.loop_candidates` lays out for a clip to choose among.
    """

    min_hz: float
    max_hz: float
    min_correlation: float
    min_periods: Annotated[int, Field(ge=1)]
    min_loop_s: Annotated[float, Field(gt=0.0)]
    placements: Annotated[int, Field(ge=1)]
    length_multiples: Annotated[tuple[Annotated[int, Field(ge=1)], ...], Field(min_length=1)]
    attack_skip_s: float
    tail_skip_s: float
    max_estimation_s: float
    sustain_decay_ratio: float
    crossfade_s: float


class QuantizeConfig(ConfigModel):
    """How a span is shaped for storage: the level it is stored at, and the way it ends.

    Storing hot spends the whole depth on the recording; ``headroom_db`` is what the dither is left to
    move in above that peak (see :func:`~optisample.dsp.quantize.headroom_peak`). ``release_fade_s`` ramps
    the last stretch of a stored span to silence, so a sample cut at the length its material asks for ends
    on silence and a tracker plays it out rather than stepping off it.
    """

    headroom_db: float
    release_fade_s: Annotated[float, Field(ge=0.0)]


class EncodeConfig(ConfigModel):
    """What :func:`optisample.dsp.surrogate.encode` needs beyond one swept ``EncodingParams`` point.

    ``peak_reference`` is the one amplitude every clip of an instrument is normalized against, set for
    a format keeping no per-sample multiplier so the balance between its samples is carried in the PCM.
    Left unset, each clip is normalized against its own peak and the balance is restored on playback.
    """

    loop: LoopConfig
    dynamics: DynamicsConfig
    headroom_db: float
    release_fade_s: float
    peak_reference: float | None = None


class CodecConfig(StageConfig):
    """How one recording becomes a stored sample: where it loops, how it is shaped, how it is quantized."""

    loop: LoopConfig
    quantize: QuantizeConfig
    dynamics: DynamicsConfig

    @property
    def encode(self) -> EncodeConfig:
        """The bundle the surrogate encoder needs: loop detection, compression and the stored headroom."""
        return EncodeConfig(
            loop=self.loop,
            dynamics=self.dynamics,
            headroom_db=self.quantize.headroom_db,
            release_fade_s=self.quantize.release_fade_s,
        )
