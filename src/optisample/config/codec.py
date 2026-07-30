from typing import Annotated

from pydantic import Field

from optisample.config.base import ConfigModel
from optisample.config.dynamics import DynamicsConfig
from optisample.config.loop import SeamConfig
from optisample.config.stage import StageConfig


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

    ``seam`` is the blend the wrap of an already-settled loop is closed with, which is the one thing the
    encoder still decides about a loop: where it sits comes settled from the loop stage.
    """

    seam: SeamConfig
    dynamics: DynamicsConfig
    headroom_db: float
    release_fade_s: float
    peak_reference: float | None = None


class CodecConfig(StageConfig):
    """How one recording becomes a stored sample: how it is shaped, and how it is quantized."""

    quantize: QuantizeConfig
    dynamics: DynamicsConfig
