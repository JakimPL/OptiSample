from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.codec import EncodeConfig

TRIMMED: Final = None  # the ``loop_choice`` storing the trimmed sample in place of any loop


@dataclass(frozen=True)
class EncodingParams:
    """One encoding configuration for a stored sample: the per-sample decision variables.

    ``compress`` asks for the dynamics stage ahead of the quantizer, which narrows the crest factor so
    more of the depth's grid carries material. ``loop_choice`` names which of
    :func:`~optisample.dsp.loop.loop_candidates` the sample is stored around, counting from 0;
    :data:`TRIMMED` stores the trimmed sample instead, which is what puts "store no loop" on the grid as
    an option of its own.
    """

    target_rate: int
    depth_bits: int
    trim_s: float | None = None
    dither: bool = True
    noise_shaping: bool = False
    loop_choice: int | None = TRIMMED
    compress: bool = False


@dataclass(frozen=True)
class EncodeContext:
    """The fixed surroundings of one :func:`~optisample.dsp.surrogate.encode.encode` call.

    Bundles the root pitch stamped on the stored sample, the encode config (loop detection, compression
    and the stored headroom), and the dither RNG.
    """

    root_pitch: int
    config: EncodeConfig
    rng: np.random.Generator | None = None
