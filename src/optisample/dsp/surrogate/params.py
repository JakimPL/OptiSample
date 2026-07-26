from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig


@dataclass(frozen=True)
class EncodingParams:
    """One encoding configuration for a stored sample: the per-sample decision variables."""

    target_rate: int
    depth_bits: int
    trim_s: float | None = None
    dither: bool = True
    noise_shaping: bool = False
    loop: bool = False


@dataclass(frozen=True)
class EncodeContext:
    """The fixed surroundings of one :func:`~optisample.dsp.surrogate.encode.encode` call.

    Bundles the root pitch stamped on the stored sample, the encode config (loop detection +
    normalization peak), and the dither RNG.
    """

    root_pitch: int
    config: EncodeConfig
    rng: np.random.Generator | None = None
