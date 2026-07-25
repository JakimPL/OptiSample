"""The knobs one optimization run consumes, bundled so the driver's call sites stay small."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig, VelocityConfig
from optisample.metrics.composite import CompositeFidelity
from optisample.optimize.plans import Method

DEFAULT_SEED: Final = 0


@dataclass(frozen=True)
class OptimizeSettings:
    """Knobs for one optimization run (bundled to keep the call site small).

    Carries the config the run needs -- the encoding sweep grid, the encode config, the prebuilt
    composite fidelity, the velocity-map shaping and the solver method -- all sourced from config at
    the entry point. ``seed`` drives the dither RNG (not a tuning knob, so it keeps a code default).
    """

    sweep: SweepConfig
    encode: EncodeConfig
    composite: CompositeFidelity
    velocity: VelocityConfig
    method: Method
    seed: int = DEFAULT_SEED
