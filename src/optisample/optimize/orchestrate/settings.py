from dataclasses import dataclass
from typing import Final

from optisample.config.dsp import EncodeConfig
from optisample.config.optimize import SweepConfig, VelocityConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.composite import CompositeFidelity
from optisample.optimize.plans import Method
from optisample.progress import NO_PROGRESS, ProgressSink

DEFAULT_SEED: Final = 137


@dataclass(frozen=True)
class OptimizeSettings:
    """Knobs for one optimization run (bundled to keep the call site small).

    Carries the config the run needs -- the encoding sweep grid, the pre-optimization reductions, the
    encode config, the prebuilt composite fidelity, the velocity-map shaping and the solver method --
    all sourced from config at the entry point. ``target`` is the tracker format the plan will be
    written as, which is what prices every stored sample the allocation considers. ``seed`` drives the
    dither RNG and ``progress`` is where each stage reports how far through it is; both describe how the
    run is carried out rather than what it computes, so they keep code defaults.
    """

    sweep: SweepConfig
    reduce: ReduceConfig
    encode: EncodeConfig
    composite: CompositeFidelity
    velocity: VelocityConfig
    method: Method
    target: ExportTarget
    seed: int = DEFAULT_SEED
    progress: ProgressSink = NO_PROGRESS
