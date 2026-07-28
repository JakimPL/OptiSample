from dataclasses import dataclass
from typing import Final

from optisample.config.dsp import EncodeConfig
from optisample.config.layers import LayersConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import SweepConfig, VelocityConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.composite import CompositeFidelity, build_composite
from optisample.optimize.plans import Method
from optisample.parallel import IN_PROCESS
from optisample.progress import NO_PROGRESS, ProgressSink

DEFAULT_SEED: Final = 137


@dataclass(frozen=True)
class OptimizeSettings:
    """Knobs for one optimization run (bundled to keep the call site small).

    Carries the config the run needs -- the encoding sweep grid, the pre-optimization reductions, the
    velocity layering, the encode config, the fidelity metric's own config, the velocity-map shaping, the
    solver method and how steeply a note's own energy scales what its distortion costs
    (:func:`~optisample.optimize.weighting.energy_weight`) -- all sourced from config at the entry point.
    ``target`` is the tracker format the
    plan will be written as, which is what prices every stored sample the allocation considers and how
    many layers it may store. ``seed`` drives the dither RNG, ``workers`` how many processes the stages
    that fan out share their work between, and ``progress`` is where each stage reports how far through
    it is; the three describe how the run is carried out rather than what it computes, so they keep code
    defaults.
    """

    sweep: SweepConfig
    reduce: ReduceConfig
    layers: LayersConfig
    encode: EncodeConfig
    metrics: MetricsConfig
    velocity: VelocityConfig
    method: Method
    energy_exponent: float
    target: ExportTarget
    seed: int = DEFAULT_SEED
    workers: int = IN_PROCESS
    progress: ProgressSink = NO_PROGRESS

    @property
    def composite(self) -> CompositeFidelity:
        """The fidelity metric this run scores with, assembled from ``metrics``.

        Held as the config it is built from, so the run has one source for the metric and every process
        that scores on its behalf assembles the same one.
        """
        return build_composite(self.metrics)
