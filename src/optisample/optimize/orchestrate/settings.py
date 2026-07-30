from dataclasses import dataclass
from typing import Final

from optisample.config.codec import EncodeConfig
from optisample.config.layers import LayersConfig
from optisample.config.loop import LoopConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import EVERY_SAMPLE, SweepConfig, VelocityConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.composite import CompositeFidelity, build_composite
from optisample.optimize.plans import Method
from optisample.parallel import IN_PROCESS
from optisample.progress import NO_PROGRESS, ProgressSink
from optisample.seed import DEFAULT_SEED

LOOPS_OFFERED: Final = True  # a run prices its loops unless it is asked to store played spans alone


@dataclass(frozen=True)
class OptimizeSettings:
    """Knobs for one optimization run (bundled to keep the call site small).

    Carries the config the run needs -- the encoding sweep grid, how a recording is looped, the
    pre-optimization reductions, the velocity layering, the encode config, the fidelity metric's own config, the velocity-map shaping, the
    solver method, how steeply a note's own energy scales what its distortion costs
    (:func:`~optisample.optimize.weighting.energy_weight`) and the most samples a plan may store -- all
    sourced from config at the entry point. ``target`` is the tracker format the
    plan will be written as, which is what prices every stored sample the allocation considers and how
    many layers it may store. ``seed`` drives the dither RNG, ``workers`` how many processes the stages
    that fan out share their work between, and ``progress`` is where each stage reports how far through
    it is; the three describe how the run is carried out rather than what it computes, so they keep code
    defaults. ``loops`` says whether the loop stage runs at all, which is the one switch that leaves every
    sample storing the span it plays.
    """

    sweep: SweepConfig
    loop: LoopConfig
    reduce: ReduceConfig
    layers: LayersConfig
    encode: EncodeConfig
    metrics: MetricsConfig
    velocity: VelocityConfig
    method: Method
    energy_exponent: float
    max_samples: int
    target: ExportTarget
    loops: bool = LOOPS_OFFERED
    seed: int = DEFAULT_SEED
    workers: int = IN_PROCESS
    progress: ProgressSink = NO_PROGRESS

    @property
    def sample_cap(self) -> int:
        """The most samples a plan may store: what the run asks for, held inside what the format numbers.

        The format's own sample count bounds every plan it could write, so a run asking for more than that
        is answered with the format's, and :data:`~optisample.config.optimize.EVERY_SAMPLE` asks for it
        directly.
        """
        if self.max_samples == EVERY_SAMPLE:
            return self.target.max_samples

        return min(self.max_samples, self.target.max_samples)

    @property
    def composite(self) -> CompositeFidelity:
        """The fidelity metric this run scores with, assembled from ``metrics``.

        Held as the config it is built from, so the run has one source for the metric and every process
        that scores on its behalf assembles the same one.
        """
        return build_composite(self.metrics)
