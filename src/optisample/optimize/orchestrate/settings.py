from dataclasses import dataclass
from typing import Final

from optisample.config.codec import EncodeConfig
from optisample.config.export import EnvelopeConfig
from optisample.config.layers import LayersConfig
from optisample.config.loop import LoopConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import EVERY_SAMPLE, SweepConfig, VelocityConfig
from optisample.config.reduce import ReduceConfig
from optisample.config.render import PlaybackConfig
from optisample.io.tracker.envelope import EnvelopeGrid, envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.metrics.composite import CompositeFidelity, build_composite
from optisample.optimize.carrier import CurveSettings
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
    (:func:`~optisample.optimize.weighting.energy_weight`), the most samples a plan may store and the
    byte totals its budget is resolved into (:func:`~optisample.optimize.dp.byte_grid`) -- all
    sourced from config at the entry point. ``target`` is the tracker format the
    plan will be written as, which is what prices every stored sample the allocation considers and how
    many layers it may store, while ``playback`` and ``envelope`` state the clock and the release a volume
    curve is written on -- which is what lets the sweep price a stored carrier against the very curve the
    module will play it down by. ``seed`` drives the dither RNG, ``workers`` how many processes the stages
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
    resolution: int | None
    target: ExportTarget
    playback: PlaybackConfig
    envelope: EnvelopeConfig
    loops: bool = LOOPS_OFFERED
    seed: int = DEFAULT_SEED
    workers: int = IN_PROCESS
    progress: ProgressSink = NO_PROGRESS

    @property
    def envelope_grid(self) -> EnvelopeGrid:
        """What a curve priced by the sweep is held to: the module's clock, the format's grids, the release.

        The same bundle the exporter writes its curves on
        (:attr:`~optisample.optimize.export.context.ExportContext.envelope_grid`), so a carrier is priced
        against a curve of exactly the resolution the written module carries.
        """
        return envelope_grid(self.target, tempo=self.playback.tempo, release_s=self.envelope.release_s)

    @property
    def curve_settings(self) -> CurveSettings:
        """What reading the curve one recording states is carried out with, for a run pricing carriers."""
        return CurveSettings(
            config=self.encode,
            target=self.target,
            grid=self.envelope_grid,
            min_attack_ticks=self.sweep.min_carried_attack_ticks,
        )

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
