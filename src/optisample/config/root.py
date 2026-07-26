from optisample.config.base import ConfigModel
from optisample.config.dsp import (
    EncodeConfig,
    LoopConfig,
    QuantizeConfig,
    SpectralConfig,
)
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import OptimizeConfig, SweepConfig, VelocityConfig
from optisample.config.reduce import ReduceConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.synth import SynthConfig
from optisample.config.tracker import TrackerConfig


class OptiConfig(ConfigModel):
    """Every tunable group; field names match the ``opticonfig`` YAML filenames one-to-one."""

    loop: LoopConfig
    quantize: QuantizeConfig
    spectral: SpectralConfig
    metrics: MetricsConfig
    sweep: SweepConfig
    optimize: OptimizeConfig
    reduce: ReduceConfig
    velocity: VelocityConfig
    render: RenderConfig
    playback: PlaybackConfig
    tracker: TrackerConfig
    synth: SynthConfig

    @property
    def encode(self) -> EncodeConfig:
        """The bundle the surrogate encoder needs: loop detection plus the normalization peak."""
        return EncodeConfig(
            loop=self.loop,
            target_peak=self.quantize.target_peak,
        )
