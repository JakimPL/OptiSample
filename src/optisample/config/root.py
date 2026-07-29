from optisample.config.base import ConfigModel
from optisample.config.dsp import (
    EncodeConfig,
    LoopConfig,
    QuantizeConfig,
    SpectralConfig,
)
from optisample.config.dynamics import DynamicsConfig
from optisample.config.layers import LayersConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import OptimizeConfig, SweepConfig, VelocityConfig
from optisample.config.reduce import ReduceConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.runtime import RuntimeConfig
from optisample.config.synth import SynthConfig
from optisample.config.tracker import TrackerConfig


class OptiConfig(ConfigModel):
    """Every tunable group; field names match the ``opticonfig`` YAML filenames one-to-one."""

    loop: LoopConfig
    quantize: QuantizeConfig
    dynamics: DynamicsConfig
    spectral: SpectralConfig
    metrics: MetricsConfig
    sweep: SweepConfig
    optimize: OptimizeConfig
    reduce: ReduceConfig
    layers: LayersConfig
    velocity: VelocityConfig
    render: RenderConfig
    playback: PlaybackConfig
    tracker: TrackerConfig
    synth: SynthConfig
    runtime: RuntimeConfig

    @property
    def encode(self) -> EncodeConfig:
        """The bundle the surrogate encoder needs: loop detection, compression and the stored headroom."""
        return EncodeConfig(
            loop=self.loop,
            dynamics=self.dynamics,
            headroom_db=self.quantize.headroom_db,
            release_fade_s=self.quantize.release_fade_s,
        )
