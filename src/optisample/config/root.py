"""The root configuration object aggregating every tunable group.

``OptiConfig`` has one field per group, each mapping 1:1 to a YAML file in the ``opticonfig`` package
(``loop`` <-> ``loop.yaml``, ``sweep`` <-> ``sweep.yaml``, ...), which is what the loader relies on and
what makes the config discoverable when tuning. ``encode`` is a derived bundle (not its own file).
"""

from __future__ import annotations

from optisample.config.base import ConfigModel
from optisample.config.dsp import EncodeConfig, LoopConfig, QuantizeConfig, SpectralConfig
from optisample.config.metrics import MetricsConfig
from optisample.config.optimize import OptimizeConfig, SweepConfig, VelocityConfig
from optisample.config.render import PlaybackConfig, RenderConfig
from optisample.config.synth import SynthConfig


class OptiConfig(ConfigModel):  # pylint: disable=too-many-instance-attributes
    """Every tunable group; field names match the ``opticonfig`` YAML filenames one-to-one."""

    loop: LoopConfig
    quantize: QuantizeConfig
    spectral: SpectralConfig
    metrics: MetricsConfig
    sweep: SweepConfig
    optimize: OptimizeConfig
    velocity: VelocityConfig
    render: RenderConfig
    playback: PlaybackConfig
    synth: SynthConfig

    @property
    def encode(self) -> EncodeConfig:
        """The bundle the surrogate encoder needs: loop detection plus the normalization peak."""
        return EncodeConfig(loop=self.loop, target_peak=self.quantize.target_peak)
