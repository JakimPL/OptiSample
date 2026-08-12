from optisample.config.analysis import AnalysisConfig
from optisample.config.base import ConfigModel
from optisample.config.cluster import ClusterConfig
from optisample.config.codec import CodecConfig, EncodeConfig
from optisample.config.export import ExportConfig
from optisample.config.loop import LoopConfig
from optisample.config.optimize import OptimizeConfig
from optisample.config.reduce import ReduceConfig
from optisample.config.runtime import RuntimeConfig
from optisample.config.subsonic import SubsonicConfig
from optisample.config.synth import SynthConfig


class OptiConfig(ConfigModel):
    """Every tunable group, gathered by the pipeline stage that acts on it.

    Field names match the ``opticonfig`` layout one-to-one: a stage
    (:class:`~optisample.config.stage.StageConfig`) is a directory holding one file per group, and the
    standalone groups are a file each.

    ``subsonic`` stands on its own because it is the band the whole run works in rather than one stage's
    knob: the very first stage reads its source past it, and every stage after that reads a dataset
    already holding what a listener has.
    """

    analysis: AnalysisConfig
    cluster: ClusterConfig
    subsonic: SubsonicConfig
    codec: CodecConfig
    loop: LoopConfig
    reduce: ReduceConfig
    optimize: OptimizeConfig
    export: ExportConfig
    synth: SynthConfig
    runtime: RuntimeConfig

    @property
    def encode(self) -> EncodeConfig:
        """The bundle the surrogate encoder needs, gathered from the two stages that decide its parts.

        The codec stage states how a span is shaped and quantized and the loop stage states how a wrap is
        blended and what level a region is held at, so assembling the bundle here keeps each knob owned by
        the stage it belongs to while the encoder receives one validated value.
        """
        return EncodeConfig(
            seam=self.loop.seam,
            envelope=self.loop.envelope,
            dynamics=self.codec.dynamics,
            headroom_db=self.codec.quantize.headroom_db,
            release_fade_s=self.codec.quantize.release_fade_s,
        )
