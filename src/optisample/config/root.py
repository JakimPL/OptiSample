from optisample.config.analysis import AnalysisConfig
from optisample.config.base import ConfigModel
from optisample.config.codec import CodecConfig
from optisample.config.export import ExportConfig
from optisample.config.optimize import OptimizeConfig
from optisample.config.reduce import ReduceConfig
from optisample.config.runtime import RuntimeConfig
from optisample.config.synth import SynthConfig


class OptiConfig(ConfigModel):
    """Every tunable group, gathered by the pipeline stage that acts on it.

    Field names match the ``opticonfig`` layout one-to-one: a stage
    (:class:`~optisample.config.stage.StageConfig`) is a directory holding one file per group, and the
    two standalone groups are a file each.
    """

    analysis: AnalysisConfig
    codec: CodecConfig
    reduce: ReduceConfig
    optimize: OptimizeConfig
    export: ExportConfig
    synth: SynthConfig
    runtime: RuntimeConfig
