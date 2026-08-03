from optisample.config.metrics import MetricsConfig
from optisample.config.ranking import RankingConfig
from optisample.config.spectral import SpectralConfig
from optisample.config.stage import StageConfig


class AnalysisConfig(StageConfig):
    """How audio is measured, wherever a stage measures it: the standalone primitives and the composite.

    ``spectral`` sizes the diagnostics a notebook or a report reads off a signal on its own, and
    ``metrics`` is the fidelity a reconstruction is scored with, so every stage comparing two signals
    compares them the same way. ``ranking`` is the listening set that fidelity is itself measured
    against, which is what puts a number on how well the composite orders what a listener hears.
    """

    spectral: SpectralConfig
    metrics: MetricsConfig
    ranking: RankingConfig
