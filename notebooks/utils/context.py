from __future__ import annotations

from dataclasses import dataclass

from notebooks.utils.viz import SpectrogramStyle
from optisample.config import OptiConfig, load_config
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.metrics import CompositeFidelity, build_composite
from trackmod.module.storage import Storage


@dataclass(frozen=True)
class NotebookContext:
    """What a notebook settles once before it draws anything: its config, and what is assembled from it.

    Every notebook reads the same config and then wants the same three things off it -- the composite its
    fidelity is scored with, the tracker format its footprints are counted in, and the reading its
    spectrograms are drawn under. Holding them together has each panel measure on one footing and keeps
    the assembly in one place, so a notebook cell states what it shows rather than how it was built.
    """

    config: OptiConfig
    composite: CompositeFidelity
    target: ExportTarget

    @property
    def storage(self) -> Storage:
        """What each kind of content costs the configured format, which every footprint is counted in."""
        return self.target.storage

    @property
    def spectrogram(self) -> SpectrogramStyle:
        """The transform and the depth every spectrogram of this notebook is drawn under.

        The depth is the one the log-spectral metrics read a signal over, so a difference visible in a
        drawn spectrogram is a difference the composite was in a position to score.
        """
        return SpectrogramStyle(
            params=self.config.analysis.spectral.stft,
            dynamic_range_db=self.config.analysis.metrics.preprocess.dynamic_range_db,
        )


def notebook_context() -> NotebookContext:
    """The context a notebook opens with, read from the config the whole project shares."""
    config = load_config()
    return NotebookContext(
        config=config,
        composite=build_composite(config.analysis.metrics),
        target=export_target(config.export.tracker),
    )
