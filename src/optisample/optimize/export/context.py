from dataclasses import dataclass

from optisample.config.dsp import EncodeConfig
from optisample.config.render import PlaybackConfig
from optisample.io.tracker.target import ExportTarget
from optisample.seed import DEFAULT_SEED


@dataclass(frozen=True)
class ExportContext:
    """Config the exporter needs beyond a plan: how to re-encode, how it plays, and what it is written as.

    ``seed`` seeds the per-sample dither so re-encoding a plan reproduces the exact bytes it budgeted.
    """

    encode: EncodeConfig
    playback: PlaybackConfig
    target: ExportTarget
    seed: int = DEFAULT_SEED
