from dataclasses import dataclass

from optisample.config.codec import EncodeConfig
from optisample.config.export import EnvelopeConfig
from optisample.config.render import PlaybackConfig
from optisample.io.tracker.envelope import EnvelopeGrid, envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.seed import DEFAULT_SEED


@dataclass(frozen=True)
class ExportContext:
    """Config the exporter needs beyond a plan: how to re-encode, how it plays, and what it is written as.

    ``envelope`` states how a released note is let go, which is the one part of the curve an instrument
    plays its voices down by that a recording never states. ``seed`` seeds the per-sample dither so
    re-encoding a plan reproduces the exact bytes it budgeted.

    ``carrier`` states what a stored sample holds, and with it which way round the export runs: set, the
    shape is fitted from the recordings and each waveform is what that written curve leaves; unset, each
    waveform holds its recording and the shape is fitted to what the stored levels leave.
    """

    encode: EncodeConfig
    playback: PlaybackConfig
    target: ExportTarget
    envelope: EnvelopeConfig
    carrier: bool
    seed: int = DEFAULT_SEED

    @property
    def envelope_grid(self) -> EnvelopeGrid:
        """What a written curve is held to: the module's own clock, the format's two grids, and the release."""
        return envelope_grid(self.target, tempo=self.playback.tempo, release_s=self.envelope.release_s)
