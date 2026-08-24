from dataclasses import dataclass

from optisample.config.codec import EncodeConfig
from optisample.config.export import EnvelopeConfig
from optisample.config.render import PlaybackConfig
from optisample.io.tracker.envelope import EnvelopeGrid, envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.seed import DEFAULT_SEED
from trackmod.core.timing.clock import tick_seconds


@dataclass(frozen=True)
class ExportContext:
    """Config the exporter needs beyond a plan: how to re-encode, how it plays, and what it is written as.

    ``envelope`` states how a released note is let go, which is the one part of the curve an instrument
    plays its voices down by that a recording never states. ``seed`` seeds the per-sample dither so
    re-encoding a plan reproduces the exact bytes it budgeted.

    ``min_carried_attack_ticks`` is the room a written curve needs to state a recording's own attack,
    counted in the ticks its corners turn on. It is what settles which way round the export runs for
    material a plan asked to store as carriers: a module whose recordings all rise slowly enough fits the
    shape first and stores what that written curve leaves, and one holding a recording that rises faster
    keeps every waveform as it was played and fits the shape to what the stored levels leave
    (:func:`~optisample.optimize.export.build.written_voices`).
    """

    encode: EncodeConfig
    playback: PlaybackConfig
    target: ExportTarget
    envelope: EnvelopeConfig
    min_carried_attack_ticks: float
    seed: int = DEFAULT_SEED

    @property
    def shortest_carried_attack_s(self) -> float:
        """The briefest attack a written curve has room to state, in seconds of the clock the module runs."""
        return self.min_carried_attack_ticks * tick_seconds(self.playback.tempo)

    @property
    def envelope_grid(self) -> EnvelopeGrid:
        """What a written curve is held to: the module's own clock, the format's two grids, and the release."""
        return envelope_grid(self.target, tempo=self.playback.tempo, release_s=self.envelope.release_s)
