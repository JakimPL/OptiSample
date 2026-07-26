from dataclasses import dataclass

from optisample.config.tracker import TrackerConfig, TrackerFormat
from trackmod.core.notes.command import NoteCommand
from trackmod.core.patterns.cell import Cell
from trackmod.core.songs.song import Song
from trackmod.limits.capability import Capability
from trackmod.limits.compliance import Compliance
from trackmod.limits.table import Limits
from trackmod.module.protocol import TrackerModule
from trackmod.module.storage import Storage
from trackmod.trackers.it.limits import it_limits
from trackmod.trackers.it.module import ITModule
from trackmod.trackers.it.settings import ITSettings
from trackmod.trackers.it.spec.storage import IT_STORAGE


@dataclass(frozen=True)
class ExportTarget:
    """The tracker format a plan is written as, together with everything that format decides.

    ``compliance`` picks which bounds the module is graded against, and through :attr:`min_rows` it also
    decides how short a pattern may be -- the canonical Impulse Tracker floor is 32 rows, so material
    laid out under it is padded up to that height.
    """

    format: TrackerFormat
    compliance: Compliance
    it: ITSettings

    def bind(self, song: Song) -> TrackerModule:
        """Hand ``song`` to this format, giving a module that reports its size and writes itself."""
        match self.format:
            case TrackerFormat.IT:
                return ITModule.from_song(song, compliance=self.compliance, settings=self.it)

    @property
    def storage(self) -> Storage:
        """What each kind of content costs this format, so a caller can budget before a song exists."""
        match self.format:
            case TrackerFormat.IT:
                return IT_STORAGE

    @property
    def limits(self) -> Limits:
        """The bounds a module of this format is held to at this compliance level."""
        match self.format:
            case TrackerFormat.IT:
                return it_limits(self.compliance)

    @property
    def min_rows(self) -> int:
        """The shortest pattern this format accepts, which laid-out material is padded up to."""
        return self.limits.bound(Capability.PATTERN_ROWS).minimum

    @property
    def max_rows(self) -> int:
        """The tallest pattern this format accepts, past which material spills into the next one."""
        return self.limits.bound(Capability.PATTERN_ROWS).maximum

    def release_cell(self) -> Cell:
        """The cell that silences a channel, spelled the way this format spells it."""
        match self.format:
            case TrackerFormat.IT:
                return Cell(note=NoteCommand.CUT)


def export_target(config: TrackerConfig) -> ExportTarget:
    """Build the target the pipeline writes through from the loaded tracker config group."""
    return ExportTarget(
        format=config.format,
        compliance=config.compliance,
        it=ITSettings(global_volume=config.it.global_volume, mix_volume=config.it.mix_volume),
    )
