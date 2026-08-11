from dataclasses import dataclass
from typing import Final

from optisample.config.tracker import TrackerConfig, TrackerFormat
from trackmod.core.instruments.unit import InstrumentUnit
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.cell import Cell
from trackmod.core.songs.song import Song
from trackmod.limits.bound import Bound
from trackmod.limits.capability import Capability
from trackmod.limits.compliance import Compliance
from trackmod.limits.table import Limits
from trackmod.module.instrument import InstrumentFile
from trackmod.module.protocol import TrackerModule
from trackmod.module.storage import Storage
from trackmod.spec.levels import MAX_VOLUME
from trackmod.trackers.it.instrument_file import ITInstrumentFile
from trackmod.trackers.it.limits import it_limits
from trackmod.trackers.it.module import ITModule
from trackmod.trackers.it.settings import ITSettings
from trackmod.trackers.it.spec.storage import IT_STORAGE
from trackmod.trackers.xm.effects.catalog import XM_EFFECTS
from trackmod.trackers.xm.instrument_file import XMInstrumentFile
from trackmod.trackers.xm.limits import xm_limits
from trackmod.trackers.xm.module import XMModule
from trackmod.trackers.xm.settings import XMSettings
from trackmod.trackers.xm.spec.storage import XM_STORAGE

_ON_THE_ROW: Final = 0


@dataclass(frozen=True)
class SampleLevels:
    """The two levels a format keeps beside one stored sample: what a bare note plays at, and the gain over it.

    ``volume`` is the level a cell stating no volume column sounds; ``gain`` multiplies whatever level
    does play. Which of them a format lets a caller state is the format's own answer
    (:attr:`ExportTarget.sample_level_bound`), so a stored level is written through this pair rather than
    into a field named by the caller.
    """

    volume: int
    gain: int


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
    xm: XMSettings

    def bind(self, song: Song) -> TrackerModule:
        """Hand ``song`` to this format, giving a module that reports its size and writes itself."""
        match self.format:
            case TrackerFormat.IT:
                return ITModule.from_song(song, compliance=self.compliance, settings=self.it)
            case TrackerFormat.XM:
                return XMModule.from_song(song, compliance=self.compliance, settings=self.xm)

    def instrument_file(self, unit: InstrumentUnit) -> InstrumentFile:
        """Hand one instrument and its own samples to this format, giving a file that stands on its own.

        A module is the piece the plan auditions itself with; a file of this kind is one voice out of it,
        which is what a player loading a single instrument reads. Both are graded against the same
        compliance level, so an instrument written beside a module is held to what that module was.
        """
        match self.format:
            case TrackerFormat.IT:
                return ITInstrumentFile.from_unit(unit, compliance=self.compliance)
            case TrackerFormat.XM:
                return XMInstrumentFile.from_unit(unit, compliance=self.compliance)

    @property
    def storage(self) -> Storage:
        """What each kind of content costs this format, so a caller can budget before a song exists."""
        match self.format:
            case TrackerFormat.IT:
                return IT_STORAGE
            case TrackerFormat.XM:
                return XM_STORAGE

    @property
    def limits(self) -> Limits:
        """The bounds a module of this format is held to at this compliance level."""
        match self.format:
            case TrackerFormat.IT:
                return it_limits(self.compliance)
            case TrackerFormat.XM:
                return xm_limits(self.compliance)

    @property
    def stores_sample_gain(self) -> bool:
        """Whether this format keeps a per-sample multiplier, so a stored level can be restored on playback.

        Impulse Tracker gives every sample its own 0-64 gain, which is what lets each one be stored as
        hot as its depth allows and the instrument's balance be written beside it. FastTracker 2 pins
        the same capability to full gain, so an instrument written as XM carries its balance in the PCM.
        """
        return self.limits.bound(Capability.SAMPLE_GAIN).minimum < MAX_VOLUME

    @property
    def sample_level_bound(self) -> Bound:
        """The steps the level one stored sample carries is written on, which is the strongest this format has.

        Impulse Tracker applies a sample's own gain whatever a pattern states, so a level written there
        holds however the sample is played. FastTracker 2 keeps its per-sample level in the volume a note
        without a volume column plays at, which is the level that format states for a sample of its own.
        """
        capability = Capability.SAMPLE_GAIN if self.stores_sample_gain else Capability.SAMPLE_VOLUME
        return self.limits.bound(capability)

    def sample_levels(self, step: int) -> SampleLevels:
        """``step`` written as the two levels this format keeps beside a sample (:attr:`sample_level_bound`).

        Whichever of the two carries the level, the other stands at full, so the pair states the step once
        and a reader of either format finds it where that format keeps it.
        """
        if self.stores_sample_gain:
            return SampleLevels(volume=MAX_VOLUME, gain=step)

        return SampleLevels(volume=step, gain=MAX_VOLUME)

    @property
    def max_instruments(self) -> int:
        """How many instruments this format numbers, which is how many velocity layers a plan may store."""
        return self.limits.bound(Capability.INSTRUMENTS).maximum

    @property
    def max_samples(self) -> int:
        """How many samples this format numbers, which caps the stored zones summed over every layer."""
        return self.limits.bound(Capability.SAMPLES).maximum

    @property
    def max_samples_per_instrument(self) -> int:
        """How many samples one instrument of this format owns, which is how wide a written slot may be.

        Impulse Tracker lets an instrument reach the whole sample table, so a velocity band is written as
        one instrument however many zones it stores. FastTracker 2 gives each instrument sixteen samples
        of its own, which is what a wide band is cut into slots to fit
        (:func:`~optisample.optimize.layers.slots.pack_slots`).
        """
        return self.limits.bound(Capability.SAMPLES_PER_INSTRUMENT).maximum

    @property
    def envelope_point_bound(self) -> Bound:
        """How many breakpoints an envelope of this format holds, which is what a fitted shape is given.

        Impulse Tracker numbers twenty-five, FastTracker 2 twelve, and the shape an instrument is played
        down by takes every one of them the release does not
        (:func:`~optisample.optimize.export.envelope.shape_nodes`).
        """
        return self.limits.bound(Capability.ENVELOPE_POINTS)

    @property
    def envelope_tick_bound(self) -> Bound:
        """The ticks an envelope breakpoint may sit on, which bounds how long a written curve runs."""
        return self.limits.bound(Capability.ENVELOPE_TICK)

    @property
    def envelope_value_bound(self) -> Bound:
        """The grid an envelope node's value sits on, which a fitted level is written onto."""
        return self.limits.bound(Capability.ENVELOPE_VALUE)

    @property
    def min_rows(self) -> int:
        """The shortest pattern this format accepts, which laid-out material is padded up to."""
        return self.limits.bound(Capability.PATTERN_ROWS).minimum

    @property
    def max_rows(self) -> int:
        """The tallest pattern this format accepts, past which material spills into the next one."""
        return self.limits.bound(Capability.PATTERN_ROWS).maximum

    @property
    def min_pitch(self) -> int:
        """The lowest MIDI note this format's keyboard reaches."""
        return Note(self.limits.bound(Capability.NOTE).minimum).midi

    @property
    def max_pitch(self) -> int:
        """The highest MIDI note this format's keyboard reaches."""
        return Note(self.limits.bound(Capability.NOTE).maximum).midi

    def tempo(self, tempo_bpm: float) -> int:
        """The clock this format counts envelope ticks in, for material played at ``tempo_bpm``.

        Both formats measure envelope time in ticks and a tick lasts ``2.5 / tempo`` seconds, so a curve
        written for one clock plays at the rate that clock runs and a recorded tempo reaches a written
        instrument by naming the tempo the module holding it plays at. A format states its tempo as a
        whole number, so the nearest one to what the material was played at is the clock it is written on.

        Raises:
            ValueError: when the nearest whole tempo falls outside the range this format states.
        """
        stated = round(tempo_bpm)
        bound = self.limits.bound(Capability.TEMPO)
        if not bound.contains(stated):
            raise ValueError(f"tempo {stated} is outside the {self.format.upper()} range {bound}")

        return stated

    def key(self, pitch: int) -> Note:
        """The key this format's keyboard plays ``pitch`` on.

        Trackers count their keyboards from C-0, one octave below MIDI's own numbering, and each format
        numbers a different stretch of that keyboard -- Impulse Tracker all ten octaves, FastTracker 2
        the lowest eight -- so which pitches a module can name is the target's to answer.

        Raises:
            ValueError: when the MIDI note falls outside the keys this format numbers.
        """
        if not self.min_pitch <= pitch <= self.max_pitch:
            raise ValueError(
                f"MIDI note {pitch} is outside the {self.format.upper()} key range "
                f"{self.min_pitch}..{self.max_pitch}"
            )

        return Note.from_midi(pitch)

    def release_cell(self) -> Cell:
        """The cell that silences a channel, spelled the way this format spells it.

        Impulse Tracker keeps a cut in the note column; FastTracker 2 numbers its note column for keys
        alone and reaches the same silence through the effect that cuts a channel within the row.
        """
        match self.format:
            case TrackerFormat.IT:
                return Cell(note=NoteCommand.CUT)
            case TrackerFormat.XM:
                return Cell(effect=XM_EFFECTS.note_cut(_ON_THE_ROW))


def export_target(config: TrackerConfig) -> ExportTarget:
    """Build the target the pipeline writes through from the loaded tracker config group."""
    return ExportTarget(
        format=config.format,
        compliance=config.compliance,
        it=ITSettings(global_volume=config.it.global_volume, mix_volume=config.it.mix_volume),
        xm=XMSettings(tracker=config.xm.tracker),
    )
