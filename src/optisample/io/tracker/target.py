from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from trackmod import (
    Compliance,
    InstrumentFile,
    InstrumentUnit,
    ITInstrumentFile,
    ITModule,
    Song,
    TrackerModule,
    XMInstrumentFile,
    XMModule,
)
from trackmod.core.notes.command import NoteCommand
from trackmod.core.notes.pitch import Note
from trackmod.core.patterns.cell import Cell
from trackmod.limits.bound import Bound
from trackmod.limits.capability import Capability
from trackmod.limits.table import Limits
from trackmod.module.storage import Storage
from trackmod.spec.levels import MAX_VOLUME
from trackmod.trackers.it.limits import it_limits
from trackmod.trackers.it.settings import ITSettings
from trackmod.trackers.it.spec.identity import INSTRUMENT_EXTENSION as IT_INSTRUMENT_EXTENSION
from trackmod.trackers.it.spec.sizes import NAME_BYTES as IT_NAME_BYTES
from trackmod.trackers.it.spec.storage import IT_STORAGE
from trackmod.trackers.xm.effects.catalog import XM_EFFECTS
from trackmod.trackers.xm.limits import xm_limits
from trackmod.trackers.xm.settings import XMSettings
from trackmod.trackers.xm.spec.identity import INSTRUMENT_EXTENSION as XM_INSTRUMENT_EXTENSION
from trackmod.trackers.xm.spec.sizes import NAME_BYTES as XM_NAME_BYTES
from trackmod.trackers.xm.spec.storage import XM_STORAGE

from optisample.config.tracker import TrackerConfig, TrackerFormat
from optisample.music import note_name

_ON_THE_ROW: Final = 0
_SHORTEST_ID: Final = 1  # instrument-id characters a label keeps however wide the key and dynamic run
_UNIT_MAKEUP: Final = 1.0  # what a set holding nothing states as the level it is balanced against
_QUIETEST_STEP: Final = 1  # the softest step that still sounds, so a quiet sample is heard rather than dropped


@dataclass(frozen=True)
class SampleLevels:
    """The levels a format keeps beside one stored sample, and the one its instrument carries over them.

    ``volume`` is the level a cell stating no volume column sounds, ``gain`` multiplies whatever level
    does play, and ``instrument`` is the instrument's own global volume, applied to every note it starts.
    Which of them a format lets a caller state is the format's own answer
    (:attr:`ExportTarget.sample_level_bounds`), so a stored level is written through this triple rather
    than into a field named by the caller.
    """

    volume: int
    gain: int
    instrument: int


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
    def sample_level_bounds(self) -> tuple[Bound, ...]:
        """The grids the level one stored sample carries is written across, strongest first.

        Impulse Tracker applies a sample's own gain and its instrument's global volume to every note it
        starts, so a level written across the pair lands on the product of two grids -- a lattice fine
        enough that the waveform under it keeps the whole of its depth. FastTracker 2 states its
        per-sample level in the volume a note without a volume column plays at and holds its instruments
        at one volume, so the pair states that single grid.
        """
        stated = Capability.SAMPLE_GAIN if self.stores_sample_gain else Capability.SAMPLE_VOLUME
        return (self.limits.bound(stated), self.limits.bound(Capability.INSTRUMENT_VOLUME))

    def sample_levels(self, steps: Sequence[int]) -> SampleLevels:
        """``steps`` written as the levels this format keeps for a sample (:attr:`sample_level_bounds`).

        Whichever field carries the per-sample level, the other stands at full, so the triple states each
        step once and a reader of either format finds it where that format keeps it.
        """
        stated, instrument = steps
        if self.stores_sample_gain:
            return SampleLevels(volume=MAX_VOLUME, gain=stated, instrument=instrument)

        return SampleLevels(volume=stated, gain=MAX_VOLUME, instrument=instrument)

    @property
    def name_bytes(self) -> int:
        """How wide a name field this format keeps for an instrument and for a sample.

        Impulse Tracker spends twenty-six bytes on a name and FastTracker 2 twenty-two, and a writer fills
        the width exactly, so a name longer than its format holds is cut where the field ends. Reading the
        width off the format is what lets a name be fitted before it is written.
        """
        match self.format:
            case TrackerFormat.IT:
                return IT_NAME_BYTES
            case TrackerFormat.XM:
                return XM_NAME_BYTES

    @property
    def instrument_extension(self) -> str:
        """The extension a standalone instrument of this format is written with, including the leading dot.

        Naming it off the format alone is what lets a caller settle where a written instrument lands
        before there is a unit to write, which is how a directory per format is prepared once for a whole
        set of recordings.
        """
        match self.format:
            case TrackerFormat.IT:
                return IT_INSTRUMENT_EXTENSION
            case TrackerFormat.XM:
                return XM_INSTRUMENT_EXTENSION

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
        (:func:`~optisample.io.tracker.envelope.shape_nodes`).
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

    def names(self, pitch: int) -> bool:
        """Whether this format's keyboard reaches ``pitch``, which is what an instrument may be rooted at.

        Trackers count their keyboards from C-0, one octave below MIDI's own numbering, and each format
        numbers a different stretch of that keyboard -- Impulse Tracker all ten octaves, FastTracker 2 the
        lowest eight -- so a recording played at the very top of a piano is one Impulse Tracker names and
        FastTracker 2 leaves to the formats that reach it.
        """
        return self.min_pitch <= pitch <= self.max_pitch

    def key(self, pitch: int) -> Note:
        """The key this format's keyboard plays ``pitch`` on (:meth:`names`).

        Raises:
            ValueError: when the MIDI note falls outside the keys this format numbers.
        """
        if not self.names(pitch):
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


def sample_label(instrument_id: str, *, pitch: int, velocity: int, target: ExportTarget) -> str:
    """The name a tracker's own sample list calls one stored recording.

    Naming the recording rather than the key tells the samples of a layered instrument apart, since one
    key may be stored once per velocity band and the tracker lists them side by side. The key and the
    dynamic are what tell them apart, so the recording's own name gives up whatever room the pair asks
    for and the whole label lands inside the field the format keeps
    (:attr:`ExportTarget.name_bytes`).
    """
    stated = f"{note_name(pitch)} v{velocity}"
    kept = max(_SHORTEST_ID, target.name_bytes - len(stated) - 1)
    return f"{instrument_id[:kept]} {stated}"


def balanced_gains(makeups: Sequence[float], target: ExportTarget) -> tuple[int, ...]:
    """``makeups`` as the per-sample steps ``target`` keeps, scaled so the largest takes the top one.

    Every sample of a set is stored as hot as its own depth allows, which spends the whole grid on one
    recording and leaves the instrument flat -- a naturally quiet key comes back as loud as a bright one.
    The per-sample level the format keeps is where that balance is restored, stated relative to the sample
    asking for the most of it so the whole set fits the steps available. A format pinning that level to
    full scale carries the balance in the PCM instead, so every sample there reports the same top step.

    The softest step that still sounds is the floor, so a sample far under the loudest is heard quietly
    rather than silenced outright.
    """
    if not target.stores_sample_gain:
        return tuple(MAX_VOLUME for _ in makeups)

    loudest = max(makeups, default=_UNIT_MAKEUP)
    return tuple(max(_QUIETEST_STEP, round(MAX_VOLUME * makeup / loudest)) for makeup in makeups)


def export_target(config: TrackerConfig) -> ExportTarget:
    """Build the target the pipeline writes through from the loaded tracker config group."""
    return ExportTarget(
        format=config.format,
        compliance=config.compliance,
        it=ITSettings(global_volume=config.it.global_volume, mix_volume=config.it.mix_volume),
        xm=XMSettings(tracker=config.xm.tracker),
    )
