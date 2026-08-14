from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

import numpy as np

from optisample.artifacts.documents.plan import PlanDocument
from optisample.artifacts.paths import PipelinePaths, PlanPaths, pipeline_paths, plan_paths
from optisample.cluster.representative import Durations, MemberReadings, Weights
from optisample.config.reduce import DedupeConfig, TrimConfig
from optisample.io.audio import mono, read_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import NOTES_SUFFIX, IngestSettings
from optisample.io.source import load_source
from optisample.keys import SampleKey, sample_key
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, Manifest, NoteEvent, ProjectSpec, SourceSample
from optisample.music import midi_to_freq, note_name, pitch_label
from optisample.optimize.orchestrate.audio import decode_recordings
from optisample.optimize.plans import zone_unit_label
from optisample.progress import ProgressSink

_READ_LABEL: Final = "Reading recordings"
_STORED_LABEL: Final = "Reading stored samples"
_UNPLAYED: Final = 0.0  # the playing time a key the material never sounds gives the recording standing there
_UNSPENT_KB: Final = 1.0  # a reading allocates nothing, so the budget its manifest is joined under goes unspent
_STATED_BY_MANIFEST: Final = 0.0  # every stage writes a manifest, which records the padding it holds itself


@unique
class Stage(StrEnum):
    """Which step of a chained run a set of recordings is read from, in the order the steps ran.

    ``SUBSET``, ``LOOPED`` and ``REDUCED`` are datasets, each holding the recordings that step left behind:
    the slice taken of the source, those recordings with a loop settled over them, and the survivors the
    pre-optimization stage kept. ``OPTIMIZED`` is what one allocation stored, decoded back from the module
    it wrote, so reading the four in turn says what each step did to the set of sounds a run carries.
    """

    SUBSET = "subset"
    LOOPED = "looped"
    REDUCED = "reduced"
    OPTIMIZED = "optimized"

    @property
    def is_dataset(self) -> bool:
        """Whether this stage wrote a dataset, which is the shape a later stage reads and a slice is cut of."""
        return self in (Stage.SUBSET, Stage.LOOPED, Stage.REDUCED)


@dataclass(frozen=True)
class RecordingSource:
    """One set of recordings a space is built across: the run holding it, and where inside that run it sits.

    ``root`` is the output root a chained run filed its steps under, ``instrument_id`` the instrument that
    run carried from stage to stage, and ``stage`` the step whose recordings are read. The three together
    name a directory on disk, which is what lets a caller hold several sets side by side and read them into
    one space.
    """

    root: Path
    instrument_id: str
    stage: Stage

    @property
    def label(self) -> str:
        """How a panel names this set, unique among the sets one run root offers."""
        return f"{self.instrument_id} · {self.stage.value}"


@dataclass(frozen=True)
class ReadingSettings:
    """What reading a set of recordings is carried out with, whichever sets a caller names.

    ``dedupe`` states the identity each recording is named by, so the key a recording carries here is the
    key the pipeline's own stages know it as. ``trim`` is the span a dataset's recordings are kept over,
    which is the treatment a run gives the recordings it stores. ``strategy`` names which allocation's
    stored samples the ``OPTIMIZED`` stage offers.

    ``keep_tail`` reads each recording through the padding past its note's release, which is how far a
    struck string is heard to fall. A run stores the span its material plays, so a note held briefly is cut
    while it is still loud and its decline is read over the little of it that was kept; reading the tail as
    well hands the decay-anchored blocks the depths the sound genuinely reaches.
    """

    strategy: str
    dedupe: DedupeConfig
    trim: TrimConfig
    keep_tail: bool
    progress: ProgressSink


@dataclass(frozen=True)
class StageRecording:
    """One recording a set offers: the audio it holds, the note it plays, and the say the music gives it.

    ``source`` names the set this take came from, so a point placed in a space gathered from several of them
    answers for the run, the instrument and the stage behind it. ``signal`` is the recording as that stage
    wrote it, read over the span its note sounds. ``key`` is the identity the pipeline knows it by and
    ``weight`` the playing time the material gives that key, which is what a representative carrying the
    music is chosen by.
    """

    source: RecordingSource
    file: Path
    key: SampleKey
    signal: Signal
    sample_rate: int
    weight: float

    @property
    def label(self) -> str:
        """How a table names this recording: the stem of the file holding it, unique within its own set."""
        return self.file.stem

    @property
    def note(self) -> str:
        """The note this recording plays, as a listener names it."""
        return note_name(self.key.pitch)

    @property
    def root_hz(self) -> float:
        """The fundamental this recording rings at, which the harmonics of its own pitch are read against."""
        return midi_to_freq(self.key.pitch)

    @property
    def duration_s(self) -> float:
        """How long the recording sounds for, as the stage left it."""
        return self.signal.size / self.sample_rate


@dataclass(frozen=True)
class RecordingCorpus:
    """Every recording one space is built across, beside the sets they were gathered from.

    ``sources`` names those sets in the order their recordings stand, and each recording names its own, so a
    point picked out of the space says which run, instrument and stage the take behind it came from.
    """

    sources: tuple[RecordingSource, ...]
    recordings: tuple[StageRecording, ...]

    @property
    def size(self) -> int:
        """How many recordings the corpus holds, which is how many points its space places."""
        return len(self.recordings)

    @property
    def weights(self) -> Weights:
        """The playing time the material gives each recording, which is the say it carries in a group.

        A weighted medoid read under these stands for the take its group leans on musically rather than
        the one sitting in the geometric middle.
        """
        return np.asarray([recording.weight for recording in self.recordings], dtype=np.float64)

    @property
    def durations_s(self) -> Durations:
        """How long each recording sounds for, which is what holds a representative to a length one can shape."""
        return np.asarray([recording.duration_s for recording in self.recordings], dtype=np.float64)

    @property
    def readings(self) -> MemberReadings:
        """What each recording carries beyond its place in a space, as the grouping reads them together."""
        return MemberReadings(weights=self.weights, durations_s=self.durations_s)


def stage_dir(paths: PipelinePaths, stage: Stage) -> Path:
    """Where ``stage`` wrote what it produced, under the tree a chained run files its steps in."""
    match stage:
        case Stage.SUBSET:
            return paths.subset_dir

        case Stage.LOOPED:
            return paths.looped_dir

        case Stage.REDUCED:
            return paths.reduced_dir

        case Stage.OPTIMIZED:
            return paths.optimized_dir


def stage_dataset(source: RecordingSource) -> SourceDataset:
    """The dataset ``source`` names: its manifest, beside its recordings.

    This is the pair every later stage reads a dataset stage through, so a slice cut of it lands in the
    shape the pipeline picks up from.

    Raises:
        ValueError: when ``source`` names what an allocation stored, which its plan states rather than a
            manifest.
    """
    if not source.stage.is_dataset:
        raise ValueError(f"{source.label} holds stored samples, which its plan names rather than a manifest")

    return _dataset(source)


def stage_material(source: RecordingSource, settings: ReadingSettings) -> tuple[NoteEvent, ...]:
    """The notes the music plays through ``source``'s dataset, which is what its dynamics are cut on.

    A stage's manifest names one played note per note of the performance, so what comes back states how
    long the instrument spends at every velocity it was struck at -- the reading a velocity axis is split
    on (:func:`~optisample.optimize.layers.bands.velocity_cells`).

    Raises:
        ValueError: when ``source`` names what an allocation stored, which its plan states rather than a
            manifest.
    """
    if not source.stage.is_dataset:
        raise ValueError(f"{source.label} holds stored samples, which its plan names rather than material")

    return tuple(_one_instrument(load_source(_dataset(source), _ingest(source, settings))).material)


def available_instruments(root: Path) -> tuple[str, ...]:
    """Every instrument a run left readable under ``root``, in the order a listing names them.

    A chain files each stage under the instrument it carried, so the instruments there are to read are the
    ones the dataset stages wrote a manifest and a recordings directory for, beside the ones the allocated
    stage stored under. Naming them is what lets a reader pick a dataset off a run it has on disk.
    """
    paths = pipeline_paths(root)
    return tuple(sorted({name for stage in Stage for name in _instruments_at(stage_dir(paths, stage), stage)}))


def available_stages(root: Path, instrument_id: str, settings: ReadingSettings) -> tuple[Stage, ...]:
    """The stages a run left readable for ``instrument_id`` under ``root``, in the order they ran.

    A chain begun from an already-sliced dataset writes no subset and a run stopped early writes no
    allocation, so this is what a reader offers as the stages there are to look at.
    """
    return tuple(source.stage for source in _rooted(root, instrument_id) if _stage_ready(source, settings))


def available_sources(root: Path, settings: ReadingSettings) -> tuple[RecordingSource, ...]:
    """Every set of recordings a run left readable under ``root``, instrument by instrument.

    A run carries as many instruments as it was handed and leaves each of them at whichever steps it
    reached, so naming the pairs is what lets a reader place several sets of one run in a single space --
    an instrument at two stages, or two instruments at one.
    """
    return tuple(
        RecordingSource(root=root, instrument_id=instrument_id, stage=stage)
        for instrument_id in available_instruments(root)
        for stage in available_stages(root, instrument_id, settings)
    )


def source_recordings(source: RecordingSource, settings: ReadingSettings) -> RecordingCorpus:
    """Every recording ``source`` offers, read the way the stage behind it wrote it.

    A dataset stage is read through the ingest that stage's successor reads it through and decoded the way
    a run decodes the recordings it stores -- onset aligned, at one common rate, and cut to the span each
    carries -- so the points a space places stand for exactly the audio the pipeline worked from. The
    allocated stage is read back from the WAVs the dumper wrote, each at the rate its own sample was stored
    at, so what a space places there is what the module actually plays.
    """
    match source.stage:
        case Stage.SUBSET | Stage.LOOPED | Stage.REDUCED:
            recordings = _dataset_recordings(source, settings)

        case Stage.OPTIMIZED:
            recordings = _stored_recordings(source, settings)

    return RecordingCorpus(sources=(source,), recordings=recordings)


def gathered_recordings(sources: Sequence[RecordingSource], settings: ReadingSettings) -> RecordingCorpus:
    """Every recording the named sets offer, gathered into the one corpus a space is built across.

    The sets are read in the order given and their recordings laid end to end, so an index into the corpus
    names one take throughout. Gathering several of them is what places them under a single frame: a space
    states each block in the spread of everything it was handed, so a point's coordinates answer for the
    company it was read beside and two sets compare on what they sound like.

    Raises:
        ValueError: when ``sources`` names no set, which is a corpus with nothing to place.
    """
    if not sources:
        raise ValueError("a corpus is gathered from at least one set of recordings")

    gathered = [source_recordings(source, settings) for source in sources]
    return RecordingCorpus(
        sources=tuple(sources),
        recordings=tuple(recording for corpus in gathered for recording in corpus.recordings),
    )


@dataclass(frozen=True)
class _Listed:
    """One recording a dataset holds: the sample naming it, and the playing time the notes it serves give it."""

    sample: SourceSample
    weight: float


@dataclass(frozen=True)
class _StoredItem:
    """One sample a plan stored: the name the dumper filed it under, the note it holds, and what it serves."""

    label: str
    pitch: int
    velocity: int
    weight: float


def _source_dir(source: RecordingSource) -> Path:
    """Where the stage ``source`` names wrote what it produced, under its own run's tree."""
    return stage_dir(pipeline_paths(source.root), source.stage)


def _dataset(source: RecordingSource) -> SourceDataset:
    """The dataset ``source`` names: its manifest, beside the recordings it joins to."""
    directory = _source_dir(source)
    return SourceDataset(
        path=directory / f"{source.instrument_id}{NOTES_SUFFIX}",
        samples_dir=directory / source.instrument_id,
    )


def _plan(source: RecordingSource, settings: ReadingSettings) -> PlanPaths:
    """Where the strategy ``settings`` names left its artifacts, under the allocated stage's directory."""
    return plan_paths(_source_dir(source) / source.instrument_id, settings.strategy)


def _rooted(root: Path, instrument_id: str) -> tuple[RecordingSource, ...]:
    """Every stage of a run named as a set, before any of them is read for what it left behind."""
    return tuple(RecordingSource(root=root, instrument_id=instrument_id, stage=stage) for stage in Stage)


def _instruments_at(directory: Path, stage: Stage) -> tuple[str, ...]:
    """The instruments ``stage`` left under ``directory``, named the way that stage files them.

    A dataset stage names an instrument by the manifest it wrote beside the recordings it joins to, and the
    allocated stage by the directory it stored one instrument's artifacts under.
    """
    match stage:
        case Stage.SUBSET | Stage.LOOPED | Stage.REDUCED:
            named = (found.name.removesuffix(NOTES_SUFFIX) for found in directory.glob(f"*{NOTES_SUFFIX}"))
            return tuple(name for name in named if (directory / name).is_dir())

        case Stage.OPTIMIZED:
            return tuple(found.name for found in directory.glob("*") if found.is_dir())


def _stage_ready(source: RecordingSource, settings: ReadingSettings) -> bool:
    """Whether ``source`` names what it is read through, so a caller offers it as a set to look at."""
    match source.stage:
        case Stage.SUBSET | Stage.LOOPED | Stage.REDUCED:
            dataset = _dataset(source)
            return dataset.path.is_file() and dataset.recordings_dir.is_dir()

        case Stage.OPTIMIZED:
            return _plan(source, settings).plan_json.is_file()


def _ingest(source: RecordingSource, settings: ReadingSettings) -> IngestSettings:
    """What a stage's manifest is joined to its recordings under while a reading is taken of it.

    Every stage of a chain writes a ``.notes.json``, which records the padding each of its recordings holds
    and the notes they answer for, so the fields a manifest states for itself are the ones read back here.
    The budget stands only because a manifest carries one; a reading allocates nothing and spends none of it.
    """
    return IngestSettings(
        instrument_id=source.instrument_id,
        budget_kb=_UNSPENT_KB,
        project=ProjectSpec(name=source.instrument_id),
        pre_roll_s=_STATED_BY_MANIFEST,
        post_roll_s=_STATED_BY_MANIFEST,
        keep_tail=settings.keep_tail,
    )


def _one_instrument(manifest: Manifest) -> InstrumentSpec:
    """The single instrument a stage's dataset holds, which is the one the chain carried through it.

    Raises:
        ValueError: when the manifest names any other number of instruments.
    """
    (instrument,) = manifest.instruments
    return instrument


def _listed_once(instrument: InstrumentSpec) -> tuple[_Listed, ...]:
    """The recordings ``instrument`` holds, one apiece, each carrying the playing time of the notes it serves.

    An ingest names one recording and one played note per note of the performance
    (:func:`~optisample.io.note_extractor.load_notes`), so the two lists pair by position and a stage
    storing one recording for a run of notes names that recording once for every note it answers.
    Gathering the entries by the file each names leaves one point per recording for a space to place, and
    hands it the whole say the music it covers earns it.

    Raises:
        ValueError: when the two lists stand at different lengths, which would pair a note's playing time
            against another note's recording.
    """
    listed: dict[Path, SourceSample] = {}
    weights: dict[Path, float] = {}
    for sample, event in zip(instrument.samples, instrument.material, strict=True):
        listed.setdefault(sample.file, sample)
        weights[sample.file] = weights.get(sample.file, _UNPLAYED) + event.weight

    return tuple(_Listed(sample=sample, weight=weights[file]) for file, sample in listed.items())


def _recording(
    source: RecordingSource,
    listed: _Listed,
    signal: Signal,
    *,
    sample_rate: int,
    dedupe: DedupeConfig,
) -> StageRecording:
    """One decoded recording as the space reads it, under the identity its own dataset names it by."""
    return StageRecording(
        source=source,
        file=listed.sample.file,
        key=sample_key(listed.sample, dedupe),
        signal=signal,
        sample_rate=sample_rate,
        weight=listed.weight,
    )


def _dataset_recordings(source: RecordingSource, settings: ReadingSettings) -> tuple[StageRecording, ...]:
    """Every recording a written dataset holds, decoded the way a run decodes the ones it stores.

    A run decodes the survivors dedup picked; a reading takes the whole listed grid through that same
    decode, so a stage's recordings arrive on the terms the pipeline itself put them on. The ones whose
    peak stayed under the silence floor carry no audio to place and are left out.
    """
    instrument = _one_instrument(load_source(_dataset(source), _ingest(source, settings)))
    listed = _listed_once(instrument)
    decoded = decode_recordings(
        [entry.sample for entry in listed],
        settings.trim,
        settings.progress,
        label=_READ_LABEL,
    )
    return tuple(
        _recording(source, entry, signal, sample_rate=decoded.sample_rate, dedupe=settings.dedupe)
        for entry, signal in zip(listed, decoded.signals)
        if signal is not None
    )


def _stored_items(plan: PlanDocument) -> tuple[_StoredItem, ...]:
    """The samples ``plan`` stored, each named the way the dumper wrote its WAV out.

    Both strategies name a stored sample after the pitch it is rooted at, so either plan reads back into
    one list of items and the recording behind each is found by the name that wrote it.
    """
    if plan.zones is not None:
        return tuple(
            _StoredItem(
                label=zone_unit_label(index, zone.representative),
                pitch=zone.representative,
                velocity=zone.representative_velocity,
                weight=zone.weight,
            )
            for index, zone in enumerate(plan.zones)
        )

    return tuple(
        _StoredItem(
            label=pitch_label(item.pitch),
            pitch=item.pitch,
            velocity=item.representative_velocity,
            weight=item.weight,
        )
        for item in plan.pitches or []
    )


def _stored_recording(source: RecordingSource, paths: PlanPaths, item: _StoredItem) -> StageRecording:
    """One stored sample read back as the WAV the dumper wrote, at the rate that sample was stored at.

    A stored sample already holds the span the allocation paid for, so it is read as it stands: what a
    space places here is the audio the module plays, resampled and requantized as the budget could afford.
    """
    file = paths.sample_wav(item.label)
    data, sample_rate = read_wav(file)
    return StageRecording(
        source=source,
        file=file,
        key=SampleKey(pitch=item.pitch, velocity=item.velocity),
        signal=mono(data),
        sample_rate=sample_rate,
        weight=item.weight,
    )


def _stored_recordings(source: RecordingSource, settings: ReadingSettings) -> tuple[StageRecording, ...]:
    """Every sample one allocation stored, in the order its plan kept them.

    Raises:
        FileNotFoundError: when the strategy named left no plan under the allocated stage's directory.
    """
    paths = _plan(source, settings)
    plan = PlanDocument.model_validate_json(paths.plan_json.read_text(encoding="utf-8"))
    items = _stored_items(plan)
    return tuple(
        _stored_recording(source, paths, item)
        for item in settings.progress.track(items, label=_STORED_LABEL, total=len(items))
    )
