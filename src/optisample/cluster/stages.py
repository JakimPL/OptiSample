from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Final

from optisample.artifacts.documents.plan import PlanDocument
from optisample.artifacts.paths import PipelinePaths, PlanPaths, pipeline_paths, plan_paths
from optisample.config.reduce import DedupeConfig, TrimConfig
from optisample.io.audio import mono, read_wav
from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import NOTES_SUFFIX, IngestSettings
from optisample.io.source import load_source
from optisample.keys import SampleKey, sample_key
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, Manifest, ProjectSpec, SourceSample
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


@dataclass(frozen=True)
class StageSettings:
    """What reading one stage's recordings is carried out with.

    ``instrument_id`` names the instrument a chained run carried from stage to stage, which is what every
    stage files its dataset under. ``dedupe`` states the identity each recording is named by, so the key a
    recording carries here is the key the pipeline's own stages know it as. ``trim`` is the span a dataset's
    recordings are kept over, which is the treatment a run gives the recordings it stores. ``strategy``
    names which allocation's stored samples the ``OPTIMIZED`` stage offers.

    ``keep_tail`` reads each recording through the padding past its note's release, which is how far a
    struck string is heard to fall. A run stores the span its material plays, so a note held briefly is cut
    while it is still loud and its decline is read over the little of it that was kept; reading the tail as
    well hands the decay-anchored blocks the depths the sound genuinely reaches.
    """

    instrument_id: str
    strategy: str
    dedupe: DedupeConfig
    trim: TrimConfig
    keep_tail: bool
    progress: ProgressSink


@dataclass(frozen=True)
class StageRecording:
    """One recording a stage offers: the audio it holds, the note it plays, and the say the music gives it.

    ``signal`` is the recording as that stage wrote it, read over the span its note sounds. ``key`` is the
    identity the pipeline knows it by and ``weight`` the playing time the material gives that key, which is
    what a representative carrying the music is chosen by.
    """

    file: Path
    key: SampleKey
    signal: Signal
    sample_rate: int
    weight: float

    @property
    def label(self) -> str:
        """How a table names this recording: the stem of the file holding it, unique within its stage."""
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
class StageCorpus:
    """Every recording one stage of a run offers, beside the stage and the instrument they came from."""

    stage: Stage
    instrument_id: str
    recordings: tuple[StageRecording, ...]

    @property
    def size(self) -> int:
        """How many recordings the stage offers, which is how many points its space holds."""
        return len(self.recordings)


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


def available_stages(root: Path, settings: StageSettings) -> tuple[Stage, ...]:
    """The stages a run left readable under ``root``, in the order they ran.

    A chain begun from an already-sliced dataset writes no subset and a run stopped early writes no
    allocation, so this is what a reader offers as the stages there are to look at.
    """
    paths = pipeline_paths(root)
    return tuple(stage for stage in Stage if _stage_ready(stage_dir(paths, stage), stage, settings))


def stage_recordings(root: Path, stage: Stage, settings: StageSettings) -> StageCorpus:
    """Every recording ``stage`` offers under the run rooted at ``root``, read the way that stage wrote it.

    A dataset stage is read through the ingest that stage's successor reads it through and decoded the way
    a run decodes the recordings it stores -- onset aligned, at one common rate, and cut to the span each
    carries -- so the points a space places stand for exactly the audio the pipeline worked from. The
    allocated stage is read back from the WAVs the dumper wrote, each at the rate its own sample was stored
    at, so what a space places there is what the module actually plays.
    """
    directory = stage_dir(pipeline_paths(root), stage)
    match stage:
        case Stage.SUBSET | Stage.LOOPED | Stage.REDUCED:
            recordings = _dataset_recordings(directory, settings)

        case Stage.OPTIMIZED:
            recordings = _stored_recordings(directory, settings)

    return StageCorpus(stage=stage, instrument_id=settings.instrument_id, recordings=recordings)


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


def _dataset(directory: Path, instrument_id: str) -> SourceDataset:
    """The dataset a stage wrote under ``directory``: its manifest, beside the recordings it joins to."""
    return SourceDataset(path=directory / f"{instrument_id}{NOTES_SUFFIX}", samples_dir=directory / instrument_id)


def _plan(directory: Path, settings: StageSettings) -> PlanPaths:
    """Where the strategy ``settings`` names left its artifacts, under the allocated stage's directory."""
    return plan_paths(directory / settings.instrument_id, settings.strategy)


def _stage_ready(directory: Path, stage: Stage, settings: StageSettings) -> bool:
    """Whether ``stage`` left what it is read through, so a caller offers it as a stage to look at."""
    match stage:
        case Stage.SUBSET | Stage.LOOPED | Stage.REDUCED:
            dataset = _dataset(directory, settings.instrument_id)
            return dataset.path.is_file() and dataset.recordings_dir.is_dir()

        case Stage.OPTIMIZED:
            return _plan(directory, settings).plan_json.is_file()


def _ingest(settings: StageSettings) -> IngestSettings:
    """What a stage's manifest is joined to its recordings under while a reading is taken of it.

    Every stage of a chain writes a ``.notes.json``, which records the padding each of its recordings holds
    and the notes they answer for, so the fields a manifest states for itself are the ones read back here.
    The budget stands only because a manifest carries one; a reading allocates nothing and spends none of it.
    """
    return IngestSettings(
        instrument_id=settings.instrument_id,
        budget_kb=_UNSPENT_KB,
        project=ProjectSpec(name=settings.instrument_id),
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


def _recording(listed: _Listed, signal: Signal, *, sample_rate: int, dedupe: DedupeConfig) -> StageRecording:
    """One decoded recording as the space reads it, under the identity its own dataset names it by."""
    return StageRecording(
        file=listed.sample.file,
        key=sample_key(listed.sample, dedupe),
        signal=signal,
        sample_rate=sample_rate,
        weight=listed.weight,
    )


def _dataset_recordings(directory: Path, settings: StageSettings) -> tuple[StageRecording, ...]:
    """Every recording a written dataset holds, decoded the way a run decodes the ones it stores.

    A run decodes the survivors dedup picked; a reading takes the whole listed grid through that same
    decode, so a stage's recordings arrive on the terms the pipeline itself put them on. The ones whose
    peak stayed under the silence floor carry no audio to place and are left out.
    """
    instrument = _one_instrument(load_source(_dataset(directory, settings.instrument_id), _ingest(settings)))
    listed = _listed_once(instrument)
    decoded = decode_recordings(
        [entry.sample for entry in listed],
        settings.trim,
        settings.progress,
        label=_READ_LABEL,
    )
    return tuple(
        _recording(entry, signal, sample_rate=decoded.sample_rate, dedupe=settings.dedupe)
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


def _stored_recording(paths: PlanPaths, item: _StoredItem) -> StageRecording:
    """One stored sample read back as the WAV the dumper wrote, at the rate that sample was stored at.

    A stored sample already holds the span the allocation paid for, so it is read as it stands: what a
    space places here is the audio the module plays, resampled and requantized as the budget could afford.
    """
    file = paths.sample_wav(item.label)
    data, sample_rate = read_wav(file)
    return StageRecording(
        file=file,
        key=SampleKey(pitch=item.pitch, velocity=item.velocity),
        signal=mono(data),
        sample_rate=sample_rate,
        weight=item.weight,
    )


def _stored_recordings(directory: Path, settings: StageSettings) -> tuple[StageRecording, ...]:
    """Every sample one allocation stored, in the order its plan kept them.

    Raises:
        FileNotFoundError: when the strategy named left no plan under ``directory``.
    """
    paths = _plan(directory, settings)
    plan = PlanDocument.model_validate_json(paths.plan_json.read_text(encoding="utf-8"))
    items = _stored_items(plan)
    return tuple(
        _stored_recording(paths, item) for item in settings.progress.track(items, label=_STORED_LABEL, total=len(items))
    )
