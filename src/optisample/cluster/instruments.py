from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.artifacts.documents.clustered import CarrierLayerRecord, clustered_document, layer_record
from optisample.artifacts.documents.sample import (
    SampleDocument,
    faithful_offer,
    read_sample,
    sample_decomposition,
    sample_loops,
)
from optisample.artifacts.paths import SAMPLE_EXTENSION, instrument_files_dir
from optisample.artifacts.serialize import write_json
from optisample.carrier.audition import carrier_audition
from optisample.carrier.instrument import CarrierInstrument, carrier_instrument
from optisample.carrier.source import CarrierSource
from optisample.carrier.store import CarrierSettings
from optisample.cluster.corpus import DescribedCorpus, describe_corpus
from optisample.cluster.partition import partition
from optisample.cluster.representative import MemberReadings, grouping
from optisample.cluster.space import sample_space
from optisample.cluster.stages import (
    Stage,
    StageCorpus,
    StageRecording,
    StageSettings,
    stage_material,
    stage_recordings,
)
from optisample.config.cluster import ClusterConfig
from optisample.config.codec import EncodeConfig
from optisample.config.loop import FeatureConfig
from optisample.dsp.envelope import decompose, level_reading
from optisample.dsp.surrogate import NO_LOOPS, EncodingParams
from optisample.io.audio import write_wav
from optisample.io.tracker.envelope import envelope_grid
from optisample.io.tracker.target import ExportTarget
from optisample.model import NoteEvent
from optisample.music import midi_to_freq
from optisample.optimize.layers.bands import VelocityBand, VelocityLayers, velocity_cells
from optisample.progress import ProgressSink, ProgressStep, counting

MANIFEST_SUFFIX: Final = ".clustered.json"
AUDITIONS_DIR: Final = "auditions"

_ALONE: Final = 1  # recordings a band holds while there is nothing to tell apart inside it
_CUT_LABEL: Final = "Cutting velocity bands"
_WRITE_LABEL: Final = "Writing clustered instruments"


@dataclass(frozen=True)
class ClusterSettings:
    """What writing a clustered set of recordings as carrier instruments is carried out with.

    ``reading`` states how the stage's recordings are decoded and named, ``cluster`` how the space they
    make is built, cut and written, and ``target`` which format the instruments are held to. ``tempo_bpm``
    is the clock the volume envelopes are counted in ticks of, which travels in the manifest since an
    instrument file carries no clock of its own.
    """

    stage: Stage
    reading: StageSettings
    cluster: ClusterConfig
    features: FeatureConfig
    encode: EncodeConfig
    target: ExportTarget
    release_s: float
    tempo_bpm: float
    seed: int
    workers: int
    progress: ProgressSink

    @property
    def instrument_id(self) -> str:
        """The instrument the run carried from stage to stage, which every artifact is filed under."""
        return self.reading.instrument_id

    @property
    def tempo(self) -> int:
        """The whole-number clock a written envelope counts its ticks in."""
        return round(self.tempo_bpm)


@dataclass(frozen=True)
class ClusteredArtifacts:
    """What one clustered carrier run left behind, and what its choices came to.

    ``calibrated`` is how many representatives were read from a ``.sample`` container the loop stage wrote,
    the rest having been split where they stood -- which is what says whether the run had loops to store.
    """

    manifest: Path
    instruments: tuple[Path, ...]
    auditions: int
    bands: tuple[CarrierLayerRecord, ...]
    recordings: int
    calibrated: int


def _calibrated_source(recording: StageRecording, document: SampleDocument) -> CarrierSource:
    """One recording as the loop stage left it: split into level and carrier, with the loops it offers."""
    return CarrierSource(
        key=recording.key,
        root_pitch=document.root_pitch,
        sample_rate=document.sample_rate,
        decomposition=sample_decomposition(document),
        loops=sample_loops(document),
        loop_index=faithful_offer(document),
        weight=recording.weight,
    )


def _split_source(recording: StageRecording, encode: EncodeConfig) -> CarrierSource:
    """One recording split where it stands, for a stage that settled no loop over it.

    The split is taken through the same weighting the loop stage takes it through, so a recording read this
    way lands on the terms every calibrated one did, and is stored as the span it plays.
    """
    reading = level_reading(recording.sample_rate, encode.envelope, midi_to_freq(recording.key.pitch))
    return CarrierSource(
        key=recording.key,
        root_pitch=recording.key.pitch,
        sample_rate=recording.sample_rate,
        decomposition=decompose(recording.signal, reading),
        loops=NO_LOOPS,
        loop_index=None,
        weight=recording.weight,
    )


def carrier_source(recording: StageRecording, encode: EncodeConfig) -> tuple[CarrierSource, bool]:
    """``recording`` as a carrier source, read from the container beside it wherever one was written.

    The loop stage writes a ``.sample`` next to every WAV it lands, holding the split and every loop the
    recording offers, so reading it back is what hands this route both halves at once. A stage that wrote
    none leaves the recording to be split where it stands, which costs the loops and nothing else.
    """
    container = recording.file.with_suffix(SAMPLE_EXTENSION)
    if container.is_file():
        return _calibrated_source(recording, read_sample(container)), True

    return _split_source(recording, encode), False


def velocity_layers(material: Sequence[NoteEvent], layers: int) -> VelocityLayers:
    """The velocity axis cut into ``layers`` bands holding equal shares of the material's playing time.

    A tracker tells a dynamic apart by the instrument a note plays through, so the axis is split before any
    of the space is cut and each band is grouped on its own.
    """
    return VelocityLayers(velocity_cells(material, layers).cells)


def _band_positions(corpus: StageCorpus, band: VelocityBand) -> tuple[int, ...]:
    """Where in the corpus the recordings this band answers for stand."""
    return tuple(index for index, recording in enumerate(corpus.recordings) if band.covers(recording.key.velocity))


def _band_readings(described: DescribedCorpus, positions: Sequence[int]) -> MemberReadings:
    """What the band's own recordings carry beyond their place in the space it makes."""
    chosen = list(positions)
    readings = described.readings
    return MemberReadings(weights=readings.weights[chosen], durations_s=readings.durations_s[chosen])


@dataclass(frozen=True)
class BandCut:
    """One velocity band cut into groups: which take stands for each, and how many each gathered."""

    band: VelocityBand
    representatives: tuple[int, ...]
    members: tuple[int, ...]


def cut_band(described: DescribedCorpus, band: VelocityBand, *, groups: int, config: ClusterConfig) -> BandCut:
    """The takes standing for the groups one band's recordings fall into, named in the corpus's own order.

    A band holding a single recording is stood for by that recording, there being nothing to tell apart, and
    one the corpus left no recording in stands for nothing -- which is what a dynamic the material plays but
    whose only takes carried no signal leaves behind.
    """
    positions = _band_positions(described.corpus, band)
    if len(positions) <= _ALONE:
        return BandCut(band=band, representatives=positions, members=(len(positions),) * len(positions))

    coordinates = sample_space([described.descriptors[index] for index in positions], config.space).coordinates
    found = grouping(
        coordinates,
        partition(coordinates, groups=min(groups, len(positions)), config=config.partition).labels,
        _band_readings(described, positions),
        config=config.partition,
    )
    return BandCut(
        band=band,
        representatives=tuple(positions[group.representative] for group in found),
        members=tuple(group.size for group in found),
    )


@dataclass(frozen=True)
class WrittenBand:
    """One velocity band written as an instrument: what landed, and what the choice came to."""

    record: CarrierLayerRecord
    path: Path
    auditions: int
    calibrated: int


def _carrier_settings(settings: ClusterSettings) -> CarrierSettings:
    """What writing one band's takes as carrier samples is carried out with, off the run's own settings."""
    stored = settings.cluster.instrument
    return CarrierSettings(
        target=settings.target,
        grid=envelope_grid(settings.target, tempo=settings.tempo, release_s=settings.release_s),
        params=EncodingParams(target_rate=stored.rate, depth_bits=stored.depth),
        config=settings.encode,
        seed=settings.seed,
    )


def _write_instrument(built: CarrierInstrument, band: VelocityBand, samples_dir: Path, target: ExportTarget) -> Path:
    """One band's instrument written as the standalone file its format loads."""
    written = target.instrument_file(built.unit)
    directory = instrument_files_dir(samples_dir, written.extension)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{band.label}{written.extension}"
    written.save(path)
    return path


def _write_auditions(
    built: CarrierInstrument,
    band: VelocityBand,
    out_dir: Path,
    *,
    tempo: int,
    step: ProgressStep,
) -> int:
    """Each stored waveform played out under the curve above it, so the pair is heard rather than read.

    A folder per band holds one file per take, named by the key it stands for, which is what puts an
    instrument's own dynamics side by side. ``step`` is taken as each file lands, so the run counts off the
    waveforms it writes rather than the bands they are gathered into.
    """
    folder = out_dir / AUDITIONS_DIR / band.label
    folder.mkdir(parents=True, exist_ok=True)
    envelope = built.unit.instrument.volume_envelope
    for carrier in built.stored:
        write_wav(
            folder / f"{carrier.source.key.label}.wav",
            carrier_audition(carrier, envelope, tempo=tempo),
            carrier.stored.sample_rate,
        )
        step()

    return len(built.stored)


def write_band(
    described: DescribedCorpus,
    cut: BandCut,
    out_dir: Path,
    settings: ClusterSettings,
    *,
    step: ProgressStep,
) -> WrittenBand:
    """One velocity band's chosen takes written as the single instrument that band's dynamics play."""
    band = cut.band
    read = [carrier_source(described.corpus.recordings[index], settings.encode) for index in cut.representatives]
    built = carrier_instrument(
        [source for source, _ in read],
        instrument_id=settings.instrument_id,
        name=f"{settings.instrument_id} {band.label}",
        settings=_carrier_settings(settings),
    )
    path = _write_instrument(built, band, out_dir / settings.instrument_id, settings.target)
    return WrittenBand(
        record=layer_record(band, built, members=cut.members, files={path.suffix: str(path.relative_to(out_dir))}),
        path=path,
        auditions=_write_auditions(built, band, out_dir, tempo=settings.tempo, step=step),
        calibrated=sum(calibrated for _, calibrated in read),
    )


def _band_cuts(described: DescribedCorpus, layers: VelocityLayers, settings: ClusterSettings) -> list[BandCut]:
    """Every band the corpus left a recording in, cut into the groups one waveform stands for each of.

    A band the corpus holds nothing for is passed over rather than written empty, since an instrument with
    no waveform sounds nothing and the dynamics it answers are already covered by the bands beside it.
    """
    bands = settings.progress.track(layers.bands, label=_CUT_LABEL, total=layers.count)
    cuts = [
        cut_band(described, band, groups=settings.cluster.partition.groups, config=settings.cluster) for band in bands
    ]
    return [cut for cut in cuts if cut.representatives]


def _written_bands(
    described: DescribedCorpus,
    cuts: Sequence[BandCut],
    out_dir: Path,
    settings: ClusterSettings,
) -> list[WrittenBand]:
    """Each cut band written as its own instrument, counted off in the waveforms the run stores.

    The bar is bounded by the takes every band chose rather than by the bands themselves, so a run asked
    for more groups states a longer stretch of work and reaches its bound as the last audition lands.
    """
    with counting(settings.progress, label=_WRITE_LABEL, total=sum(len(cut.representatives) for cut in cuts)) as step:
        return [write_band(described, cut, out_dir, settings, step=step) for cut in cuts]


def write_clustered(root: Path, out_dir: Path, settings: ClusterSettings) -> ClusteredArtifacts:
    """Write one clustered set of recordings as the carrier instruments a tracker loads.

    The velocity axis is cut first, since a keymap names no dynamic and one band is therefore one
    instrument; each band's own recordings are then placed as a space and cut into groups, and the take
    standing for each group becomes a stored waveform. What every waveform holds is the timbre alone --
    the level it moves through is handed to the one volume envelope its band carries
    (:func:`~optisample.carrier.instrument.carrier_instrument`), which is what lets a shallow depth spend
    its whole grid on sound.

    Raises:
        ValueError: when the stage holds what an allocation stored, which names no material to cut on.
    """
    corpus = stage_recordings(root, settings.stage, settings.reading)
    described = describe_corpus(
        corpus,
        features=settings.features,
        config=settings.cluster.descriptor,
        workers=settings.workers,
        progress=settings.progress,
    )
    layers = velocity_layers(stage_material(root, settings.stage, settings.reading), settings.cluster.instrument.layers)
    written = _written_bands(described, _band_cuts(described, layers, settings), out_dir, settings)
    manifest = out_dir / f"{settings.instrument_id}{MANIFEST_SUFFIX}"
    write_json(
        manifest,
        clustered_document(
            settings.instrument_id,
            stage=settings.stage.value,
            tempo=settings.tempo,
            groups=settings.cluster.partition.groups,
            layers=[band.record for band in written],
        ),
    )
    return ClusteredArtifacts(
        manifest=manifest,
        instruments=tuple(band.path for band in written),
        auditions=sum(band.auditions for band in written),
        bands=tuple(band.record for band in written),
        recordings=corpus.size,
        calibrated=sum(band.calibrated for band in written),
    )
