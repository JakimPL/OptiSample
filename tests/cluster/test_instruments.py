from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod import BitDepth

from optisample.artifacts.documents.clustered import ClusteredDocument
from optisample.artifacts.documents.loops import LevelRecord, LoopQualityRecord, SettledLoopRecord
from optisample.artifacts.documents.sample import SampleDocument
from optisample.artifacts.serialize import write_msgpack
from optisample.cluster.corpus import describe_corpus
from optisample.cluster.instruments import (
    _CUT_LABEL,
    _WRITE_LABEL,
    AUDITIONS_DIR,
    MANIFEST_SUFFIX,
    ClusterSettings,
    carrier_source,
    cut_band,
    velocity_layers,
    write_clustered,
)
from optisample.cluster.stages import (
    ReadingSettings,
    RecordingCorpus,
    RecordingSource,
    Stage,
    StageRecording,
    source_recordings,
)
from optisample.config import OptiConfig
from optisample.io.audio import write_wav
from optisample.io.note_extractor import NoteRecord, dump_notes
from optisample.io.tracker.target import export_target
from optisample.keys import SampleKey
from optisample.model import NoteEvent
from optisample.optimize.layers.bands import VelocityBand
from optisample.progress import NO_PROGRESS, ProgressSink

SR = 44_100
PITCHES = (55, 58, 61, 64, 67, 70)
VELOCITIES = (30, 100)
INSTRUMENT = "tiny"
_SOUNDS_S = 0.9
_GROUPS = 2
_LAYERS = 2


@pytest.fixture
def run_root(tmp_path: Path, piano_note: Callable[..., NDArray[np.float64]]) -> Path:
    """A run whose first stage holds a small grid of notes, over two dynamics and a spread of keys."""
    stage_dir = tmp_path / "0_subset"
    samples_dir = stage_dir / INSTRUMENT
    samples_dir.mkdir(parents=True)
    records = []
    index = 0
    for pitch in PITCHES:
        for velocity in VELOCITIES:
            signal = piano_note(pitch, velocity, _SOUNDS_S, seed=pitch * velocity)
            write_wav(samples_dir / f"{index:04d}_p{pitch}_v{velocity}.wav", signal, SR)
            records.append(NoteRecord(index=index, pitch=pitch, velocity=velocity, duration_s=_SOUNDS_S))
            index += 1

    dump_notes(records, stage_dir / f"{INSTRUMENT}{'.notes.json'}")
    return tmp_path


@pytest.fixture
def cluster_settings(config: OptiConfig, run_root: Path) -> Callable[..., ClusterSettings]:
    """Factory: what a clustered run over the tiny stage is carried out with."""

    def _settings(
        *,
        groups: int = _GROUPS,
        layers: int = _LAYERS,
        depth: BitDepth = BitDepth.EIGHT,
        progress: ProgressSink = NO_PROGRESS,
    ) -> ClusterSettings:
        cluster = config.cluster.model_dump()
        cluster["partition"]["groups"] = groups
        cluster["partition"]["max_groups"] = max(cluster["partition"]["max_groups"], groups)
        cluster["instrument"]["layers"] = layers
        cluster["instrument"]["depth"] = depth
        cluster["instrument"]["rate"] = 22_050
        return ClusterSettings(
            source=RecordingSource(root=run_root, instrument_id=INSTRUMENT, stage=Stage.SUBSET),
            reading=ReadingSettings(
                strategy="grouped",
                dedupe=config.reduce.dedupe,
                trim=config.reduce.trim,
                keep_tail=False,
                progress=progress,
            ),
            cluster=type(config.cluster).model_validate(cluster),
            features=config.loop.features,
            encode=config.encode,
            instruments=config.export.instruments,
            target=export_target(config.export.tracker),
            release_s=config.export.envelope.release_s,
            tempo_bpm=config.export.playback.tempo,
            seed=137,
            workers=1,
            progress=progress,
        )

    return _settings


def _from_subset(run_root: Path) -> RecordingSource:
    """The set the tiny stage's takes came from, which each of them names."""
    return RecordingSource(root=run_root, instrument_id=INSTRUMENT, stage=Stage.SUBSET)


def _quality(spectral_distance: float) -> LoopQualityRecord:
    return LoopQualityRecord(seam_step=1.0, level_drift_db=0.0, spectral_distance=spectral_distance)


def _offer(start: int, spectral_distance: float) -> SettledLoopRecord:
    return SettledLoopRecord(
        start=start,
        end=start + 1000,
        start_s=start / SR,
        end_s=(start + 1000) / SR,
        quality=_quality(spectral_distance),
        level=LevelRecord(seconds=[0.0, 1.0], values_db=[0.0, -6.0]),
    )


# --- how the dynamics are cut -------------------------------------------------------------------------------


def test_the_velocity_axis_is_tiled_by_the_bands_it_is_cut_into() -> None:
    """Every dynamic resolves to exactly one band, so a note always finds the instrument meant for it."""
    material = [NoteEvent(pitch=60, velocity=velocity, duration_s=1.0) for velocity in (10, 40, 80, 120)]
    layers = velocity_layers(material, 2)

    assert layers.count == 2
    assert layers.bands[0].lowest == 0
    assert layers.bands[-1].highest == 127
    for lower, upper in zip(layers.bands, layers.bands[1:]):
        assert upper.lowest == lower.highest + 1


# --- what the run says about itself while it runs -------------------------------------------------------------


class _Counted:
    """A sink recording what each bar was bounded by, and what had landed by the time it counted a step.

    ``watch`` is read as every step is taken, so a test states not only how far a bar reached but whether
    the work it stands for had actually happened by then.
    """

    def __init__(self, watch: Callable[[], int]) -> None:
        self.totals: dict[str, int] = {}
        self.watched: dict[str, list[int]] = {}
        self.watch = watch

    def track[ItemT](self, items: Iterable[ItemT], *, label: str, total: int) -> Iterator[ItemT]:
        """Yield every item, recording the bound and what stood finished at each step."""
        self.totals[label] = total
        seen = self.watched.setdefault(label, [])
        for item in items:
            yield item
            seen.append(self.watch())


def test_the_writing_bar_is_bounded_by_the_waveforms_stored_rather_than_the_bands(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """A run asked for more groups does more writing, so the bound follows the takes rather than the bands."""
    out_dir = tmp_path / "clustered"
    counted = _Counted(lambda: 0)

    written = write_clustered(out_dir, cluster_settings(progress=counted))

    assert counted.totals[_CUT_LABEL] == _LAYERS
    assert counted.totals[_WRITE_LABEL] == written.auditions == _LAYERS * _GROUPS


def test_the_writing_bar_counts_a_step_only_once_a_waveform_has_landed(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """A bar reaching its bound before the files exist reports nothing, so each step follows its audition."""
    out_dir = tmp_path / "clustered"
    auditions = out_dir / AUDITIONS_DIR
    counted = _Counted(lambda: len(list(auditions.rglob("*.wav"))))

    write_clustered(out_dir, cluster_settings(progress=counted))

    landed = counted.watched[_WRITE_LABEL]
    assert landed[0] >= 1
    assert landed == sorted(landed)


# --- what a run writes ---------------------------------------------------------------------------------------


def test_a_run_writes_one_instrument_per_velocity_band(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """A keymap names no dynamic, so each band is written as an instrument of its own."""
    out_dir = tmp_path / "clustered"
    written = write_clustered(out_dir, cluster_settings())

    assert len(written.instruments) == _LAYERS
    assert len(written.bands) == _LAYERS
    for path in written.instruments:
        assert path.is_file()
        assert path.read_bytes()[:4] == b"IMPI"


def test_each_band_stores_one_waveform_per_group(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """The take standing for each group becomes one stored sample, so the count follows the cut."""
    written = write_clustered(tmp_path / "clustered", cluster_settings(groups=_GROUPS))

    for band in written.bands:
        assert len(band.samples) == _GROUPS
        assert sum(sample.members for sample in band.samples) == len(PITCHES)


def test_the_manifest_states_what_the_run_chose(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """A tracker file carries no clock and no dynamics, so the manifest is where a reader finds them."""
    out_dir = tmp_path / "clustered"
    written = write_clustered(out_dir, cluster_settings())
    document = ClusteredDocument.model_validate(json.loads(written.manifest.read_text()))

    assert written.manifest.name == f"{INSTRUMENT}{MANIFEST_SUFFIX}"
    assert document.instrument_id == INSTRUMENT
    assert document.stage == Stage.SUBSET.value
    assert document.tempo > 0
    assert [band.band for band in document.bands] == [band.band for band in written.bands]


def test_every_stored_waveform_is_auditioned(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """The pair is meant to be heard, so each take is played out under the curve above it."""
    out_dir = tmp_path / "clustered"
    written = write_clustered(out_dir, cluster_settings())

    assert written.auditions == sum(len(band.samples) for band in written.bands)
    assert len(list((out_dir / AUDITIONS_DIR).rglob("*.wav"))) == written.auditions


def test_a_stage_that_wrote_no_container_splits_its_recordings_where_they_stand(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """The first stage settles no loop, so its takes are split here and stored as the span they play."""
    written = write_clustered(tmp_path / "clustered", cluster_settings())

    assert written.recordings == len(PITCHES) * len(VELOCITIES)
    assert written.calibrated == 0
    assert all(not sample.looped for band in written.bands for sample in band.samples)


def test_a_shallower_depth_stores_fewer_bytes(
    run_root: Path,
    tmp_path: Path,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """Depth is the axis the carrier buys its bytes back on, and the run states what it spent."""
    deep = write_clustered(tmp_path / "deep", cluster_settings(depth=BitDepth.SIXTEEN))
    shallow = write_clustered(tmp_path / "shallow", cluster_settings(depth=BitDepth.EIGHT))

    assert sum(band.stored_bytes for band in shallow.bands) * 2 == sum(band.stored_bytes for band in deep.bands)


# --- reading a recording back through the container the loop stage wrote --------------------------------------


def _write_container(samples_dir: Path, stem: str, pitch: int, sample_rate: int, frames: int) -> None:
    """A ``.sample`` beside the WAV of the same stem, holding a split and one loop the stage settled."""
    level = np.linspace(1.0, 0.25, frames, dtype="<f4")
    carrier = np.ones(frames, dtype="<f4")
    document = SampleDocument(
        key=f"p{pitch:03d}_v100",
        root_pitch=pitch,
        sample_rate=sample_rate,
        frames=frames,
        carrier=carrier.tobytes(),
        level=level.tobytes(),
        loops=[_offer(frames // 4, 5.0), _offer(frames // 3, 2.0)],
        provenance={"stage": "loop", "instrument_id": INSTRUMENT, "index": 0, "source": f"{stem}.wav"},
    )
    write_msgpack(samples_dir / f"{stem}.sample", document)


def test_a_recording_with_a_container_beside_it_is_read_through_it(
    run_root: Path,
    config: OptiConfig,
) -> None:
    """The loop stage writes the split and the loops together, so reading the container hands over both."""
    samples_dir = run_root / "0_subset" / INSTRUMENT
    wav = sorted(samples_dir.glob("*.wav"))[0]
    frames = 4096
    _write_container(samples_dir, wav.stem, PITCHES[0], SR, frames)

    recording = StageRecording(
        source=_from_subset(run_root),
        file=wav,
        key=SampleKey(pitch=PITCHES[0], velocity=100),
        signal=np.zeros(frames, dtype=np.float64),
        sample_rate=SR,
        weight=1.0,
    )
    source, calibrated = carrier_source(recording, config.encode)

    assert calibrated
    assert source.frames == frames
    assert source.loop_index == 1  # the offer standing closest to the material it replaces
    assert len(source.loops) == 2


def test_a_recording_without_a_container_is_split_where_it_stands(
    run_root: Path,
    config: OptiConfig,
    piano_note: Callable[..., NDArray[np.float64]],
) -> None:
    """A stage that settled no loop leaves the split to be taken here, and the take stores what it plays."""
    wav = sorted((run_root / "0_subset" / INSTRUMENT).glob("*.wav"))[0]
    signal = piano_note(PITCHES[0], 100, _SOUNDS_S, seed=1)
    recording = StageRecording(
        source=_from_subset(run_root),
        file=wav,
        key=SampleKey(pitch=PITCHES[0], velocity=100),
        signal=signal,
        sample_rate=SR,
        weight=1.0,
    )

    source, calibrated = carrier_source(recording, config.encode)

    assert not calibrated
    assert source.loops == ()
    assert source.loop_index is None


def test_a_band_holding_one_recording_is_stood_for_by_it(
    run_root: Path,
    config: OptiConfig,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """There is nothing to tell apart inside a band of one, so that take stands for the whole of it."""
    settings = cluster_settings()
    corpus = source_recordings(settings.source, settings.reading)
    alone = RecordingCorpus(sources=corpus.sources, recordings=corpus.recordings[:1])
    described = describe_corpus(
        alone,
        features=config.loop.features,
        config=settings.cluster.descriptor,
        workers=1,
        progress=NO_PROGRESS,
    )
    band = VelocityBand(0, 127)

    cut = cut_band(described, band, groups=settings.cluster.partition.groups, config=settings.cluster)

    assert cut.representatives == (0,)
    assert cut.members == (1,)


def test_a_band_the_corpus_left_empty_is_passed_over(
    run_root: Path,
    config: OptiConfig,
    cluster_settings: Callable[..., ClusterSettings],
) -> None:
    """An instrument with no waveform sounds nothing, so a band holding no recording is not written."""
    settings = cluster_settings()
    corpus = source_recordings(settings.source, settings.reading)
    described = describe_corpus(
        corpus,
        features=config.loop.features,
        config=settings.cluster.descriptor,
        workers=1,
        progress=NO_PROGRESS,
    )
    unplayed = VelocityBand(VELOCITIES[-1] + 1, 127)

    cut = cut_band(described, unplayed, groups=settings.cluster.partition.groups, config=settings.cluster)

    assert cut.representatives == ()
    assert cut.members == ()
