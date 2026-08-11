from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from optisample.io.dataset import SourceDataset
from optisample.io.note_extractor import NO_TEMPO, index_of_wav, read_manifest
from optisample.io.sample_dir import WAV_GLOB, read_takes


@dataclass(frozen=True)
class RecordingSource:
    """One recording of a written dataset: the file holding it, and the pitch it was played at.

    The pitch is what the instrument written from this recording is rooted at, so a key sounding it plays
    the note the recording holds and every other key transposes from there.
    """

    file: Path
    root_pitch: int

    @property
    def name(self) -> str:
        """What an instrument written from this recording is named, which is the stem it shares with its WAV."""
        return self.file.stem


@dataclass(frozen=True)
class DatasetRecordings:
    """Every recording one written dataset holds, beside the clock the material was played at.

    ``tempo_bpm`` is what the dataset states for itself, which is the clock the volume envelopes written
    from these recordings are counted in ticks of. A dataset recorded away from a clock leaves it open and
    the caller supplies the one its own export plays at.
    """

    sources: tuple[RecordingSource, ...]
    tempo_bpm: float | None


def _manifest_recordings(notes_json: Path, samples_dir: Path) -> DatasetRecordings:
    """The recordings a NoteExtractor dataset holds, each joined to the note naming the pitch it plays.

    Every note reaching one recording is a note at that recording's own pitch
    (:func:`~optisample.artifacts.dataset.note_records`), so the manifest states one pitch per render
    index and a WAV carries that index in the leading token of its name.

    Raises:
        ValueError: when a WAV of ``samples_dir`` carries a render index the manifest names no note at.
    """
    parsed = read_manifest(notes_json)
    pitches = {note.render.index: note.pitch for note in parsed.notes}
    sources: list[RecordingSource] = []
    for wav in sorted(samples_dir.glob(WAV_GLOB)):
        index = index_of_wav(wav)
        if index not in pitches:
            raise ValueError(
                f"recording {wav.name!r} carries render index {index}, which {notes_json} names no note at"
            )

        sources.append(RecordingSource(file=wav, root_pitch=pitches[index]))

    return DatasetRecordings(sources=tuple(sources), tempo_bpm=parsed.tempo_bpm)


def _directory_recordings(samples_dir: Path) -> DatasetRecordings:
    """The recordings a directory of takes holds, each at the pitch its own filename spells."""
    takes = read_takes(samples_dir)
    return DatasetRecordings(
        sources=tuple(RecordingSource(file=take.file, root_pitch=take.pitch) for take in takes),
        tempo_bpm=NO_TEMPO,
    )


def dataset_recordings(source: SourceDataset) -> DatasetRecordings:
    """Every recording ``source`` holds, read the way that shape of dataset states what it holds.

    A manifest dataset joins each WAV to the note whose render index it carries, and a directory of takes
    reads the pitch each filename spells, so one call answers for either shape and the instruments written
    from a dataset stand for exactly the recordings sitting beside them.
    """
    if source.is_directory:
        return _directory_recordings(source.path)

    return _manifest_recordings(source.path, source.recordings_dir)
