from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from optisample.io.note_extractor import NOTES_SUFFIX


def instrument_name(source: Path) -> str:
    """The instrument a source path names: a directory by its own name, a manifest by its base name.

    One spelling covers both shapes, so the id a run files its artifacts under reads the same whether it
    was handed a ``.notes.json`` or the directory of recordings the takes sit in.
    """
    if source.is_dir():
        return source.name

    name = source.name
    if name.endswith(NOTES_SUFFIX):
        return name[: -len(NOTES_SUFFIX)]

    return source.stem


@dataclass(frozen=True)
class SourceDataset:
    """Where a run reads its recordings: a manifest to join, or a directory of takes read as it stands.

    ``samples_dir`` names where a manifest's recordings live, for the datasets keeping them apart from
    the manifest itself. Leaving it out reads them from the manifest's own sibling ``<name>/``, which is
    the layout every dataset this project writes uses.
    """

    path: Path
    samples_dir: Path | None

    @property
    def is_directory(self) -> bool:
        """Whether the source is a directory of recordings, which is read as the grid its names spell."""
        return self.path.is_dir()

    @property
    def recordings_dir(self) -> Path:
        """The directory holding this dataset's WAVs, whichever of the two shapes the source takes.

        A directory of recordings holds its own, so it answers with itself and ``samples_dir`` speaks
        for the manifest sources it was written for.
        """
        if self.is_directory:
            return self.path

        if self.samples_dir is not None:
            return self.samples_dir

        return self.path.parent / instrument_name(self.path)


@dataclass(frozen=True)
class SubsetDataset:
    """Where a subset dataset landed and how much of its source it holds.

    ``source`` reads the slice the way its source was read: a slice of a manifest dataset is a manifest
    dataset, and a slice of a directory of recordings is a directory of recordings.

    ``pitches`` and ``velocities`` are the closed ranges the kept notes span, which is what says
    whether a subset small enough to iterate on still exercises the whole instrument.

    ``brief_notes`` is how many of the source's notes sounded for less than the slice admits, so a source
    recorded largely in fragments states that at the way in rather than through the loops a later stage
    fails to settle.
    """

    source: SourceDataset
    kept_notes: int
    source_notes: int
    recordings: int
    brief_notes: int
    pitches: tuple[int, int]
    velocities: tuple[int, int]
