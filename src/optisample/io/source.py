from __future__ import annotations

from pathlib import Path

from optisample.config.subset import IntakeConfig
from optisample.io.dataset import SourceDataset, SubsetDataset
from optisample.io.note_extractor import IngestSettings, load_notes
from optisample.io.sample_dir import load_sample_dir, write_sample_dir_subset
from optisample.io.subset import write_subset
from optisample.model import Manifest


def load_source(source: SourceDataset, settings: IngestSettings) -> Manifest:
    """The manifest a run reads off its source, whichever of the two shapes it was handed.

    A ``.notes.json`` is joined to its recordings and carries the performance it was extracted from; a
    directory of recordings is read as the grid its filenames spell, each take standing for one note.
    Both answer with the same single-instrument manifest, so every stage past the ingest reads one shape.
    """
    if source.is_directory:
        return load_sample_dir(source.path, settings)

    return load_notes(source.path, source.recordings_dir, settings)


def write_source_subset(
    source: SourceDataset,
    out_dir: Path,
    *,
    instrument_id: str,
    fraction: float,
    intake: IntakeConfig,
) -> SubsetDataset:
    """Write the ``fraction`` of ``source`` spanning its pitch and velocity ranges, in the shape it came in.

    A slice of a manifest dataset is a manifest dataset and a slice of a directory of recordings is a
    directory of recordings, so the stage after it reads the slice the way it would have read the whole.
    Either shape holds its notes to the length ``intake`` admits and lands past the band it passes, which
    are the two treatments the way in gives a source.
    """
    if source.is_directory:
        return write_sample_dir_subset(
            source.path, out_dir, instrument_id=instrument_id, fraction=fraction, intake=intake
        )

    return write_subset(source, out_dir, instrument_id=instrument_id, fraction=fraction, intake=intake)
