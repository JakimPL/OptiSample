from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from optisample.io.note_extractor import (
    NOTES_SUFFIX,
    ManifestNote,
    NotesManifest,
    index_of_wav,
)

_NOTES_FIELD: Final = "notes"
_CONFIG_FIELD: Final = "config"
_NOTE_COUNT_FIELD: Final = "note_count"
_WHOLE: Final = 1.0


@dataclass(frozen=True)
class SubsetDataset:
    """Where a subset dataset landed and how much of its source it holds.

    ``pitches`` and ``velocities`` are the closed ranges the kept notes span, which is what says
    whether a subset small enough to iterate on still exercises the whole instrument.
    """

    notes_json: Path
    samples_dir: Path
    kept_notes: int
    source_notes: int
    recordings: int
    pitches: tuple[int, int]
    velocities: tuple[int, int]


def even_ranks(count: int, picks: int) -> tuple[int, ...]:
    """``picks`` positions spread evenly over ``range(count)``, reaching both ends.

    A single pick lands in the middle, the position most typical of the axis it is drawn from; two or
    more include the extremes, so the picks span the full range. Asking for at least as many picks as
    there are positions keeps every one of them.
    """
    if picks <= 0 or count <= 0:
        return ()

    if picks >= count:
        return tuple(range(count))

    if picks == 1:
        return ((count - 1) // 2,)

    return tuple(round(pick * (count - 1) / (picks - 1)) for pick in range(picks))


def _allotments(sizes: Sequence[int], keep: int) -> tuple[int, ...]:
    """How many notes each pitch contributes so the subset holds ``keep`` of them.

    Notes go round the pitches in passes: every pitch is represented before any takes a second, and a
    second before any takes a third, so a subset spreads across the keyboard rather than deepening at a
    handful of pitches. Within a pass the pitches carrying the most material come first, so a pass that
    runs out leaves the extra notes where there is most to hear, and a pitch that has given everything
    it holds steps aside for the rest. A subset smaller than the number of pitches holds the pitches
    spread evenly across the keyboard.
    """
    if keep <= len(sizes):
        selected = set(even_ranks(len(sizes), keep))
        return tuple(int(index in selected) for index in range(len(sizes)))

    allotted = [1] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda index: (-sizes[index], index))
    given = len(sizes)
    while given < keep and any(allotted[index] < sizes[index] for index in order):
        for index in order:
            if given == keep:
                break

            if allotted[index] < sizes[index]:
                allotted[index] += 1
                given += 1

    return tuple(allotted)


def _velocity_ordered_pitches(notes: Sequence[ManifestNote]) -> list[list[int]]:
    """The positions of every note, grouped by ascending pitch and sorted by velocity within a pitch.

    Both axes arrive sorted, so a share taken at even ranks of a group spans that pitch's dynamics and
    the groups themselves span the keyboard.
    """
    by_pitch: dict[int, list[int]] = {}
    for position, note in enumerate(notes):
        by_pitch.setdefault(note.pitch, []).append(position)

    return [
        sorted(positions, key=lambda position: notes[position].velocity) for _, positions in sorted(by_pitch.items())
    ]


def select_positions(notes: Sequence[ManifestNote], fraction: float) -> tuple[int, ...]:
    """Positions of the notes a subset holding ``fraction`` of ``notes`` keeps, in source order.

    Notes are grouped by pitch, each pitch is allotted a share of the subset, and the notes a pitch
    contributes are those its velocities spread evenly over. The subset therefore covers the pitch
    range the source plays and, within each pitch, the dynamics it was played across -- which is what
    makes a small slice representative enough to predict how the whole dataset behaves.

    Raises:
        ValueError: if ``fraction`` falls outside ``(0, 1]``.
    """
    if not 0.0 < fraction <= _WHOLE:
        raise ValueError(f"subset fraction must fall in (0, 1], got {fraction}")

    groups = _velocity_ordered_pitches(notes)
    keep = max(1, round(fraction * len(notes)))
    allotted = _allotments([len(group) for group in groups], keep)
    return tuple(
        sorted(group[rank] for group, picks in zip(groups, allotted) for rank in even_ranks(len(group), picks))
    )


def _copy_recordings(indices: frozenset[int], samples_dir: Path, out_dir: Path) -> int:
    """Copy every recording the kept notes join to into ``out_dir`` and state how many landed.

    A filename carries the render index a later ingest joins on, so copying it under its own name
    leaves the subset resolving its recordings exactly as the source dataset does.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for wav in sorted(samples_dir.glob("*.wav")):
        if index_of_wav(wav) in indices:
            shutil.copy2(wav, out_dir / wav.name)
            copied += 1

    return copied


def _subset_manifest(raw: dict[str, Any], entries: list[Any]) -> dict[str, Any]:
    """The source manifest carrying only ``entries``, with its declared note count following suit."""
    config: dict[str, Any] = {**raw.get(_CONFIG_FIELD, {}), _NOTE_COUNT_FIELD: len(entries)}
    return {**raw, _CONFIG_FIELD: config, _NOTES_FIELD: entries}


def write_subset(
    notes_json: Path | str,
    samples_dir: Path | str,
    out_dir: Path | str,
    *,
    instrument_id: str,
    fraction: float,
) -> SubsetDataset:
    """Write the ``fraction`` of a NoteExtractor dataset that spans its pitch and velocity ranges.

    The output root is itself a NoteExtractor dataset -- ``<instrument_id>.notes.json`` beside its
    ``<instrument_id>/`` recordings -- so a later ingest resolves it by default and every stage reads it
    the way it reads the source. Each kept note is written exactly as the source states it, so the
    subset measures the same material, at the same lengths, that the whole dataset would.
    """
    raw: dict[str, Any] = json.loads(Path(notes_json).read_text(encoding="utf-8"))
    parsed = NotesManifest.model_validate(raw)
    positions = select_positions(parsed.notes, fraction)
    kept = [parsed.notes[position] for position in positions]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{instrument_id}{NOTES_SUFFIX}"
    entries: list[Any] = raw[_NOTES_FIELD]
    target.write_text(
        json.dumps(_subset_manifest(raw, [entries[position] for position in positions]), indent=2) + "\n",
        encoding="utf-8",
    )
    recordings = _copy_recordings(
        frozenset(note.render.index for note in kept), Path(samples_dir), out_dir / instrument_id
    )
    pitches = [note.pitch for note in kept]
    velocities = [note.velocity for note in kept]
    return SubsetDataset(
        notes_json=target,
        samples_dir=out_dir / instrument_id,
        kept_notes=len(kept),
        source_notes=len(parsed.notes),
        recordings=recordings,
        pitches=(min(pitches), max(pitches)),
        velocities=(min(velocities), max(velocities)),
    )
