from __future__ import annotations

import json
import shutil
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from optisample.config.dynamic_axis import DynamicAxisConfig
from optisample.config.subset import IntakeConfig
from optisample.config.subsonic import SubsonicConfig
from optisample.dsp.subsonic import remove_subsonic
from optisample.dynamic_axis import dynamic_of
from optisample.io.audio import read_wav, write_wav
from optisample.io.dataset import SourceDataset, SubsetDataset
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
_AT_LEAST_ONE: Final = 1  # notes a slice holds however small the share asked of it
_NOTHING_HELD_OUT: Final = 0  # notes a floor removes where a caller names its recordings itself


class Played(Protocol):
    """A recorded note as the selection reads it: the two axes a slice has to span, and how long it sounds.

    Stated as a protocol so the same rule slices both shapes of dataset -- the notes a manifest lists
    and the takes a directory of recordings names -- from one implementation.

    The dynamic axis is read off ``velocity`` and ``cc_averages`` together
    (:func:`~optisample.dynamic_axis.dynamic_of`), so both readings arrive here and the configured one
    decides which of them a slice spreads along.
    """

    @property
    def pitch(self) -> int: ...

    @property
    def velocity(self) -> int: ...

    @property
    def cc_averages(self) -> Mapping[int, float]: ...

    @property
    def duration_s(self) -> float: ...


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


def _dynamic_ordered_pitches(notes: Sequence[Played], dynamic_axis: DynamicAxisConfig) -> list[list[int]]:
    """The positions of every note, grouped by ascending pitch and sorted by dynamic within a pitch.

    Both axes arrive sorted, so a share taken at even ranks of a group spans that pitch's dynamics and
    the groups themselves span the keyboard. The dynamic is the reading ``dynamic_axis`` names, which is
    what makes the spread cover the range an instrument was actually played across.
    """
    by_pitch: dict[int, list[int]] = {}
    for position, note in enumerate(notes):
        by_pitch.setdefault(note.pitch, []).append(position)

    return [
        sorted(positions, key=lambda position: dynamic_of(notes[position], dynamic_axis))
        for _, positions in sorted(by_pitch.items())
    ]


def kept_count(notes: int, fraction: float) -> int:
    """How many of ``notes`` a subset holding ``fraction`` of them keeps, which is one at the least.

    Raises:
        ValueError: if ``fraction`` falls outside ``(0, 1]``.
    """
    if not 0.0 < fraction <= _WHOLE:
        raise ValueError(f"subset fraction must fall in (0, 1], got {fraction}")

    return max(_AT_LEAST_ONE, round(fraction * notes))


def select_count(notes: Sequence[Played], keep: int, dynamic_axis: DynamicAxisConfig) -> tuple[int, ...]:
    """Positions of the ``keep`` notes a subset holds, in source order.

    Notes are grouped by pitch, each pitch is allotted a share of the subset, and the notes a pitch
    contributes are those its dynamics spread evenly over. The subset therefore covers the pitch
    range the source plays and, within each pitch, the dynamics it was played across -- which is what
    makes a small slice representative enough to predict how the whole dataset behaves.

    Asking for at least as many notes as there are keeps every one of them, so a caller counting its
    share against a larger source than it hands over receives the whole of what it handed over.
    """
    groups = _dynamic_ordered_pitches(notes, dynamic_axis)
    allotted = _allotments([len(group) for group in groups], keep)
    return tuple(
        sorted(group[rank] for group, picks in zip(groups, allotted) for rank in even_ranks(len(group), picks))
    )


def select_positions(notes: Sequence[Played], fraction: float, dynamic_axis: DynamicAxisConfig) -> tuple[int, ...]:
    """Positions of the notes a subset holding ``fraction`` of ``notes`` keeps, in source order.

    Raises:
        ValueError: if ``fraction`` falls outside ``(0, 1]``.
    """
    return select_count(notes, kept_count(len(notes), fraction), dynamic_axis)


def sounding_positions(notes: Sequence[Played], min_duration_s: float) -> tuple[int, ...]:
    """Positions of the notes sounding for at least ``min_duration_s``, in source order.

    A note is measured over the span every stage after the slice reads it as -- its onset through the end
    of its release. One sounding for less than a loop may run offers the loop stage nothing to settle and
    a group nothing to stand behind, so where the floor stands is where a run's material becomes usable.
    """
    return tuple(position for position, note in enumerate(notes) if note.duration_s >= min_duration_s)


@dataclass(frozen=True)
class Admission:
    """Which of a source's notes a slice draws on, and how many the length floor held out.

    ``positions`` are the kept notes in source order. ``brief`` is how many sounded for less than the
    floor asked, which a slice reports so a source recorded largely in fragments says so at the way in.
    """

    positions: tuple[int, ...]
    brief: int


def admit(
    notes: Sequence[Played],
    *,
    fraction: float,
    min_duration_s: float,
    dynamic_axis: DynamicAxisConfig,
) -> Admission:
    """The notes a slice keeps: the share ``fraction`` asks for, drawn from those sounding long enough.

    The floor is read first and the share is counted against the source as it arrived, so a slice comes
    out the size the caller asked for while the notes filling it are all material a run can work with.
    Where the floor holds back more than the share leaves room for, what survives it is kept entire.

    Raises:
        ValueError: if ``fraction`` falls outside ``(0, 1]``.
        ValueError: when every note sounds for less than ``min_duration_s``, leaving nothing to draw on.
    """
    keep = kept_count(len(notes), fraction)
    sounding = sounding_positions(notes, min_duration_s)
    if not sounding:
        raise ValueError(
            f"every one of the {len(notes)} notes sounds for less than {min_duration_s} s, "
            "which is the length a slice admits"
        )

    carried = [notes[position] for position in sounding]
    return Admission(
        positions=tuple(sounding[rank] for rank in select_count(carried, keep, dynamic_axis)),
        brief=len(notes) - len(sounding),
    )


def notes_recorded_by(notes: Sequence[ManifestNote], recordings: Collection[int]) -> tuple[int, ...]:
    """Positions of the notes ``recordings`` accounts for, in source order.

    A recording is named by the render index its notes join on, so naming a set of recordings picks out
    exactly the material those recordings carry. That is what lets a slice be chosen by a rule reading the
    recordings themselves -- what each one sounds like -- while the notes it keeps follow from the choice.
    """
    return tuple(position for position, note in enumerate(notes) if note.render.index in recordings)


def _kept_recordings(indices: frozenset[int], samples_dir: Path) -> tuple[Path, ...]:
    """The recordings the kept notes join to, in the order a listing names them."""
    return tuple(wav for wav in sorted(samples_dir.glob("*.wav")) if index_of_wav(wav) in indices)


def _copy_recordings(indices: frozenset[int], samples_dir: Path, out_dir: Path) -> int:
    """Copy every recording the kept notes join to into ``out_dir`` and state how many landed.

    A filename carries the render index a later ingest joins on, so copying it under its own name
    leaves the subset resolving its recordings exactly as the source dataset does.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = _kept_recordings(indices, samples_dir)
    for wav in kept:
        shutil.copy2(wav, out_dir / wav.name)

    return len(kept)


def _clean_recordings(indices: frozenset[int], samples_dir: Path, out_dir: Path, subsonic: SubsonicConfig) -> int:
    """Write every recording the kept notes join to into ``out_dir`` past the band under hearing.

    This is where a run takes its material in, so it is where the depth beneath hearing comes off: each
    take is read, run through the roll-off :func:`~optisample.dsp.subsonic.remove_subsonic` states, and
    written back under its own name. Doing it once, at the way in, leaves every later stage reading a
    dataset that already holds what a listener has -- so the content each stage measures, stores and
    scores is the content the one before it did, at the level it was captured on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = _kept_recordings(indices, samples_dir)
    for wav in kept:
        signal, sample_rate = read_wav(wav)
        write_wav(out_dir / wav.name, remove_subsonic(signal, sample_rate, subsonic), sample_rate)

    return len(kept)


def _subset_manifest(raw: dict[str, Any], entries: list[Any]) -> dict[str, Any]:
    """The source manifest carrying only ``entries``, with its declared note count following suit."""
    config: dict[str, Any] = {**raw.get(_CONFIG_FIELD, {}), _NOTE_COUNT_FIELD: len(entries)}
    return {**raw, _CONFIG_FIELD: config, _NOTES_FIELD: entries}


@dataclass(frozen=True)
class _Source:
    """One dataset's manifest held both ways a slice reads it.

    ``notes`` is what the picking reads, validated against the shape the pipeline expects, and ``document``
    is the manifest exactly as its source wrote it, which is what a kept note is carried through as.
    """

    document: dict[str, Any]
    notes: tuple[ManifestNote, ...]

    @property
    def entries(self) -> list[Any]:
        """Each note as its source states it, standing at the position the parsed notes stand at."""
        written: list[Any] = self.document[_NOTES_FIELD]
        return written


def _read_source(notes_json: Path) -> _Source:
    """The manifest at ``notes_json``, read once for both the picking and the writing."""
    document: dict[str, Any] = json.loads(notes_json.read_text(encoding="utf-8"))
    return _Source(document=document, notes=tuple(NotesManifest.model_validate(document).notes))


@dataclass(frozen=True)
class _Destination:
    """The sibling pair a written slice is read back through: its manifest, beside its own recordings."""

    notes_json: Path
    samples_dir: Path


def _destination(out_dir: Path, instrument_id: str) -> _Destination:
    """Where a slice filed under ``instrument_id`` lands, in the layout a later ingest resolves by default."""
    return _Destination(notes_json=out_dir / f"{instrument_id}{NOTES_SUFFIX}", samples_dir=out_dir / instrument_id)


def _write_notes(
    source: _Source,
    positions: Sequence[int],
    destination: _Destination,
) -> tuple[ManifestNote, ...]:
    """Write the notes at ``positions`` as a manifest of the source's shape, and state which they were.

    Every slicing rule lands here, so a dataset chosen by the spread of its keys and one chosen by what its
    recordings sound like carry their notes identically and are read back the same way.

    Raises:
        ValueError: when ``positions`` names no note, which would write a dataset holding no material.
    """
    if not positions:
        raise ValueError("a written slice holds one note at the least")

    entries = source.entries
    destination.notes_json.parent.mkdir(parents=True, exist_ok=True)
    destination.notes_json.write_text(
        json.dumps(_subset_manifest(source.document, [entries[position] for position in positions]), indent=2) + "\n",
        encoding="utf-8",
    )
    return tuple(source.notes[position] for position in positions)


def _sliced(
    source: _Source,
    kept: Sequence[ManifestNote],
    destination: _Destination,
    recordings: int,
    brief_notes: int,
) -> SubsetDataset:
    """What a written slice reads back as: where it landed, how much of its source it holds, and its ranges."""
    pitches = [note.pitch for note in kept]
    velocities = [note.velocity for note in kept]
    return SubsetDataset(
        source=SourceDataset(path=destination.notes_json, samples_dir=destination.samples_dir),
        kept_notes=len(kept),
        source_notes=len(source.notes),
        recordings=recordings,
        brief_notes=brief_notes,
        pitches=(min(pitches), max(pitches)),
        velocities=(min(velocities), max(velocities)),
    )


def _indices(kept: Sequence[ManifestNote]) -> frozenset[int]:
    """The render indices the kept notes join their recordings on."""
    return frozenset(note.render.index for note in kept)


def write_subset(
    dataset: SourceDataset,
    out_dir: Path | str,
    *,
    instrument_id: str,
    fraction: float,
    intake: IntakeConfig,
) -> SubsetDataset:
    """Write the ``fraction`` of a NoteExtractor dataset that spans its pitch and velocity ranges.

    The output root is itself a NoteExtractor dataset -- ``<instrument_id>.notes.json`` beside its
    ``<instrument_id>/`` recordings -- so a later ingest resolves it by default and every stage reads it
    the way it reads the source. Each kept note is written exactly as the source states it, so the
    subset measures the same material, at the same lengths, that the whole dataset would.

    This is the pipeline's way in, so it is where a run settles what it works with: the notes sounding
    for at least ``min_duration_s`` are what the share is drawn from (:func:`admit`), and every take that
    lands is written past the band under hearing, so every stage after it reads a dataset already
    carrying the content a listener has.
    """
    source = _read_source(dataset.path)
    destination = _destination(Path(out_dir), instrument_id)
    admitted = admit(
        source.notes,
        fraction=fraction,
        min_duration_s=intake.min_duration_s,
        dynamic_axis=intake.dynamic_axis,
    )
    kept = _write_notes(source, admitted.positions, destination)
    written = _clean_recordings(_indices(kept), dataset.recordings_dir, destination.samples_dir, intake.subsonic)
    return _sliced(source, kept, destination, written, admitted.brief)


def write_recording_subset(
    dataset: SourceDataset,
    out_dir: Path | str,
    *,
    instrument_id: str,
    recordings: Collection[int],
) -> SubsetDataset:
    """Write the slice of a NoteExtractor dataset that ``recordings`` accounts for.

    ``recordings`` names the takes the slice holds by the render index each of them joins on. What lands is
    a dataset of the same shape as the source -- the notes those recordings answer for, each written
    exactly as the source states it, beside a copy of every named recording -- so a set of takes chosen by
    a rule of the caller's own is read by every later stage the way a subset is. Each take is carried over
    as it stands, which keeps a set chosen off a stage of a run holding exactly the audio that stage did.

    Raises:
        ValueError: when the named recordings answer for no note of the source.
    """
    source = _read_source(dataset.path)
    destination = _destination(Path(out_dir), instrument_id)
    kept = _write_notes(source, notes_recorded_by(source.notes, recordings), destination)
    written = _copy_recordings(_indices(kept), dataset.recordings_dir, destination.samples_dir)
    return _sliced(source, kept, destination, written, _NOTHING_HELD_OUT)
