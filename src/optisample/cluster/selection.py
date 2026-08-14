from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from optisample.cluster.representative import Group
from optisample.cluster.stages import RecordingCorpus, RecordingSource, StageRecording, stage_dataset
from optisample.io.dataset import SourceDataset, SubsetDataset
from optisample.io.note_extractor import index_of_wav
from optisample.io.subset import write_recording_subset


@dataclass(frozen=True)
class Pick:
    """One recording a cut sample space chose, beside the group it was chosen to stand for.

    ``recording`` is a real take a listener can play and ``place`` where it sits in the space it was chosen
    from. ``members`` is how many recordings its group gathered and ``playing_s`` the playing time the
    material gives all of them, so a pick is read beside the amount of music it was chosen to carry.
    """

    group: int
    place: int
    recording: StageRecording
    members: int
    playing_s: float

    @property
    def source(self) -> RecordingSource:
        """The set this take came from, which is the dataset a slice holding it is cut of."""
        return self.recording.source

    @property
    def render_index(self) -> int:
        """The index this recording's notes join on, which is what names it inside its own dataset."""
        return index_of_wav(self.recording.file)


@dataclass(frozen=True)
class Selection:
    """The recordings one cut of a sample space chose, one per group, in the order the groups stand.

    A selection carries a rule the pipeline's own stages have none of: dedup keeps a recording by the
    identity and the length it holds, while these were kept for what they sound like. Writing one out as a
    dataset is what puts that choice in front of ``loop``, ``reduce`` and ``optimize``.

    A space gathered from several sets hands back picks from several of them, so each pick names the set it
    came from and the sets are written apart.
    """

    picks: tuple[Pick, ...]

    @property
    def size(self) -> int:
        """How many recordings the selection holds, which is one for every group the space was cut into."""
        return len(self.picks)

    @property
    def covered(self) -> int:
        """How many recordings the picks stand for, which is the whole corpus the space was cut from."""
        return sum(pick.members for pick in self.picks)

    @property
    def playing_s(self) -> float:
        """The playing time the material gives every recording the picks stand for."""
        return sum(pick.playing_s for pick in self.picks)

    @property
    def sources(self) -> tuple[RecordingSource, ...]:
        """The sets the picks came from, in the order a pick first names each of them."""
        return tuple(dict.fromkeys(pick.source for pick in self.picks))

    def picks_from(self, source: RecordingSource) -> tuple[Pick, ...]:
        """The takes ``source`` contributed, in the order the groups behind them stand."""
        return tuple(pick for pick in self.picks if pick.source == source)

    def render_indices(self, source: RecordingSource) -> frozenset[int]:
        """The indices ``source``'s chosen recordings join on, which is what a slice holding them is named by.

        An index names a recording inside one manifest alone, so each set answers for its own and a slice is
        cut of the dataset those indices belong to.
        """
        return frozenset(pick.render_index for pick in self.picks_from(source))


def selection(corpus: RecordingCorpus, groups: Sequence[Group]) -> Selection:
    """The take standing for each group, beside what its group asked it to cover.

    The groups are read against the corpus the space was built from, so a pick names the very recording a
    point stood for and carries the file that take is stored in.
    """
    return Selection(picks=tuple(_pick(corpus, group) for group in groups))


def _pick(corpus: RecordingCorpus, group: Group) -> Pick:
    """One group's chosen take, carrying the size and the playing time of the group it stands for."""
    place = group.representative
    return Pick(
        group=group.label,
        place=place,
        recording=corpus.recordings[place],
        members=group.size,
        playing_s=sum(corpus.recordings[member].weight for member in group.members.tolist()),
    )


@dataclass(frozen=True)
class WrittenSet:
    """One set's share of a selection, as the dataset it landed as."""

    source: RecordingSource
    dataset: SubsetDataset


@dataclass(frozen=True)
class WrittenSelection:
    """Where a chosen set of recordings landed, beside the choice it was written from.

    ``written`` holds one dataset per set the picks came from, in the order the selection names them.
    """

    selection: Selection
    written: tuple[WrittenSet, ...]

    @property
    def recordings(self) -> int:
        """How many recordings the written datasets hold between them."""
        return sum(entry.dataset.recordings for entry in self.written)

    @property
    def kept_notes(self) -> int:
        """How many notes those recordings answer for, which is what a later stage weighs its work by."""
        return sum(entry.dataset.kept_notes for entry in self.written)


def write_selection(chosen: Selection, out_dir: Path) -> WrittenSelection:
    """Write the recordings ``chosen`` names out as one dataset per set they came from, under ``out_dir``.

    Each set lands in ``<out_dir>/<instrument>/<stage>/`` as a NoteExtractor dataset --
    ``<instrument>.notes.json`` beside its ``<instrument>/`` recordings -- so ``loop``, ``reduce`` and
    ``optimize`` read a clustered choice the way they read a subset. Holding the sets apart is what lets one
    instrument read at two stages answer for its picks separately, each slice cut of the dataset its own
    indices belong to. Every note the chosen recordings answer for is carried over exactly as its source
    states it and each recording is copied under its own name, so a written dataset measures the same
    material, at the same lengths, as the set the choice was made in.

    What a dataset plays is the material its own chosen recordings answer for, which is what a later stage
    weighs its allocation by; the reach each take was chosen for stays with :class:`Selection`, where the
    group behind a pick states the recordings and the playing time it stands for.

    Raises:
        ValueError: when a pick came from a set holding what an allocation stored, which its plan names
            rather than a manifest.
    """
    datasets = [(source, stage_dataset(source)) for source in chosen.sources]
    return WrittenSelection(
        selection=chosen,
        written=tuple(_written_set(source, dataset, chosen, out_dir) for source, dataset in datasets),
    )


def _written_set(
    source: RecordingSource,
    dataset: SourceDataset,
    chosen: Selection,
    out_dir: Path,
) -> WrittenSet:
    """One set's chosen takes written as the dataset holding them, under a directory of that set's own."""
    return WrittenSet(
        source=source,
        dataset=write_recording_subset(
            dataset,
            out_dir / source.instrument_id / source.stage.value,
            instrument_id=source.instrument_id,
            recordings=chosen.render_indices(source),
        ),
    )
