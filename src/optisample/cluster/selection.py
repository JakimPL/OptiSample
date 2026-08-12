from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from optisample.cluster.representative import Group
from optisample.cluster.stages import Stage, StageCorpus, StageRecording
from optisample.config.cluster import Representative
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
    def render_index(self) -> int:
        """The index this recording's notes join on, which is what names it inside its own dataset."""
        return index_of_wav(self.recording.file)


@dataclass(frozen=True)
class Selection:
    """The recordings one cut of a sample space chose, one per group, in the order the groups stand.

    A selection carries a rule the pipeline's own stages have none of: dedup keeps a recording by the
    identity and the length it holds, while these were kept for what they sound like. Writing one out as a
    dataset is what puts that choice in front of ``loop``, ``reduce`` and ``optimize``.
    """

    stage: Stage
    instrument_id: str
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
    def render_indices(self) -> frozenset[int]:
        """The indices the chosen recordings join on, which is what a slice holding them is named by."""
        return frozenset(pick.render_index for pick in self.picks)


def selection(corpus: StageCorpus, groups: Sequence[Group], *, rule: Representative) -> Selection:
    """The take standing for each group under ``rule``, beside what its group asked it to cover.

    The groups are read against the corpus the space was built from, so a pick names the very recording a
    point stood for and carries the file that take is stored in.
    """
    return Selection(
        stage=corpus.stage,
        instrument_id=corpus.instrument_id,
        picks=tuple(_pick(corpus, group, rule) for group in groups),
    )


def _pick(corpus: StageCorpus, group: Group, rule: Representative) -> Pick:
    """One group's chosen take, carrying the size and the playing time of the group it stands for."""
    place = group.representative(rule)
    return Pick(
        group=group.label,
        place=place,
        recording=corpus.recordings[place],
        members=group.size,
        playing_s=sum(corpus.recordings[member].weight for member in group.members.tolist()),
    )


@dataclass(frozen=True)
class WrittenSelection:
    """Where a chosen set of recordings landed as a dataset, beside the choice it was written from."""

    selection: Selection
    dataset: SubsetDataset


def write_selection(source: SourceDataset, chosen: Selection, out_dir: Path) -> WrittenSelection:
    """Write the recordings ``chosen`` names out of ``source`` as a dataset, under ``out_dir``.

    The output root is itself a NoteExtractor dataset -- ``<instrument_id>.notes.json`` beside its
    ``<instrument_id>/`` recordings -- so ``loop``, ``reduce`` and ``optimize`` read a clustered choice the
    way they read a subset. Every note the chosen recordings answer for is carried over exactly as the
    source states it and each recording is copied under its own name, so the written dataset measures the
    same material, at the same lengths, as the stage the choice was made in.

    What the dataset plays is the material the chosen recordings themselves answer for, which is what a
    later stage weighs its allocation by; the reach each take was chosen for stays with
    :class:`Selection`, where the group behind a pick states the recordings and the playing time it
    stands for.

    The manifest names exactly the chosen recordings, which is what a later stage reads the directory by.
    """
    return WrittenSelection(
        selection=chosen,
        dataset=write_recording_subset(
            source,
            out_dir,
            instrument_id=chosen.instrument_id,
            recordings=chosen.render_indices,
        ),
    )
