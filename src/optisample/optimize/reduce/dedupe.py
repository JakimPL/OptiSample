from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from sys import maxsize
from typing import Final

from optisample.config.dsp import LoopConfig
from optisample.config.reduce import ReduceConfig
from optisample.io.audio import probe_wav
from optisample.io.note_extractor import index_of_wav
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.music import semitone_ratio
from optisample.optimize.reduce.keys import (
    DedupeGroup,
    SampleKey,
    dedupe_group,
    sample_key,
)
from optisample.progress import ProgressSink

_UNINDEXED: Final = maxsize  # a recording whose filename carries no render index ranks after indexed ones
NO_MATERIAL_S: Final = 0.0  # a pitch the material leaves unplayed only has the loop floor to satisfy
_PROBE_LABEL: Final = "Probing recordings"


@dataclass(frozen=True)
class Candidate:
    """One recorded WAV competing for the single slot its identity keeps.

    ``usable_duration_s`` is what survives the loader's ``lead_in_s`` and ``trail_out_s`` trims, so it
    measures the recording from the note onset to its release end -- the span the material is scored
    against. ``order`` ranks recordings that are equally suitable, by render index and then filename, so
    the survivor is the same on every run.
    """

    sample: SourceSample
    key: SampleKey
    usable_duration_s: float
    order: tuple[int, str]


@dataclass(frozen=True)
class Selection:
    """The recording one identity keeps, alongside what it was chosen against."""

    key: SampleKey
    sample: SourceSample
    duration_s: float
    required_duration_s: float
    considered: int

    @property
    def covers_material(self) -> bool:
        """Whether the kept recording holds every note its key has to serve, in full."""
        return holds_material(self.duration_s, self.required_duration_s)


def holds_material(duration_s: float, required_s: float) -> bool:
    """Whether a recording of ``duration_s`` covers the ``required_s`` the material asks of its key.

    The one rule both the selection and the run summary read, so a recording judged long enough when it
    was kept reads as long enough everywhere it is reported.
    """
    return duration_s >= required_s


def required_duration_s(longest_note_s: float, reduce: ReduceConfig, loop: LoopConfig) -> float:
    """Seconds a kept recording must hold at a pitch for the material there to play in full.

    Three demands set the length. A sample serving keys above its own root runs faster by
    :func:`~optisample.music.semitone_ratio`, consuming stored frames at that rate, so the longest note at
    the pitch grows by the ratio of ``transposition_headroom_semitones``. The trim's ``max_length_s``
    bounds that, since a recording is only ever asked for the span the trim keeps. Loop detection
    separately needs room to work in -- the attack it skips, the shortest loop it accepts, and the tail it
    leaves alone -- which holds the requirement up wherever the trim would cut under it.
    """
    transposed = longest_note_s * semitone_ratio(reduce.dedupe.transposition_headroom_semitones)
    loop_floor = loop.attack_skip_s + loop.min_loop_s + loop.tail_skip_s
    return max(min(transposed, reduce.trim.max_length_s), loop_floor)


def longest_note_by_pitch(material: Sequence[NoteEvent]) -> dict[int, float]:
    """The longest note the material holds at each pitch, which every recording there must cover.

    The requirement is pitch-wide rather than per identity because any survivor at a pitch can end up
    serving the pitch's longest note: as the reference the nearest-velocity lookup routes that note to,
    or as the representative a zone stores and repitches.
    """
    longest: dict[int, float] = {}
    for event in material:
        longest[event.pitch] = max(longest.get(event.pitch, NO_MATERIAL_S), event.duration_s)

    return longest


def _rank(sample: SourceSample) -> tuple[int, str]:
    """Deterministic rank among equally suitable recordings: the render index, then the filename.

    The render index is the recording's own identity in the trimmed grid, so it survives a manifest
    being reordered. Filenames the trimmer did not write carry no index and rank after the ones that do.
    """
    try:
        index = index_of_wav(sample.file)
    except ValueError:
        index = _UNINDEXED

    return index, sample.file.name


def _candidate(sample: SourceSample, reduce: ReduceConfig) -> Candidate:
    """Turn one listed recording into the identity and onset-aligned length dedup ranks it by.

    The WAV header states the length, so the whole recorded grid is ranked from the headers alone. The
    length is read as the trim will leave it -- the padding at either end gone and ``max_length_s``
    bounding what remains -- so recordings are ranked on the span each of them will actually contribute.
    """
    info = probe_wav(sample.file)
    onset_aligned = max(info.duration_s - sample.lead_in_s - sample.trail_out_s, NO_MATERIAL_S)
    return Candidate(
        sample=sample,
        key=sample_key(sample, reduce.dedupe),
        usable_duration_s=min(onset_aligned, reduce.trim.max_length_s),
        order=_rank(sample),
    )


def _keep(candidates: Sequence[Candidate], required_s: float) -> Candidate:
    """The shortest candidate that still covers ``required_s``, or the longest one when none does."""
    covering = [candidate for candidate in candidates if candidate.usable_duration_s >= required_s]
    if covering:
        return min(covering, key=lambda candidate: (candidate.usable_duration_s, candidate.order))

    return min(candidates, key=lambda candidate: (-candidate.usable_duration_s, candidate.order))


def select_recordings(
    instrument: InstrumentSpec,
    reduce: ReduceConfig,
    loop: LoopConfig,
    progress: ProgressSink,
) -> tuple[Selection, ...]:
    """Reduce the recorded grid to one survivor per identity, shrinking what the optimizer explores.

    A tracker stores one sample per key, so every recording sharing a
    :class:`~optisample.optimize.reduce.keys.DedupeGroup` competes for a single slot. The shortest
    survivor that still covers its pitch's requirement (see :func:`required_duration_s`) leaves the
    encoder the least material to work through; when every recording in a group falls short, the longest
    one keeps as much of the note as was recorded, and its :attr:`Selection.covers_material` says so.

    Selections come back in :class:`~optisample.optimize.reduce.keys.SampleKey` order, so the audio map,
    the pitch tasks and every label derived from them are stable across runs.
    """
    longest = longest_note_by_pitch(instrument.material)
    groups: dict[DedupeGroup, list[Candidate]] = {}
    listed = instrument.samples
    for sample in progress.track(listed, label=_PROBE_LABEL, total=len(listed)):
        candidate = _candidate(sample, reduce)
        groups.setdefault(dedupe_group(candidate.key, reduce.dedupe.key), []).append(candidate)

    selections = []
    for group, candidates in groups.items():
        required_s = required_duration_s(longest.get(group.pitch, NO_MATERIAL_S), reduce, loop)
        kept = _keep(candidates, required_s)
        selections.append(
            Selection(
                key=kept.key,
                sample=kept.sample,
                duration_s=kept.usable_duration_s,
                required_duration_s=required_s,
                considered=len(candidates),
            )
        )

    return tuple(sorted(selections, key=lambda selection: selection.key))
