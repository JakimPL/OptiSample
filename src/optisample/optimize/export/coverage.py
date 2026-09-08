from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from optisample.io.tracker.target import ExportTarget
from optisample.optimize.plans import SampleUnit
from trackmod.core.instruments.keymap import KeyAssignment, Keymap
from trackmod.core.notes.pitch import Note
from trackmod.spec.pitch import NOTE_COUNT


@dataclass(frozen=True)
class KeyCoverage:
    """How much of a format's keyboard the written instruments answer.

    ``numbered`` is the keys the format offers, ``played`` the keys the material itself reached, and
    ``answered`` how many resolve to a sample once the rest are filled from their nearest neighbor.
    """

    numbered: int
    played: int
    answered: int

    @property
    def filled(self) -> int:
        """Keys answered by the recording nearest them rather than by one of their own."""
        return self.answered - self.played


def played_keys(units: Sequence[SampleUnit]) -> int:
    """How many distinct keys the plan's stored samples were given material at.

    A layered plan stores a key once per band it is played in, so counting the keys themselves states
    the stretch of keyboard the recordings came from however many layers reach it.
    """
    return len({key for unit in units for key in unit.keys})


def _shifted(assignment: KeyAssignment, interval: int) -> KeyAssignment | None:
    """``assignment`` moved ``interval`` semitones, for the keys whose sounded note a tracker numbers.

    Moving the assignment as an interval keeps every key routed to one sample at a single offset from
    it, which is the chromatic playback a repitched zone already relies on and what FastTracker 2 asks
    of a sample several keys share. A tracker states a transposition as the note it sounds, so a
    recording reaches five octaves below its own key and just under five above; a shift past that
    answers with nothing.
    """
    sounded = assignment.note.value + interval
    if not 0 <= sounded < NOTE_COUNT:
        return None

    return KeyAssignment(sample=assignment.sample, note=Note(sounded))


def _answering(routing: Mapping[Note, KeyAssignment], played: Sequence[int], key: int) -> KeyAssignment | None:
    """The closest recording a tracker can sound at ``key``, taking the lower of two equally close ones.

    A tie goes downward because the pre-optimization stage prices every recording with room to be played
    up -- ``transposition_headroom_semitones`` reserves bandwidth for exactly that interval, and the
    useful-rate bound is measured over it -- so the lower recording is the one whose stored spectrum
    already covers the distance. Recordings are tried outward from the key, so one too far for a tracker
    to name hands the key to the next one out, and a key past every recording's reach waits as it stood.
    """
    for nearest in sorted(played, key=lambda candidate: (abs(candidate - key), candidate)):
        shifted = _shifted(routing[Note(nearest)], key - nearest)
        if shifted is not None:
            return shifted

    return None


def covered_routing(routing: Mapping[Note, KeyAssignment], target: ExportTarget) -> dict[Note, KeyAssignment]:
    """``routing`` widened to answer every key the format numbers, each from the recording nearest it.

    An instrument is recorded over the keys its material plays, which leaves the rest of the keyboard to
    whatever the format makes of a key naming no sample: Impulse Tracker sounds silence, and FastTracker
    2, whose keymap numbers every key to a real slot, sounds the instrument's first sample at that
    sample's own tuning. Filling from the recording nearest each key widens a written instrument to the
    stretch its recordings reach -- the lowest reaches down toward the bottom key, the highest up toward
    the top, and each stretch between two of them goes to whichever side is closer (:func:`_answering`).
    The reach is the five octaves either way a tracker can name (:func:`_shifted`), so the keys answered
    run unbroken from the first key some recording reaches to the last.
    """
    if not routing:
        return {}

    played = sorted(key.value for key in routing)
    covered = dict(routing)
    for pitch in range(target.min_pitch, target.max_pitch + 1):
        key = target.key(pitch)
        if key in covered:
            continue

        answering = _answering(routing, played, key.value)
        if answering is not None:
            covered[key] = answering

    return covered


def key_coverage(keymaps: Sequence[Keymap], target: ExportTarget, *, played: int) -> KeyCoverage:
    """What the written routings answer of the keyboard, read off the keymaps as they were written.

    Each written instrument is filled from the recordings it owns and a recording reaches five octaves
    either way, so an instrument holding one stretch of the keyboard answers as much of it as that reach
    allows. The count reports the least covered instrument, which is the figure a reader may take as true
    of every one of them.
    """
    keys = [target.key(pitch).value for pitch in range(target.min_pitch, target.max_pitch + 1)]
    answered = min((sum(keymap[key] is not None for key in keys) for keymap in keymaps), default=0)
    return KeyCoverage(numbered=len(keys), played=played, answered=answered)
