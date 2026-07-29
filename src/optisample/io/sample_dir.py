from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.io.audio import probe_wav
from optisample.io.dataset import SourceDataset, SubsetDataset
from optisample.io.note_extractor import IngestSettings
from optisample.io.subset import select_positions
from optisample.model import InstrumentSpec, Manifest, NoteEvent, SourceSample
from optisample.music import MIDI_MAX_VELOCITY, named_pitch

WAV_GLOB: Final = "*.wav"
UNKNOWN_VELOCITY: Final = MIDI_MAX_VELOCITY

_TOKEN_SEPARATOR: Final = "_"
_NUMBERED_PITCH_PATTERN: Final = re.compile(r"^p(\d+)$")
_VELOCITY_PATTERN: Final = re.compile(r"^v(\d+)$")
_CONTROLLER_PATTERN: Final = re.compile(r"^cc(\d+)-(\d+(?:\.\d+)?)$")


@dataclass(frozen=True)
class RecordedTake:
    """One WAV of a directory of recordings: what its filename spells, and the length its header reports.

    ``duration_s`` is the whole recorded span; the lead-in an ingest trims comes off it when the take is
    read as a note, so a folder recorded with padding measures the same span a manifest dataset does.
    """

    file: Path
    pitch: int
    velocity: int
    duration_s: float
    cc_averages: dict[int, float]


@dataclass(frozen=True)
class _SpelledName:
    """What a WAV's filename says about the take it holds, with the velocity open where it names none."""

    pitch: int
    velocity: int | None
    cc_averages: dict[int, float]


def _numbered_pitch(token: str) -> int | None:
    """The pitch a ``p<number>`` token states, which is how every dataset this project writes names one."""
    match = _NUMBERED_PITCH_PATTERN.match(token)
    return int(match.group(1)) if match else None


def _spelled_pitch(token: str) -> int | None:
    """The pitch a scientific note-name token spells, which is how a sample library names its takes."""
    try:
        return named_pitch(token)
    except ValueError:
        return None


def _pitch_of(tokens: Sequence[str], name: str) -> int:
    """The pitch a filename states: its ``p<number>`` token where it carries one, else its note name.

    The numbered form is read first because the datasets this project writes carry both spellings at
    once (``0000_p029_F1_v018``), so leading with it settles them on the one they agree at.

    Raises:
        ValueError: when no token of ``name`` states a pitch at all.
    """
    numbered = next((pitch for pitch in map(_numbered_pitch, tokens) if pitch is not None), None)
    if numbered is not None:
        return numbered

    spelled = next((pitch for pitch in map(_spelled_pitch, tokens) if pitch is not None), None)
    if spelled is None:
        raise ValueError(f"recording {name!r} names no pitch: a p<number> or note-name token states one")

    return spelled


def _velocity_of(tokens: Sequence[str]) -> int | None:
    """The velocity a ``v<number>`` token states, left open for the filenames naming none."""
    return next((int(match.group(1)) for match in map(_VELOCITY_PATTERN.match, tokens) if match), None)


def _controllers_of(tokens: Sequence[str]) -> dict[int, float]:
    """Every ``cc<number>-<average>`` token as the controller average it records."""
    matches = (_CONTROLLER_PATTERN.match(token) for token in tokens)
    return {int(match.group(1)): float(match.group(2)) for match in matches if match}


def _spell_name(path: Path) -> _SpelledName:
    """Read one recording's filename for the key, dynamic and controller averages it names."""
    tokens = path.stem.split(_TOKEN_SEPARATOR)
    return _SpelledName(
        pitch=_pitch_of(tokens, path.name),
        velocity=_velocity_of(tokens),
        cc_averages=_controllers_of(tokens),
    )


def _resolved_velocities(spelled: Sequence[_SpelledName], paths: Sequence[Path]) -> tuple[int, ...]:
    """The velocity each take is read at: what its name spells, or one shared dynamic where none does.

    Filenames silent on how hard each key was struck describe a set of recordings holding one take per
    key, so reading them all at :data:`UNKNOWN_VELOCITY` puts them in a single dynamic -- which is the
    most such a set says about itself, and leaves every played velocity routing to it.

    Raises:
        ValueError: when some filenames spell a velocity and others leave it out, which would read one
            axis two ways within the same set of recordings.
    """
    named = [name.velocity for name in spelled]
    if all(velocity is None for velocity in named):
        return tuple(UNKNOWN_VELOCITY for _ in named)

    silent = [path.name for path, velocity in zip(paths, named) if velocity is None]
    if silent:
        raise ValueError(f"recordings naming no velocity beside ones that do: {', '.join(silent)}")

    return tuple(velocity for velocity in named if velocity is not None)


def read_takes(samples_dir: Path) -> tuple[RecordedTake, ...]:
    """Every WAV of ``samples_dir`` as the take its filename and header describe, in filename order.

    Filename order is the order a later stage ranks duplicates in, so the same directory reduces to the
    same survivors on every run.

    Raises:
        ValueError: when ``samples_dir`` holds no WAV to read.
    """
    paths = sorted(samples_dir.glob(WAV_GLOB))
    if not paths:
        raise ValueError(f"no WAV recordings in {samples_dir}")

    spelled = [_spell_name(path) for path in paths]
    velocities = _resolved_velocities(spelled, paths)
    return tuple(
        RecordedTake(
            file=path.resolve(),
            pitch=name.pitch,
            velocity=velocity,
            duration_s=probe_wav(path).duration_s,
            cc_averages=name.cc_averages,
        )
        for path, name, velocity in zip(paths, spelled, velocities)
    )


def _recorded_samples(takes: Sequence[RecordedTake], lead_in_s: float) -> list[SourceSample]:
    """Every take as a recording the optimizer may store, aligned so frame 0 lands on the note onset."""
    return [
        SourceSample(
            file=take.file,
            pitch=take.pitch,
            velocity=take.velocity,
            cc_averages=take.cc_averages,
            lead_in_s=lead_in_s,
        )
        for take in takes
    ]


def _recorded_material(takes: Sequence[RecordedTake], lead_in_s: float) -> list[NoteEvent]:
    """Every take as one note held for as long as the recording sounds, measured from its onset."""
    return [
        NoteEvent(
            pitch=take.pitch,
            velocity=take.velocity,
            cc_averages=take.cc_averages,
            duration_s=take.duration_s - lead_in_s,
        )
        for take in takes
    ]


def load_sample_dir(samples_dir: Path | str, settings: IngestSettings) -> Manifest:
    """Read a directory of recordings as a single-instrument manifest, its filenames standing for the material.

    A directory states which keys were recorded and how hard, and says nothing of a song, so each take is
    read as one note held for as long as its recording sounds. Usage weight then follows take length and
    the recorded grid itself is what the allocation is optimized over, which is what a set of samples
    with no performance behind it asks for.

    ``pre_roll_s`` comes off the front of every take as its lead-in, so a set recorded with padding
    measures its notes from the onset the way a manifest dataset does.
    """
    takes = read_takes(Path(samples_dir).resolve())
    instrument = InstrumentSpec(
        id=settings.instrument_id,
        budget_kb=settings.budget_kb,
        samples=_recorded_samples(takes, settings.pre_roll_s),
        material=_recorded_material(takes, settings.pre_roll_s),
        pre_roll_s=settings.pre_roll_s,
        post_roll_s=settings.post_roll_s,
    )
    return Manifest(project=settings.project, instruments=[instrument])


def write_sample_dir_subset(
    samples_dir: Path | str,
    out_dir: Path | str,
    *,
    instrument_id: str,
    fraction: float,
) -> SubsetDataset:
    """Write the ``fraction`` of a directory of recordings that spans its pitch and velocity ranges.

    The output is itself a directory of recordings, at ``<out_dir>/<instrument_id>``, each take copied
    under its own name -- so the slice reads back exactly the way its source does, and the ranges it
    spans say whether it still exercises the whole instrument.
    """
    takes = read_takes(Path(samples_dir))
    kept = [takes[position] for position in select_positions(takes, fraction)]
    target = Path(out_dir) / instrument_id
    target.mkdir(parents=True, exist_ok=True)
    for take in kept:
        shutil.copy2(take.file, target / take.file.name)

    pitches = [take.pitch for take in kept]
    velocities = [take.velocity for take in kept]
    return SubsetDataset(
        source=SourceDataset(path=target, samples_dir=None),
        kept_notes=len(kept),
        source_notes=len(takes),
        recordings=len(kept),
        pitches=(min(pitches), max(pitches)),
        velocities=(min(velocities), max(velocities)),
    )
