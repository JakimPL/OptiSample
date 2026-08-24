from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from optisample.config.subset import IntakeConfig
from optisample.dsp.subsonic import remove_subsonic
from optisample.io.audio import probe_wav, read_wav, write_wav
from optisample.io.dataset import SourceDataset, SubsetDataset
from optisample.io.note_extractor import NO_PADDING_S, IngestSettings
from optisample.io.subset import admit
from optisample.model import InstrumentSpec, Manifest, NoteEvent, SourceSample
from optisample.music import MIDI_MAX_VELOCITY, named_pitch

WAV_GLOB: Final = "*.wav"
UNKNOWN_VELOCITY: Final = MIDI_MAX_VELOCITY

_TOKEN_SEPARATOR: Final = "_"
_NUMBERED_PITCH_PATTERN: Final = re.compile(r"^p(\d+)$")
_VELOCITY_PATTERN: Final = re.compile(r"^v(\d+)$")
_DECIMAL_POINT_MARKER: Final = "p"  # a controller average spells its point this way, which every filesystem accepts
_MINUS_SIGN_MARKER: Final = "m"  # and its sign this way, for the same reason
_CONTROLLER_PATTERN: Final = re.compile(rf"^cc(\d+)-({_MINUS_SIGN_MARKER}?\d+(?:[{_DECIMAL_POINT_MARKER}.]\d+)?)$")


@dataclass(frozen=True)
class RecordedTake:
    """One WAV of a directory of recordings: what its filename spells, and the length its header reports.

    ``duration_s`` is the whole recorded span; the padding an ingest trims comes off either end when the
    take is read as a note, so a folder recorded with padding measures the same span a manifest dataset
    does.
    """

    file: Path
    pitch: int
    velocity: int
    duration_s: float
    cc_averages: dict[int, float]


@dataclass(frozen=True)
class _Padding:
    """The audio at either end of a directory's takes that its ingest was told to leave out."""

    lead_in_s: float
    trail_out_s: float


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


def _controller_average(spelled: str) -> float:
    """The average a ``cc<number>-<average>`` token spells, with its markers read back as the signs they stand for.

    A recording is named for what it holds, and a controller average is a decimal, so the writer spells the
    point and the sign as the letters :data:`_DECIMAL_POINT_MARKER` and :data:`_MINUS_SIGN_MARKER` -- which
    is what keeps the name to characters every filesystem carries. Reading them back here is what lets a
    directory of recordings state the same averages its manifest does. A point written as itself reads the
    same way, so a name from either hand is understood.
    """
    return float(spelled.replace(_MINUS_SIGN_MARKER, "-").replace(_DECIMAL_POINT_MARKER, "."))


def _controllers_of(tokens: Sequence[str]) -> dict[int, float]:
    """Every ``cc<number>-<average>`` token as the controller average it records."""
    matches = (_CONTROLLER_PATTERN.match(token) for token in tokens)
    return {int(match.group(1)): _controller_average(match.group(2)) for match in matches if match}


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


def _padding(settings: IngestSettings) -> _Padding:
    """What the ingest flags say a directory's takes hold around each note.

    A directory names no rolls of its own, so the flags are taken at their word at both ends, and
    ``keep_tail`` stores each take through its release padding.
    """
    return _Padding(
        lead_in_s=settings.pre_roll_s,
        trail_out_s=NO_PADDING_S if settings.keep_tail else settings.post_roll_s,
    )


def _sounding_s(take: RecordedTake, padding: _Padding) -> float:
    """How long one take's note sounds: its recorded span, less the padding at either end.

    Raises:
        ValueError: when the padding claims the whole take, leaving no note to store.
    """
    sounding = take.duration_s - padding.lead_in_s - padding.trail_out_s
    if sounding <= 0.0:
        raise ValueError(
            f"recording {take.file.name!r} holds {take.duration_s:.3f}s, all of it padding: "
            f"{padding.lead_in_s:.3f}s before the onset and {padding.trail_out_s:.3f}s after the release"
        )

    return sounding


def _recorded_samples(takes: Sequence[RecordedTake], padding: _Padding) -> list[SourceSample]:
    """Every take as a recording the optimizer may store, over the span its note sounds."""
    return [
        SourceSample(
            file=take.file,
            pitch=take.pitch,
            velocity=take.velocity,
            cc_averages=take.cc_averages,
            lead_in_s=padding.lead_in_s,
            trail_out_s=padding.trail_out_s,
        )
        for take in takes
    ]


def _recorded_material(takes: Sequence[RecordedTake], padding: _Padding) -> list[NoteEvent]:
    """Every take as one note held for as long as the recording sounds between its padding."""
    return [
        NoteEvent(
            pitch=take.pitch,
            velocity=take.velocity,
            cc_averages=take.cc_averages,
            duration_s=_sounding_s(take, padding),
        )
        for take in takes
    ]


def load_sample_dir(samples_dir: Path | str, settings: IngestSettings) -> Manifest:
    """Read a directory of recordings as a single-instrument manifest, its filenames standing for the material.

    A directory states which keys were recorded and how hard, and says nothing of a song, so each take is
    read as one note held for as long as its recording sounds. Usage weight then follows take length and
    the recorded grid itself is what the allocation is optimized over, which is what a set of samples
    with no performance behind it asks for.

    ``pre_roll_s`` comes off the front of every take and ``post_roll_s`` off the end, so a set recorded
    with padding measures its notes between them the way a manifest dataset does.

    Raises:
        ValueError: when the padding claims a whole take, leaving no note to store.
    """
    takes = read_takes(Path(samples_dir).resolve())
    padding = _padding(settings)
    instrument = InstrumentSpec(
        id=settings.instrument_id,
        budget_kb=settings.budget_kb,
        samples=_recorded_samples(takes, padding),
        material=_recorded_material(takes, padding),
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
    intake: IntakeConfig,
) -> SubsetDataset:
    """Write the ``fraction`` of a directory of recordings that spans its pitch and velocity ranges.

    The output is itself a directory of recordings, at ``<out_dir>/<instrument_id>``, each take written
    under its own name -- so the slice reads back exactly the way its source does, and the ranges it
    spans say whether it still exercises the whole instrument. This is the pipeline's way in for a
    directory of takes, so it is where the band under hearing comes off
    (:func:`~optisample.dsp.subsonic.remove_subsonic`), leaving every later stage a dataset that already
    holds the content a listener has.

    A take is held to ``min_duration_s`` over the span its own header reports, which is what a directory
    states about itself; a source declaring the padding it was cut with says so through its manifest and
    is sliced against the note's own sounding span instead.
    """
    takes = read_takes(Path(samples_dir))
    admitted = admit(
        takes,
        fraction=fraction,
        min_duration_s=intake.min_duration_s,
        dynamic_axis=intake.dynamic_axis,
    )
    kept = [takes[position] for position in admitted.positions]
    target = Path(out_dir) / instrument_id
    target.mkdir(parents=True, exist_ok=True)
    for take in kept:
        signal, sample_rate = read_wav(take.file)
        write_wav(target / take.file.name, remove_subsonic(signal, sample_rate, intake.subsonic), sample_rate)

    pitches = [take.pitch for take in kept]
    velocities = [take.velocity for take in kept]
    return SubsetDataset(
        source=SourceDataset(path=target, samples_dir=None),
        kept_notes=len(kept),
        source_notes=len(takes),
        recordings=len(kept),
        brief_notes=admitted.brief,
        pitches=(min(pitches), max(pitches)),
        velocities=(min(velocities), max(velocities)),
    )
