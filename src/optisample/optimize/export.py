"""Turn an optimized :class:`InstrumentPlan` into a playable :class:`ITModule`.

This is the bridge from the optimizer's decisions to the hand-rolled IT writer. For each pitch the
solver kept, we re-encode its representative recording with the chosen params (deterministically, so
the byte count matches the plan exactly) and store it as one dedicated sample. Because every key owns
its sample, the note map is the identity and the pitch is carried entirely by a per-sample
``C5Speed`` -- key ``p`` plays at its natural rate when

    C5Speed = stored_rate * 2**((60 - p) / 12)   (60 = C-5, IT's reference key).

The material becomes a sequence of patterns: each event triggers its pitch with the velocity->volume
map applied to the volume column, held for the event's duration, then cut. Pitch-zone grouping (P5)
and loops/envelopes (P6) would change the samples and note map, not this wiring.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from optisample.dsp.surrogate import encode, semitone_ratio
from optisample.io.it_writer import (
    NOTE_CUT,
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    identity_note_map,
)
from optisample.metrics.base import Signal
from optisample.model import NoteEvent
from optisample.optimize.orchestrate import AudioMap, InstrumentPlan

_C5_KEY = 60  # IT reference key C-5; a sample plays at C5Speed when triggered here.
_MAX_IT_NOTE = 119  # IT keys span C-0..B-9.
_MAX_ROWS = 200  # IT patterns hold at most 200 rows.
_TICKS_PER_ROW_BASE = 2.5  # one tick lasts 2.5 / tempo seconds; a row lasts speed ticks.
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note_name(pitch: int) -> str:
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def c5speed_for_pitch(stored_rate: int, pitch: int) -> int:
    """C5Speed that makes key ``pitch`` play the sample (stored at ``stored_rate``) at its natural rate."""
    return int(round(stored_rate * semitone_ratio(_C5_KEY - pitch)))


def _build_samples(
    plan: InstrumentPlan, audio: AudioMap, sample_rate: int, seed: int
) -> tuple[tuple[ITSample, ...], dict[int, int]]:
    """Re-encode each pitch's representative with its chosen params; return samples + key->sample-number."""
    rng = np.random.default_rng(seed)
    samples: list[ITSample] = []
    assignment: dict[int, int] = {}
    for index, pitch_plan in enumerate(plan.pitches):
        pitch = pitch_plan.pitch
        if not 0 <= pitch <= _MAX_IT_NOTE:
            raise ValueError(f"pitch {pitch} is outside the IT key range 0..{_MAX_IT_NOTE}")
        representative: Signal = audio[(pitch, pitch_plan.representative_velocity)]
        stored = encode(representative, sample_rate, pitch_plan.chosen.params, root_pitch=pitch, rng=rng)
        samples.append(
            ITSample(
                name=f"{plan.instrument_id[:18]} {_note_name(pitch)}",
                pcm=stored.pcm,
                depth_bits=stored.depth_bits,
                c5speed=c5speed_for_pitch(stored.sample_rate, pitch),
            )
        )
        assignment[pitch] = index + 1  # sample numbers are 1-based in the note map
    return tuple(samples), assignment


def _material_patterns(
    material: Sequence[NoteEvent], plan: InstrumentPlan, playback: ITPlayback
) -> tuple[tuple[ITPattern, ...], tuple[int, ...]]:
    """Lay the material events into one or more patterns, applying the velocity->volume map."""
    row_seconds = playback.speed * _TICKS_PER_ROW_BASE / playback.tempo
    patterns: list[ITPattern] = []
    cells: list[tuple[int, int, ITCell]] = []
    cursor = 0
    for event in material:
        rows = min(max(1, int(round(event.duration_s / row_seconds))), _MAX_ROWS - 2)
        if cells and cursor + rows + 1 > _MAX_ROWS:
            patterns.append(ITPattern(rows=cursor, cells=tuple(cells)))
            cells, cursor = [], 0
        volume = plan.velocity_map.volume(event.velocity)
        cells.append((cursor, 0, ITCell(note=event.pitch, instrument=1, volume=volume)))
        cells.append((cursor + rows, 0, ITCell(note=NOTE_CUT)))
        cursor += rows + 1
    patterns.append(ITPattern(rows=max(cursor, 1), cells=tuple(cells)))
    return tuple(patterns), tuple(range(len(patterns)))


def build_it_module(
    plan: InstrumentPlan, audio: AudioMap, sample_rate: int, material: Sequence[NoteEvent], seed: int = 0
) -> ITModule:
    """Assemble a complete :class:`ITModule` from an optimized plan, its recordings and its material."""
    samples, assignment = _build_samples(plan, audio, sample_rate, seed)
    instrument = ITInstrument(name=plan.instrument_id[:25], note_map=identity_note_map(assignment))
    playback = ITPlayback()
    patterns, orders = _material_patterns(material, plan, playback)
    return ITModule(
        name=plan.instrument_id[:25],
        samples=samples,
        instruments=(instrument,),
        patterns=patterns,
        orders=orders,
        playback=playback,
    )
