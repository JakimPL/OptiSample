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
from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.render import PlaybackConfig
from optisample.dsp.surrogate import EncodeContext, StoredSample, encode
from optisample.io.it_writer import (
    MAX_ROWS,
    NOTE_CUT,
    TICKS_PER_ROW_BASE,
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    identity_note_map,
    it_playback,
)
from optisample.metrics.base import Signal
from optisample.model import NoteEvent
from optisample.music import note_name, semitone_ratio
from optisample.optimize.grouping import GroupedInstrumentPlan
from optisample.optimize.orchestrate import InstrumentPlan
from optisample.optimize.tasks import AudioMap
from optisample.optimize.velocity_map import VelocityVolumeMap

_C5_KEY = 60  # IT reference key C-5; a sample plays at C5Speed when triggered here.
_MAX_IT_NOTE = 119  # IT keys span C-0..B-9.


@dataclass(frozen=True)
class ExportContext:
    """Config the IT exporter needs beyond a plan: how to re-encode each sample and how it plays back.

    ``seed`` seeds the per-sample dither so re-encoding a plan reproduces the exact bytes it budgeted.
    """

    encode: EncodeConfig
    playback: PlaybackConfig
    seed: int = 0


def c5speed_for_pitch(stored_rate: int, pitch: int) -> int:
    """C5Speed that makes key ``pitch`` play the sample (stored at ``stored_rate``) at its natural rate."""
    return int(round(stored_rate * semitone_ratio(_C5_KEY - pitch)))


def _it_loop(stored: StoredSample) -> tuple[int, int] | None:
    """The stored sample's loop as the ``(begin, end)`` frame pair the IT writer expects (or ``None``)."""
    return None if stored.loop is None else (stored.loop.start, stored.loop.end)


def _build_samples(
    plan: InstrumentPlan, audio: AudioMap, sample_rate: int, ctx: ExportContext
) -> tuple[tuple[ITSample, ...], dict[int, int]]:
    """Re-encode each pitch's representative with its chosen params; return samples + key->sample-number."""
    rng = np.random.default_rng(ctx.seed)
    samples: list[ITSample] = []
    assignment: dict[int, int] = {}
    for index, pitch_plan in enumerate(plan.pitches):
        pitch = pitch_plan.pitch
        if not 0 <= pitch <= _MAX_IT_NOTE:
            raise ValueError(f"pitch {pitch} is outside the IT key range 0..{_MAX_IT_NOTE}")
        representative: Signal = audio[(pitch, pitch_plan.representative_velocity)]
        encode_ctx = EncodeContext(root_pitch=pitch, config=ctx.encode, rng=rng)
        stored = encode(representative, sample_rate, pitch_plan.chosen.params, encode_ctx)
        samples.append(
            ITSample(
                name=f"{plan.instrument_id[:18]} {note_name(pitch)}",
                pcm=stored.pcm,
                depth_bits=stored.depth_bits,
                c5speed=c5speed_for_pitch(stored.sample_rate, pitch),
                loop=_it_loop(stored),
            )
        )
        assignment[pitch] = index + 1  # sample numbers are 1-based in the note map
    return tuple(samples), assignment


def _material_patterns(
    material: Sequence[NoteEvent], velocity_map: VelocityVolumeMap, playback: ITPlayback
) -> tuple[tuple[ITPattern, ...], tuple[int, ...]]:
    """Lay the material events into one or more patterns, applying the velocity->volume map."""
    row_seconds = playback.speed * TICKS_PER_ROW_BASE / playback.tempo
    patterns: list[ITPattern] = []
    cells: list[tuple[int, int, ITCell]] = []
    cursor = 0
    for event in material:
        rows = min(max(1, int(round(event.duration_s / row_seconds))), MAX_ROWS - 2)
        if cells and cursor + rows + 1 > MAX_ROWS:
            patterns.append(ITPattern(rows=cursor, cells=tuple(cells)))
            cells, cursor = [], 0
        volume = velocity_map.volume(event.velocity)
        cells.append((cursor, 0, ITCell(note=event.pitch, instrument=1, volume=volume)))
        cells.append((cursor + rows, 0, ITCell(note=NOTE_CUT)))
        cursor += rows + 1
    patterns.append(ITPattern(rows=max(cursor, 1), cells=tuple(cells)))
    return tuple(patterns), tuple(range(len(patterns)))


def build_it_module(
    plan: InstrumentPlan, audio: AudioMap, sample_rate: int, material: Sequence[NoteEvent], ctx: ExportContext
) -> ITModule:
    """Assemble a complete :class:`ITModule` from an optimized plan, its recordings and its material."""
    samples, assignment = _build_samples(plan, audio, sample_rate, ctx)
    instrument = ITInstrument(name=plan.instrument_id[:25], note_map=identity_note_map(assignment))
    playback = it_playback(ctx.playback)
    patterns, orders = _material_patterns(material, plan.velocity_map, playback)
    return ITModule(
        name=plan.instrument_id[:25],
        samples=samples,
        instruments=(instrument,),
        patterns=patterns,
        orders=orders,
        playback=playback,
    )


def _build_zone_samples(
    plan: GroupedInstrumentPlan, audio: AudioMap, sample_rate: int, ctx: ExportContext
) -> tuple[tuple[ITSample, ...], dict[int, int]]:
    """Re-encode one representative per zone; map every key the zone covers to that shared sample.

    A zone's sample is rooted at its representative pitch (``C5Speed`` tuned so the representative key
    plays natural); the note map then sends each covered key to that same sample, and the tracker
    repitches it by ``key - representative`` semitones -- exactly the transpose the surrogate scored.
    """
    rng = np.random.default_rng(ctx.seed)
    samples: list[ITSample] = []
    assignment: dict[int, int] = {}
    for index, zone in enumerate(plan.zones):
        representative: Signal = audio[(zone.representative, zone.representative_velocity)]
        encode_ctx = EncodeContext(root_pitch=zone.representative, config=ctx.encode, rng=rng)
        stored = encode(representative, sample_rate, zone.chosen.params, encode_ctx)
        samples.append(
            ITSample(
                name=f"{plan.instrument_id[:18]} {note_name(zone.representative)}",
                pcm=stored.pcm,
                depth_bits=stored.depth_bits,
                c5speed=c5speed_for_pitch(stored.sample_rate, zone.representative),
                loop=_it_loop(stored),
            )
        )
        for pitch in zone.pitches:
            if not 0 <= pitch <= _MAX_IT_NOTE:
                raise ValueError(f"pitch {pitch} is outside the IT key range 0..{_MAX_IT_NOTE}")
            assignment[pitch] = index + 1  # sample numbers are 1-based in the note map
    return tuple(samples), assignment


def build_grouped_it_module(
    plan: GroupedInstrumentPlan, audio: AudioMap, sample_rate: int, material: Sequence[NoteEvent], ctx: ExportContext
) -> ITModule:
    """Assemble a complete :class:`ITModule` from a grouped plan (one sample per zone, repitched)."""
    samples, assignment = _build_zone_samples(plan, audio, sample_rate, ctx)
    instrument = ITInstrument(name=plan.instrument_id[:25], note_map=identity_note_map(assignment))
    playback = it_playback(ctx.playback)
    patterns, orders = _material_patterns(material, plan.velocity_map, playback)
    return ITModule(
        name=plan.instrument_id[:25],
        samples=samples,
        instruments=(instrument,),
        patterns=patterns,
        orders=orders,
        playback=playback,
    )
