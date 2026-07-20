"""Turn an optimized plan into a playable :class:`ITModule`.

This is the bridge from the optimizer's decisions to the hand-rolled IT writer. For each stored sample
the solver kept, we re-encode its representative recording with the chosen params (deterministically, so
the byte count matches the plan exactly) and route the keys it serves to it through the note map. A key
``p`` plays its sample at its natural rate when

    C5Speed = stored_rate * 2**((60 - p) / 12)   (60 = C-5, IT's reference key).

The material becomes a sequence of patterns: each event triggers its pitch with the velocity->volume
map applied to the volume column, held for the event's duration, then cut. The two strategies differ
only in the *unit list*: ungrouped stores one sample per key (an identity note map), grouped stores one
repitched sample per zone (every covered key routed to it). Both feed the same build loop and pattern
assembly below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.render import PlaybackConfig
from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
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
    require_it_note,
)
from optisample.metrics.base import Signal
from optisample.model import NoteEvent
from optisample.music import note_name, semitone_ratio
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan
from optisample.optimize.tasks import AudioMap
from optisample.optimize.velocity_map import VelocityVolumeMap

_C5_KEY = 60  # IT reference key C-5; a sample plays at C5Speed when triggered here.


@dataclass(frozen=True)
class ExportContext:
    """Config the IT exporter needs beyond a plan: how to re-encode each sample and how it plays back.

    ``seed`` seeds the per-sample dither so re-encoding a plan reproduces the exact bytes it budgeted.
    """

    encode: EncodeConfig
    playback: PlaybackConfig
    seed: int = 0


@dataclass(frozen=True)
class _SampleUnit:
    """One stored sample to build and every key that triggers it.

    ``representative`` is the pitch the recording is rooted at: it is re-encoded here and its ``C5Speed``
    is tuned so that key plays natural. ``keys`` is a single key for the ungrouped strategy and a whole
    zone for grouping, where the tracker repitches the shared sample by ``key - representative`` semitones
    -- exactly the transpose the surrogate scored.
    """

    representative: int
    representative_velocity: int
    params: EncodingParams
    keys: tuple[int, ...]


def c5speed_for_pitch(stored_rate: int, pitch: int) -> int:
    """C5Speed that makes key ``pitch`` play the sample (stored at ``stored_rate``) at its natural rate."""
    return int(round(stored_rate * semitone_ratio(_C5_KEY - pitch)))


def _it_loop(stored: StoredSample) -> tuple[int, int] | None:
    """The stored sample's loop as the ``(begin, end)`` frame pair the IT writer expects (or ``None``)."""
    return None if stored.loop is None else (stored.loop.start, stored.loop.end)


def _plan_units(plan: InstrumentPlan) -> tuple[_SampleUnit, ...]:
    """Ungrouped: one unit per kept pitch, each key owning its own sample (an identity note map)."""
    return tuple(
        _SampleUnit(
            representative=pitch_plan.pitch,
            representative_velocity=pitch_plan.representative_velocity,
            params=pitch_plan.chosen.params,
            keys=(pitch_plan.pitch,),
        )
        for pitch_plan in plan.pitches
    )


def _zone_units(plan: GroupedInstrumentPlan) -> tuple[_SampleUnit, ...]:
    """Grouped: one unit per zone, every key the zone covers routed to its representative's sample."""
    return tuple(
        _SampleUnit(
            representative=zone.representative,
            representative_velocity=zone.representative_velocity,
            params=zone.chosen.params,
            keys=zone.pitches,
        )
        for zone in plan.zones
    )


def _build_unit_samples(
    instrument_id: str, units: Sequence[_SampleUnit], audio: AudioMap, sample_rate: int, ctx: ExportContext
) -> tuple[tuple[ITSample, ...], dict[int, int]]:
    """Re-encode each unit's representative and map every key it serves to the resulting sample.

    Units are encoded in order from one seeded RNG, so the byte layout reproduces the plan exactly;
    the returned assignment is 1-based, matching how the IT note map numbers samples.
    """
    rng = np.random.default_rng(ctx.seed)
    samples: list[ITSample] = []
    assignment: dict[int, int] = {}
    for index, unit in enumerate(units):
        representative: Signal = audio[(unit.representative, unit.representative_velocity)]
        encode_ctx = EncodeContext(root_pitch=unit.representative, config=ctx.encode, rng=rng)
        stored = encode(representative, sample_rate, unit.params, encode_ctx)
        samples.append(
            ITSample(
                name=f"{instrument_id[:18]} {note_name(unit.representative)}",
                pcm=stored.pcm,
                depth_bits=stored.depth_bits,
                c5speed=c5speed_for_pitch(stored.sample_rate, unit.representative),
                loop=_it_loop(stored),
            )
        )
        for key in unit.keys:
            require_it_note(key)
            assignment[key] = index + 1  # sample numbers are 1-based in the note map
    return tuple(samples), assignment


def _event_rows(duration_s: float, row_seconds: float) -> int:
    """A note's length in pattern rows: at least one row, capped so the note plus its cut fit a pattern."""
    return min(max(1, int(round(duration_s / row_seconds))), MAX_ROWS - 2)


def _material_patterns(
    material: Sequence[NoteEvent], velocity_map: VelocityVolumeMap, playback: ITPlayback
) -> tuple[tuple[ITPattern, ...], tuple[int, ...]]:
    """Lay the material events into one or more patterns, applying the velocity->volume map.

    Each note writes a note-on cell (its pitch, the instrument, and its velocity mapped to a volume) at
    the current row and a note-cut cell one row past its length, so it occupies its duration in rows plus
    one. Notes fill rows back to back; when the next note would overflow ``MAX_ROWS`` the current pattern
    is flushed and a fresh one begins, so a long piece spans several patterns played in order.
    """
    row_seconds = playback.speed * TICKS_PER_ROW_BASE / playback.tempo
    patterns: list[ITPattern] = []
    cells: list[tuple[int, int, ITCell]] = []
    cursor = 0
    for event in material:
        rows = _event_rows(event.duration_s, row_seconds)
        if cells and cursor + rows + 1 > MAX_ROWS:
            patterns.append(ITPattern(rows=cursor, cells=tuple(cells)))
            cells, cursor = [], 0
        volume = velocity_map.volume(event.velocity)
        cells.append((cursor, 0, ITCell(note=event.pitch, instrument=1, volume=volume)))
        cells.append((cursor + rows, 0, ITCell(note=NOTE_CUT)))
        cursor += rows + 1
    patterns.append(ITPattern(rows=max(cursor, 1), cells=tuple(cells)))
    return tuple(patterns), tuple(range(len(patterns)))


def _assemble_module(
    plan: InstrumentPlan | GroupedInstrumentPlan,
    samples: tuple[ITSample, ...],
    assignment: dict[int, int],
    material: Sequence[NoteEvent],
    ctx: ExportContext,
) -> ITModule:
    """Wire pre-built samples into a module: the identity note map plus the material patterns.

    Everything strategy-specific is already resolved into ``samples``/``assignment``; the module name,
    instrument name and pattern wiring are identical for both, so they live here once.
    """
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


def build_it_module(
    plan: InstrumentPlan, audio: AudioMap, sample_rate: int, material: Sequence[NoteEvent], ctx: ExportContext
) -> ITModule:
    """Assemble a complete :class:`ITModule` from an ungrouped plan (one sample per kept key)."""
    samples, assignment = _build_unit_samples(plan.instrument_id, _plan_units(plan), audio, sample_rate, ctx)
    return _assemble_module(plan, samples, assignment, material, ctx)


def build_grouped_it_module(
    plan: GroupedInstrumentPlan, audio: AudioMap, sample_rate: int, material: Sequence[NoteEvent], ctx: ExportContext
) -> ITModule:
    """Assemble a complete :class:`ITModule` from a grouped plan (one repitched sample per zone)."""
    samples, assignment = _build_unit_samples(plan.instrument_id, _zone_units(plan), audio, sample_rate, ctx)
    return _assemble_module(plan, samples, assignment, material, ctx)


def build_module(
    plan: InstrumentPlan | GroupedInstrumentPlan,
    audio: AudioMap,
    sample_rate: int,
    material: Sequence[NoteEvent],
    ctx: ExportContext,
) -> ITModule:
    """Assemble the IT module for either strategy, dispatching on the plan type."""
    if isinstance(plan, GroupedInstrumentPlan):
        return build_grouped_it_module(plan, audio, sample_rate, material, ctx)
    return build_it_module(plan, audio, sample_rate, material, ctx)
