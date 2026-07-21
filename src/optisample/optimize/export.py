"""Turn an optimized plan into a playable :class:`ITModule`.

This is the bridge from the optimizer's decisions to the hand-rolled IT writer. For each stored sample
the solver kept, we re-encode its representative recording with the chosen params (deterministically, so
the byte count matches the plan exactly) and route the keys it serves to it through the note map. A key
``p`` plays its sample at its natural rate when

    C5Speed = stored_rate * 2**((IT_C5_NOTE - p) / 12)   (IT_C5_NOTE = C-5, IT's reference key).

The material becomes a sequence of patterns: each event triggers its pitch with the velocity->volume
map applied to the volume column, held for the event's duration, then cut. The two strategies differ
only in the units a plan reports (:meth:`~optisample.optimize.plans.StrategyPlan.sample_units`):
ungrouped stores one sample per key (an identity note map), grouped stores one repitched sample per zone
(every covered key routed to it). Both feed the same build loop and pattern assembly below, so this
module reads the plan through :class:`~optisample.optimize.plans.StrategyPlan` and never branches on it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.dsp import EncodeConfig
from optisample.config.render import PlaybackConfig
from optisample.dsp.surrogate import EncodeContext, StoredSample, encode
from optisample.io.it_writer import (
    IT_C5_NOTE,
    MAX_ROWS,
    NAME_BYTES,
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
from optisample.optimize.plans import SampleUnit, StrategyPlan
from optisample.optimize.tasks import AudioMap
from optisample.optimize.velocity_map import VelocityVolumeMap

_NAME_MAX_CHARS: Final = NAME_BYTES - 1  # leave the final byte of every 26-B name field as a null terminator.
_SAMPLE_LABEL_CHARS: Final = 18  # instrument-id chars kept before the " <note>" suffix in a sample name.
DEFAULT_SEED: Final = 0  # default dither seed; re-encoding a plan with it reproduces the exact budgeted bytes.


@dataclass(frozen=True)
class ExportContext:
    """Config the IT exporter needs beyond a plan: how to re-encode each sample and how it plays back.

    ``seed`` seeds the per-sample dither so re-encoding a plan reproduces the exact bytes it budgeted.
    """

    encode: EncodeConfig
    playback: PlaybackConfig
    seed: int = DEFAULT_SEED


def c5speed_for_pitch(stored_rate: int, pitch: int) -> int:
    """C5Speed that makes key ``pitch`` play the sample (stored at ``stored_rate``) at its natural rate."""
    return int(round(stored_rate * semitone_ratio(IT_C5_NOTE - pitch)))


def row_seconds(playback: ITPlayback) -> float:
    """Seconds one pattern row lasts at ``playback``'s speed/tempo (``speed`` ticks, each 2.5/tempo seconds)."""
    return playback.speed * TICKS_PER_ROW_BASE / playback.tempo


def _it_loop(stored: StoredSample) -> tuple[int, int] | None:
    """The stored sample's loop as the ``(begin, end)`` frame pair the IT writer expects (or ``None``)."""
    return None if stored.loop is None else (stored.loop.start, stored.loop.end)


def encode_plan_units(
    units: Sequence[SampleUnit],
    audio: AudioMap,
    sample_rate: int,
    encode_config: EncodeConfig,
    seed: int,
) -> Iterator[tuple[SampleUnit, StoredSample]]:
    """Re-encode each unit's representative recording in plan order from one seeded RNG.

    The exporter and the artifact dumper share this loop so the decoded PCM stays byte-identical between
    ``module.it`` and the inspection WAVs. One RNG advances once per unit in iteration order, so every
    stored sample's dither is reproducible from ``seed``.
    """
    rng = np.random.default_rng(seed)
    for unit in units:
        representative: Signal = audio[(unit.representative, unit.representative_velocity)]
        encode_ctx = EncodeContext(root_pitch=unit.representative, config=encode_config, rng=rng)
        yield unit, encode(representative, sample_rate, unit.params, encode_ctx)


def _build_unit_samples(
    instrument_id: str, units: Sequence[SampleUnit], audio: AudioMap, sample_rate: int, ctx: ExportContext
) -> tuple[tuple[ITSample, ...], dict[int, int]]:
    """Re-encode each unit's representative and map every key it serves to the resulting sample.

    Units are encoded in order from one seeded RNG, so the byte layout reproduces the plan exactly;
    the returned assignment is 1-based, matching how the IT note map numbers samples.
    """
    samples: list[ITSample] = []
    assignment: dict[int, int] = {}
    for index, (unit, stored) in enumerate(encode_plan_units(units, audio, sample_rate, ctx.encode, ctx.seed)):
        samples.append(
            ITSample(
                name=f"{instrument_id[:_SAMPLE_LABEL_CHARS]} {note_name(unit.representative)}",
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


def _event_rows(duration_s: float, seconds_per_row: float) -> int:
    """A note's length in pattern rows: at least one row, capped so the note plus its cut fit a pattern."""
    return min(max(1, int(round(duration_s / seconds_per_row))), MAX_ROWS - 2)


def _material_patterns(
    material: Sequence[NoteEvent], velocity_map: VelocityVolumeMap, playback: ITPlayback
) -> tuple[tuple[ITPattern, ...], tuple[int, ...]]:
    """Lay the material events into one or more patterns, applying the velocity->volume map.

    Each note writes a note-on cell (its pitch, the instrument, and its velocity mapped to a volume) at
    the current row and a note-cut cell one row past its length, so it occupies its duration in rows plus
    one. Notes fill rows back to back; when the next note would overflow ``MAX_ROWS`` the current pattern
    is flushed and a fresh one begins, so a long piece spans several patterns played in order.
    """
    seconds_per_row = row_seconds(playback)
    patterns: list[ITPattern] = []
    cells: list[tuple[int, int, ITCell]] = []
    cursor = 0
    for event in material:
        rows = _event_rows(event.duration_s, seconds_per_row)
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
    plan: StrategyPlan,
    samples: tuple[ITSample, ...],
    assignment: dict[int, int],
    material: Sequence[NoteEvent],
    ctx: ExportContext,
) -> ITModule:
    """Wire pre-built samples into a module: the identity note map plus the material patterns.

    Everything strategy-specific is already resolved into ``samples``/``assignment``; the module name,
    instrument name and pattern wiring are identical for both, so they live here once.
    """
    instrument = ITInstrument(name=plan.instrument_id[:_NAME_MAX_CHARS], note_map=identity_note_map(assignment))
    playback = it_playback(ctx.playback)
    patterns, orders = _material_patterns(material, plan.velocity_map, playback)
    return ITModule(
        name=plan.instrument_id[:_NAME_MAX_CHARS],
        samples=samples,
        instruments=(instrument,),
        patterns=patterns,
        orders=orders,
        playback=playback,
    )


def build_module(
    plan: StrategyPlan,
    audio: AudioMap,
    sample_rate: int,
    material: Sequence[NoteEvent],
    ctx: ExportContext,
) -> ITModule:
    """Assemble a complete :class:`ITModule` from either strategy's plan.

    The plan's :meth:`~optisample.optimize.plans.StrategyPlan.sample_units` reports the stored samples --
    one per key (ungrouped) or one per zone (grouped) -- and the build below is identical for both.
    """
    samples, assignment = _build_unit_samples(plan.instrument_id, plan.sample_units(), audio, sample_rate, ctx)
    return _assemble_module(plan, samples, assignment, material, ctx)
