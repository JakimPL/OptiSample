"""Serialize an optimized plan and its stored samples to the JSON the dump tree exposes.

Pure formatting: coerce numpy/non-finite values to JSON-safe Python, and lay a plan out as one
document. The two strategies share the whole per-item encoding block and differ only in their leading
fields (a pitch vs. a zone) and whether a ``method`` is recorded, so :func:`plan_json` builds both from
one shape -- keeping the ``metrics.json``/``plan.objective`` guarantee reading from a single serializer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from optisample.dsp.loop import Loop
from optisample.music import note_name
from optisample.optimize.operating_points import OperatingPoint
from optisample.optimize.plans import GroupedInstrumentPlan, InstrumentPlan, PitchPlan, Zone, ZoneOption
from optisample.optimize.velocity_map import VelocityVolumeMap


def _json_safe(value: Any) -> Any:
    """Recursively coerce numpy scalars to Python and non-finite floats (silence -> -inf) to null."""
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _velocity_map_json(velocity_map: VelocityVolumeMap) -> dict[str, Any]:
    reference_volume = max((anchor.volume for anchor in velocity_map.anchors), default=0)
    return {
        "reference_volume": reference_volume,
        "anchors": [
            {"velocity": anchor.velocity, "loudness_lufs": anchor.loudness_lufs, "volume": anchor.volume}
            for anchor in velocity_map.anchors
        ],
        "volumes": list(velocity_map.volumes),
    }


def _loop_json(loop: Loop | None) -> dict[str, int] | None:
    """The loop actually stored (``{start, end}``), or ``None`` when the sample was not looped."""
    return None if loop is None else {"start": loop.start, "end": loop.end}


def _budget_json(plan: InstrumentPlan | GroupedInstrumentPlan) -> dict[str, int]:
    return {
        "module_budget_bytes": plan.module_budget_bytes,
        "sample_budget_bytes": plan.sample_budget_bytes,
        "used_bytes": plan.used_bytes,
        "module_bytes": plan.module_bytes,
    }


def _encoding_json(chosen: OperatingPoint | ZoneOption, hull_size: int, loop: Loop | None) -> dict[str, Any]:
    """The stored-encoding block both strategies share: chosen params, geometry, cost and hull size.

    ``loop`` is the loop *actually stored* after re-encoding (not merely the one the sweep requested).
    """
    return {
        "target_rate": chosen.params.target_rate,
        "depth_bits": chosen.params.depth_bits,
        "trim_s": chosen.params.trim_s,
        "loop": _loop_json(loop),
        "frames": chosen.frames,
        "stored_bytes": chosen.stored_bytes,
        "distortion": chosen.distortion,
        "hull_size": hull_size,
    }


def _pitch_item(pitch: PitchPlan, loop: Loop | None) -> dict[str, Any]:
    return {
        "pitch": pitch.pitch,
        "note": note_name(pitch.pitch),
        "weight": pitch.weight,
        "representative_velocity": pitch.representative_velocity,
        **_encoding_json(pitch.chosen, len(pitch.hull), loop),
    }


def _zone_item(zone: Zone, loop: Loop | None) -> dict[str, Any]:
    return {
        "keys": [zone.pitches[0], zone.pitches[-1]],
        "pitches": list(zone.pitches),
        "representative": zone.representative,
        "representative_velocity": zone.representative_velocity,
        "weight": zone.weight,
        **_encoding_json(zone.chosen, len(zone.hull), loop),
    }


def plan_json(plan: InstrumentPlan | GroupedInstrumentPlan, loops: Sequence[Loop | None]) -> dict[str, Any]:
    """One plan document for either strategy; ``loops`` are the per-item *stored* loops, in plan order."""
    if isinstance(plan, GroupedInstrumentPlan):
        head: dict[str, Any] = {"strategy": "grouped", "instrument_id": plan.instrument_id}
        items = {"zones": [_zone_item(zone, loop) for zone, loop in zip(plan.zones, loops)]}
    else:
        head = {"strategy": "ungrouped", "instrument_id": plan.instrument_id, "method": plan.method}
        items = {"pitches": [_pitch_item(pitch, loop) for pitch, loop in zip(plan.pitches, loops)]}
    return {
        **head,
        "objective": plan.objective,
        "budget": _budget_json(plan),
        "velocity_map": _velocity_map_json(plan.velocity_map),
        **items,
    }
