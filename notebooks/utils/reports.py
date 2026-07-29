from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from notebooks.utils.views import Row
from optisample.artifacts.paths import PlanPaths, plan_paths, reduced_paths
from optisample.artifacts.serialize import (
    EncodingRecord,
    MetricsDocument,
    PitchItemRecord,
    PlanDocument,
    ReducedDocument,
    ReductionDocument,
    StoredFormatRecord,
    ZoneItemRecord,
)
from optisample.metrics import bytes_to_kib
from optisample.music import labelled_pitch, note_name
from optisample.optimize.plans import FIRST_LAYER

_STRATEGIES: Final = ("ungrouped", "grouped")
_REFERENCE_STEM: Final = "reference"
_REFERENCE_TAIL: Final = "_ref.wav"
_ON: Final = "on"
_OFF: Final = "-"
_PERCENT: Final = 100.0
_HZ_PER_KHZ: Final = 1000.0


def has_reduction(root: Path, instrument_id: str) -> bool:
    """Whether a reduce run has left its document under ``root``, so the explorers have something to read."""
    return reduced_paths(root, instrument_id).reduction_json.is_file()


def read_reduced(root: Path, instrument_id: str) -> ReducedDocument:
    """The document a reduce run left beside the dataset it wrote under ``root``."""
    return ReducedDocument.model_validate_json(
        reduced_paths(root, instrument_id).reduction_json.read_text(encoding="utf-8")
    )


def read_plan(paths: PlanPaths) -> PlanDocument:
    """One strategy's plan: budgets, the velocity map, and the encoding chosen for every kept item."""
    return PlanDocument.model_validate_json(paths.plan_json.read_text(encoding="utf-8"))


def read_metrics(paths: PlanPaths) -> MetricsDocument:
    """One strategy's per-note surrogate fidelity, summing back to the plan's objective."""
    return MetricsDocument.model_validate_json(paths.metrics_json.read_text(encoding="utf-8"))


def available_strategies(instrument_dir: Path) -> list[str]:
    """The strategies whose plan an allocation left under ``instrument_dir``, in the order they run."""
    return [strategy for strategy in _STRATEGIES if plan_paths(instrument_dir, strategy).plan_json.is_file()]


# --- what the pre-optimization stage decided ---------------------------------------------------------


def reduction_rows(reduction: ReductionDocument) -> list[Row]:
    """Each reduction axis as one row reading before -> after, which is the stage's whole outcome."""
    per_key = len(reduction.grids) or 1
    return [
        {"axis": "recordings", "before": reduction.listed_recordings, "after": reduction.kept_recordings},
        {"axis": "played notes", "before": reduction.played_notes, "after": reduction.scored_classes},
        {
            "axis": "stored rate (kHz)",
            "before": round(sum(grid.useful_rate_hz for grid in reduction.grids) / per_key / _HZ_PER_KHZ, 1),
            "after": round(sum(grid.stored.target_rate for grid in reduction.grids) / per_key / _HZ_PER_KHZ, 1),
        },
    ]


def recording_rows(reduction: ReductionDocument) -> list[Row]:
    """One row per surviving recording, measured against the material its pitch asks of it."""
    return [
        {
            "key": recording.key,
            "pitch": recording.pitch,
            "velocity": recording.velocity,
            "dur_s": round(recording.duration_s, 3),
            "required_s": round(recording.required_duration_s, 3),
            "covers": recording.covers_material,
            "shortfall_s": round(max(0.0, recording.required_duration_s - recording.duration_s), 3),
        }
        for recording in reduction.recordings
    ]


def _format_label(stored: StoredFormatRecord) -> str:
    """The settled format as a table cell reads it: stored rate, depth, and whether it is compressed."""
    marks = "c" if stored.compress else ""
    return f"{stored.target_rate // 1000}k/{stored.depth_bits}{marks}"


def stored_format_rows(reduction: ReductionDocument) -> list[Row]:
    """One row per played pitch: the band it asked for, the format it stores at, and what is swept over it."""
    return [
        {
            "pitch": grid.pitch,
            "note": grid.note,
            "useful_rate_hz": round(grid.useful_rate_hz),
            "stored": _format_label(grid.stored),
            "swept": grid.swept,
        }
        for grid in reduction.grids
    ]


def loop_rows(reduction: ReductionDocument) -> list[Row]:
    """One row per loop candidate the sweep may store a pitch around, with what each is worth.

    ``seam`` counts the wrap's jump in the loop's own frame-to-frame steps and ``timbre_db`` the distance
    between the loop's spectrum and the material past it, so a row states the case for the loop the
    allocation went on to buy.
    """
    return [
        {
            "pitch": grid.pitch,
            "note": grid.note,
            "choice": loop.choice,
            "start_s": round(loop.start_s, 3),
            "end_s": round(loop.end_s, 3),
            "length_s": round(loop.end_s - loop.start_s, 3),
            "seam": round(loop.seam_step, 2),
            "timbre_db": round(loop.spectral_distance, 2),
        }
        for grid in reduction.grids
        for loop in grid.loops
    ]


def survivor_rows(document: ReducedDocument) -> list[Row]:
    """One row per WAV a reduce run wrote, as the reduced dataset holds it."""
    return [
        {
            "index": sample.index,
            "key": sample.key,
            "file": sample.file,
            "frames": sample.frames,
            "dur_s": round(sample.duration_s, 3),
        }
        for sample in document.samples
    ]


# --- what the allocation bought ----------------------------------------------------------------------


def _encoding_cells(encoding: EncodingRecord) -> Row:
    """The stored-encoding block every plan item carries, as the cells a table shows it through."""
    return {
        "rate_hz": encoding.target_rate,
        "depth": encoding.depth_bits,
        "comp": _ON if encoding.compress else _OFF,
        "loop": _ON if encoding.loop is not None else _OFF,
        "decay_to": _OFF if encoding.decay is None else round(encoding.decay.final_gain, 3),
        "frames": encoding.frames,
        "kib": round(bytes_to_kib(encoding.stored_bytes), 3),
        "distortion": round(encoding.distortion, 4),
        "hull": encoding.hull_size,
    }


def instrument_rows(plan: PlanDocument) -> list[Row]:
    """One row per written instrument: the dynamics and keys it answers for, and what it cost.

    A key played across several dynamics is stored once per band it is played in, so this is where a
    plan's vocabulary is read: how the budget was divided between velocity resolution and everything
    else. A plan keeping one recording per key in one band reports the single instrument it is written
    as, and a band a format writes as several instruments reports the stretch of keyboard each holds.
    """
    return [
        {
            "id": instrument.index,
            "name": instrument.name,
            "band": instrument.band,
            "keys": instrument.keys,
            "samples": instrument.samples,
            "kib": round(bytes_to_kib(instrument.stored_bytes), 3),
            "weight_s": round(instrument.weight, 2),
            "objective": round(instrument.objective_share, 4),
        }
        for instrument in plan.instruments
    ]


def _bands_by_layer(plan: PlanDocument) -> dict[int, str]:
    """The velocity band each layer of the split answers for, read off the instruments written for it."""
    return {instrument.layer: instrument.band for instrument in plan.instruments}


def _pitch_row(item: PitchItemRecord, band: str) -> Row:
    """One kept pitch, holding the single key it was recorded at."""
    return {
        "band": band,
        "keys": str(item.pitch),
        "span": 1,
        "note": item.note,
        "rep_vel": item.representative_velocity,
        "weight_s": round(item.weight, 2),
        **_encoding_cells(item),
    }


def _zone_row(item: ZoneItemRecord, band: str) -> Row:
    """One pitch zone, holding every key its representative is transposed across."""
    return {
        "band": band,
        "keys": f"{item.keys[0]}-{item.keys[-1]}",
        "span": len(item.pitches),
        "note": note_name(item.representative),
        "rep_vel": item.representative_velocity,
        "weight_s": round(item.weight, 2),
        **_encoding_cells(item),
    }


def plan_item_rows(plan: PlanDocument) -> list[Row]:
    """One row per item the plan kept: the keys it serves, and the encoding it spends its bytes on.

    Both strategies read through the same cells, so the two plans line up column for column and a
    budget moved between them stays readable. Each row opens with the velocity band it was stored for,
    which is what tells the several rows a layered plan keeps for one key apart.
    """
    bands = _bands_by_layer(plan)
    if plan.zones is not None:
        return [_zone_row(zone, bands[zone.layer]) for zone in plan.zones]

    return [_pitch_row(pitch, bands[FIRST_LAYER]) for pitch in plan.pitches or []]


def budget_rows(plan: PlanDocument) -> list[Row]:
    """The plan's byte accounting beside what the module it exports to actually occupies."""
    items = plan.zones if plan.zones is not None else plan.pitches or []
    return [
        {
            "strategy": plan.strategy,
            "objective": round(plan.objective, 4),
            "items": len(items),
            "sample_budget_kib": round(bytes_to_kib(plan.budget.sample_budget_bytes), 1),
            "used_kib": round(bytes_to_kib(plan.budget.used_bytes), 1),
            "spent_pct": round(_PERCENT * plan.budget.used_bytes / plan.budget.sample_budget_bytes, 1),
            "module_kib": round(bytes_to_kib(plan.module.total_bytes), 1),
            "pcm_kib": round(bytes_to_kib(plan.module.pcm_bytes), 1),
        }
    ]


@dataclass(frozen=True, order=True)
class ComparedNote:
    """One A/B pair an allocation wrote: the velocity layer that played the note, and the note itself.

    Ordering is by layer and then by ``stem``, which leads with the zero-padded MIDI pitch, so a listing
    walks each layer's keyboard in order.
    """

    layer: str
    stem: str

    @property
    def pitch(self) -> int:
        """The MIDI pitch this pair was written for, which leads the stem it was filed under."""
        return labelled_pitch(self.stem)

    @property
    def label(self) -> str:
        """How a selector names this pair: the layer's velocity band, then the note."""
        return f"{self.layer} {self.stem}"


def note_metric_rows(metrics: MetricsDocument) -> list[Row]:
    """One row per covered pitch of each velocity layer, and the share of the objective it carries."""
    return [
        {
            "pitch": note.pitch,
            "note": note.note,
            "layer": note.layer,
            "served_by": note.served_by,
            "weight_s": round(note.weight, 2),
            "mean_distortion": round(note.mean_distortion, 4),
            "objective": round(note.objective_contribution, 4),
            "classes": len(note.events),
            "render": note.render_source,
        }
        for note in metrics.notes
    ]


def _measurement(value: float | None) -> float | str:
    """A measurement as a table cell, naming the readings that had no finite value behind them."""
    return "n/a" if value is None else round(value, 4)


def event_rows(metrics: MetricsDocument, compared: ComparedNote) -> list[Row]:
    """Every scored note class one A/B pair covers, with the sub-scores its fidelity is composed of.

    A key played across several dynamics is reconstructed once per velocity layer, so the pair names the
    layer as well as the pitch and the rows are the classes that layer's own sample answered for.
    """
    return [
        {
            "velocity": event.velocity,
            "dur_s": round(event.duration_s, 3),
            "weight_s": round(event.weight, 2),
            "volume": event.volume,
            "fidelity": round(event.fidelity, 4),
            **{name: _measurement(value) for name, value in event.breakdown.items()},
            **{name: _measurement(value) for name, value in event.diagnostics.items()},
        }
        for note in metrics.notes
        if (note.layer, note.pitch) == (compared.layer, compared.pitch)
        for event in note.events
    ]


# --- the audio each stage left behind ----------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """One WAV an inspection folder holds: what it is, and where it sits."""

    label: str
    path: Path


def audition_pitches(root: Path, instrument_id: str) -> list[str]:
    """The pitch folders a reduce run filled with auditions, in keyboard order."""
    auditions_dir = reduced_paths(root, instrument_id).auditions_dir
    if not auditions_dir.is_dir():
        return []

    return sorted(folder.name for folder in auditions_dir.iterdir() if folder.is_dir())


def auditions(root: Path, instrument_id: str, pitch: str) -> list[Clip]:
    """The recording one pitch was judged against, followed by every encoding swept for it."""
    folder = reduced_paths(root, instrument_id).auditions_dir / pitch
    files = sorted(folder.glob("*.wav"), key=lambda wav: (wav.stem != _REFERENCE_STEM, wav.stem))
    return [Clip(label=wav.stem, path=wav) for wav in files]


def compared_notes(paths: PlanPaths) -> list[ComparedNote]:
    """Every A/B pair an allocation wrote, by velocity layer and then in keyboard order."""
    if not paths.compare_dir.is_dir():
        return []

    return sorted(
        ComparedNote(layer=wav.parent.name, stem=wav.name.removesuffix(_REFERENCE_TAIL))
        for wav in paths.compare_dir.glob(f"*/*{_REFERENCE_TAIL}")
    )


def comparison(paths: PlanPaths, compared: ComparedNote) -> tuple[Path, Path]:
    """One pair: the recording as scored, beside what the module produces for it through that layer."""
    return paths.reference_wav(compared.layer, compared.stem), paths.rendered_wav(compared.layer, compared.stem)


def stored_samples(paths: PlanPaths) -> list[Clip]:
    """Every sample the plan stores, decoded back to a WAV bit-identical to the module's own."""
    if not paths.samples_dir.is_dir():
        return []

    return [Clip(label=wav.stem, path=wav) for wav in sorted(paths.samples_dir.glob("*.wav"))]


def module_render(paths: PlanPaths) -> Path | None:
    """The whole module rendered through openmpt123, when the run asked for one and could have it."""
    return paths.module_render if paths.module_render.is_file() else None


def report_text(paths: PlanPaths) -> str:
    """The human-readable report a strategy opens with, as written."""
    return paths.report.read_text(encoding="utf-8")


def infeasible_reason(paths: PlanPaths) -> str | None:
    """Why a strategy left no plan, when the budget afforded no allocation at all."""
    if paths.infeasible.is_file():
        return paths.infeasible.read_text(encoding="utf-8")

    return None
