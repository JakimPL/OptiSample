from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from notebooks.utils.views import Row
from optisample.artifacts.documents.loops import LoopsDocument, read_loops
from optisample.artifacts.documents.metrics import MetricsDocument
from optisample.artifacts.documents.plan import (
    EncodingRecord,
    PitchItemRecord,
    PlanDocument,
    ZoneItemRecord,
)
from optisample.artifacts.documents.reduction import (
    ReducedDocument,
    ReductionDocument,
    StoredFormatRecord,
)
from optisample.artifacts.paths import PlanPaths, looped_paths, plan_paths, reduced_paths
from optisample.metrics import bytes_to_kib
from optisample.music import labelled_pitch, note_name
from optisample.optimize.plans import FIRST_LAYER

_STRATEGIES: Final = ("ungrouped", "grouped")
_REFERENCE_STEM: Final = "reference"
_RECORDING_STEM: Final = "recording"
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


def has_loops(root: Path, instrument_id: str) -> bool:
    """Whether a loop run has left its document under ``root``, so the loop explorer has something to read."""
    return looped_paths(root, instrument_id).loops_json.is_file()


def read_loop_document(root: Path, instrument_id: str) -> LoopsDocument:
    """The document a loop run left beside the dataset it wrote under ``root``."""
    return read_loops(looped_paths(root, instrument_id).loops_json)


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


def loop_rows(document: LoopsDocument) -> list[Row]:
    """One row per loop a recording offers, with where it sits and what measuring it said.

    ``offer`` is the index an encoding names the loop by, running from the cheapest stored span upward, so
    reading the rows of one key down states the stretch of stored length the sweep gets to price. ``seam``
    counts the wrap's jump in the frame-to-frame motion the waveform makes there, ``drift_db`` the fall
    across the region that holding it at one level flattened, and ``timbre_db`` the distance between the
    loop's spectrum and the material past it, so a row states the case for that one loop. ``rejected``
    counts the candidates the settlement turned down for this recording.
    """
    return [
        {
            "key": record.key,
            "pitch": record.pitch,
            "note": record.note,
            "offer": offer,
            "start_s": round(stored.start_s, 3),
            "end_s": round(stored.end_s, 3),
            "length_s": round(stored.end_s - stored.start_s, 3),
            "seam": round(stored.quality.seam_step, 2),
            "drift_db": round(stored.quality.level_drift_db, 2),
            "timbre_db": round(stored.quality.spectral_distance, 2),
            "rejected": len(record.rejected),
        }
        for record in document.recordings
        for offer, stored in enumerate(record.offered)
    ]


def unlooped_rows(document: LoopsDocument) -> list[Row]:
    """One row per recording stored over the span it plays, with how many candidates were measured for it.

    A recording reaches this table either because its material is too short for the shortest accepted loop,
    which shows as ``tried`` of zero, or because every candidate fell outside a gate.
    """
    return [
        {
            "key": record.key,
            "pitch": record.pitch,
            "note": record.note,
            "search_s": round(record.search_s, 3),
            "tried": len(record.rejected),
        }
        for record in document.recordings
        if not record.offered
    ]


def rejected_loop_rows(document: LoopsDocument) -> list[Row]:
    """One row per candidate the settlement passed over, with the gate it fell outside of.

    Reading these beside :func:`loop_rows` says why a recording offers the loops it does, or none at all,
    which is what retuning the quality gates is read off.
    """
    return [
        {
            "key": record.key,
            "note": record.note,
            "start_s": round(rejected.start_s, 3),
            "end_s": round(rejected.end_s, 3),
            "seam": round(rejected.quality.seam_step, 2),
            "drift_db": round(rejected.quality.level_drift_db, 2),
            "timbre_db": round(rejected.quality.spectral_distance, 2),
            "gate": rejected.gate,
        }
        for record in document.recordings
        for rejected in record.rejected
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
    """The stored-encoding block every plan item carries, as the cells a table shows it through.

    ``loop`` names which of the loops the stage offered this recording the budget bought, counting from
    the cheapest, so reading it beside ``kib`` says how much of a note the plan paid to keep.
    """
    return {
        "rate_hz": encoding.target_rate,
        "depth": encoding.depth_bits,
        "comp": _ON if encoding.compress else _OFF,
        "loop": _OFF if encoding.loop_index is None else encoding.loop_index,
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

    ``drift_db`` reads what the one volume envelope an instrument carries costs the sample of its own it
    suits worst, which is what says whether the keys sharing it want writing as several instruments.
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
            "drift_db": round(instrument.envelope_drift_db, 2),
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


def _audition_folders(auditions_dir: Path) -> list[str]:
    """The folders an auditioning stage left under ``auditions_dir``, in keyboard order."""
    if not auditions_dir.is_dir():
        return []

    return sorted(folder.name for folder in auditions_dir.iterdir() if folder.is_dir())


def audition_pitches(root: Path, instrument_id: str) -> list[str]:
    """The pitch folders a reduce run filled with auditions, in keyboard order."""
    return _audition_folders(reduced_paths(root, instrument_id).auditions_dir)


def loop_audition_keys(root: Path, instrument_id: str) -> list[str]:
    """The recordings a loop run auditioned its settled loop for, in keyboard order."""
    return _audition_folders(looped_paths(root, instrument_id).auditions_dir)


def loop_auditions(root: Path, instrument_id: str, key: str) -> list[Clip]:
    """One recording, followed by the loop settled for it played out against it."""
    folder = looped_paths(root, instrument_id).auditions_dir / key
    files = sorted(folder.glob("*.wav"), key=lambda wav: wav.stem != _RECORDING_STEM)
    return [Clip(label=wav.stem, path=wav) for wav in files]


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
