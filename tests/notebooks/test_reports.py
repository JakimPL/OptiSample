from pathlib import Path

import numpy as np
import pytest

from notebooks.utils import reports
from notebooks.utils.reports import ComparedNote
from optisample.artifacts.paths import plan_paths, reduced_paths
from optisample.artifacts.serialize import (
    BudgetRecord,
    DecayRecord,
    EventMetricRecord,
    InstrumentRecord,
    KeptRecordingRecord,
    KeyboardRecord,
    LoopCandidateRecord,
    LoopRecord,
    MetricsDocument,
    ModuleSizeRecord,
    NarrowedGridRecord,
    NoteMetricRecord,
    PitchItemRecord,
    PlanDocument,
    ReducedDocument,
    ReductionDocument,
    RepresentativeEventRecord,
    ScreenRecord,
    ShortlistedEncodingRecord,
    VelocityMapDocument,
    WrittenSampleRecord,
    ZoneItemRecord,
    write_json,
)
from optisample.io.audio import write_wav

SR = 8_000
_INSTRUMENT = "Piano"
_LAYER = "v000-v127"  # the one band an unlayered plan writes, which is the folder its pairs land in
_ENCODING = {
    "target_rate": 11_025,
    "depth_bits": 8,
    "compress": True,
    "trim_s": 1.5,
    "loop": LoopRecord(start=100, end=900),
    "decay": DecayRecord(start_s=0.11, end_s=1.5, final_gain=0.2),
    "frames": 16_538,
    "stored_bytes": 16_538,
    "distortion": 2.5,
    "hull_size": 3,
}


def _reduction() -> ReductionDocument:
    return ReductionDocument(
        listed_recordings=8,
        kept_recordings=3,
        played_notes=12,
        scored_classes=5,
        grid_size=18,
        recordings=[
            KeptRecordingRecord(
                key="p060_C4_v100",
                pitch=60,
                velocity=100,
                duration_s=2.0,
                required_duration_s=1.5,
                covers_material=True,
            ),
            KeptRecordingRecord(
                key="p067_G4_v080",
                pitch=67,
                velocity=80,
                duration_s=0.5,
                required_duration_s=2.0,
                covers_material=False,
            ),
        ],
        grids=[
            NarrowedGridRecord(
                pitch=60,
                note="C4",
                useful_rate_hz=12_345.6,
                shortlist=[
                    ShortlistedEncodingRecord(target_rate=22_050, depth_bits=16, compress=False, loop_choice=0),
                    ShortlistedEncodingRecord(target_rate=11_025, depth_bits=8, compress=True, loop_choice=1),
                ],
                loops=[
                    LoopCandidateRecord(choice=0, start_s=0.05, end_s=0.55, seam_step=1.25, spectral_distance=3.5),
                    LoopCandidateRecord(choice=1, start_s=0.70, end_s=1.20, seam_step=0.75, spectral_distance=1.5),
                ],
            ),
            NarrowedGridRecord(
                pitch=67,
                note="G4",
                useful_rate_hz=9_000.0,
                shortlist=[
                    ShortlistedEncodingRecord(target_rate=11_025, depth_bits=8, compress=False, loop_choice=None)
                ],
                loops=[],
            ),
        ],
    )


def _plan(strategy: str) -> PlanDocument:
    shared = {
        "instrument_id": _INSTRUMENT,
        "objective": 12.3456,
        "energy_exponent": 0.5,
        "budget": BudgetRecord(
            module_budget_bytes=98_304, sample_budget_bytes=97_000, used_bytes=48_500, module_bytes=49_000
        ),
        "module": ModuleSizeRecord(total_bytes=50_000, header_bytes=800, pcm_bytes=48_500, pattern_bytes=700),
        "keyboard": KeyboardRecord(numbered=120, played=7, answered=120),
        "reduction": _reduction(),
        "velocity_map": VelocityMapDocument(reference_volume=64, anchors=[], volumes=[64] * 128),
        "instruments": [
            InstrumentRecord(
                index=0,
                name=_INSTRUMENT,
                layer=0,
                band=_LAYER,
                lowest_velocity=0,
                highest_velocity=127,
                lowest_pitch=60,
                highest_pitch=66,
                keys=7,
                samples=1,
                stored_bytes=48_500,
                weight=4.0,
                objective_share=12.3456,
            )
        ],
    }
    if strategy == "grouped":
        return PlanDocument(
            strategy="grouped",
            zones=[
                ZoneItemRecord(
                    layer=0,
                    keys=[60, 66],
                    pitches=list(range(60, 67)),
                    representative=63,
                    representative_velocity=100,
                    weight=4.0,
                    **_ENCODING,
                )
            ],
            **shared,
        )
    return PlanDocument(
        strategy="ungrouped",
        method="exact",
        pitches=[PitchItemRecord(pitch=60, note="C4", weight=2.0, representative_velocity=100, **_ENCODING)],
        **shared,
    )


def _metrics() -> MetricsDocument:
    return MetricsDocument(
        strategy="ungrouped",
        instrument_id=_INSTRUMENT,
        sample_rate=SR,
        objective=12.3456,
        plan_objective=12.3456,
        notes=[
            NoteMetricRecord(
                pitch=60,
                note="C4",
                layer=_LAYER,
                served_by="p060_C4",
                representative=60,
                weight=2.0,
                mean_distortion=1.25,
                objective_contribution=2.5,
                render_source="surrogate",
                render_rate=SR,
                representative_event=RepresentativeEventRecord(velocity=100, duration_s=1.5),
                events=[
                    EventMetricRecord(
                        velocity=100,
                        duration_s=1.5,
                        weight=2.0,
                        volume=64,
                        fidelity=1.25,
                        breakdown={"mrstft": 0.5},
                        diagnostics={"snr_db": 20.0, "loudness_delta_lu": float("-inf")},
                    )
                ],
            )
        ],
    )


@pytest.fixture
def reduced_root(tmp_path: Path) -> Path:
    """A reduce run's output tree: its document, and one pitch's auditions as real WAVs."""
    root = tmp_path / "reduced"
    paths = reduced_paths(root, _INSTRUMENT)
    paths.reduction_json.parent.mkdir(parents=True)
    write_json(
        paths.reduction_json,
        ReducedDocument(
            instrument_id=_INSTRUMENT,
            dedupe_key="pitch_velocity",
            sample_rate=SR,
            samples=[
                WrittenSampleRecord(
                    index=0, key="p060_C4_v100", file="0000_p060_C4_v100.wav", frames=16_000, duration_s=2.0
                )
            ],
            screen=ScreenRecord(silenced=[], unplayable=[], dropped_notes=0),
            reduction=_reduction(),
        ),
    )
    folder = paths.auditions_dir / "p060_C4"
    folder.mkdir(parents=True)
    for stem in ("reference", "r11025_d8_c", "r22050_d16"):
        write_wav(folder / f"{stem}.wav", np.zeros(SR, dtype=np.float64), SR)

    return root


@pytest.fixture
def instrument_dir(tmp_path: Path) -> Path:
    """An allocation's output tree for one instrument: both strategies, with ungrouped fully dumped."""
    root = tmp_path / "artifacts" / _INSTRUMENT
    for strategy in ("ungrouped", "grouped"):
        paths = plan_paths(root, strategy)
        paths.directory.mkdir(parents=True)
        write_json(paths.plan_json, _plan(strategy))

    paths = plan_paths(root, "ungrouped")
    write_json(paths.metrics_json, _metrics())
    paths.report.write_text("Instrument 'Piano'\n", encoding="utf-8")
    paths.layer_dir(_LAYER).mkdir(parents=True)
    for stem in ("p060_C4_ref", "p060_C4_render"):
        write_wav(paths.layer_dir(_LAYER) / f"{stem}.wav", np.zeros(SR, dtype=np.float64), SR)

    return root


# --- reading the pre-optimization stage back ----------------------------------------------------------


def test_the_reduction_document_reads_back_as_it_was_written(reduced_root: Path) -> None:
    document = reports.read_reduced(reduced_root, _INSTRUMENT)

    assert (document.instrument_id, document.dedupe_key, document.sample_rate) == (_INSTRUMENT, "pitch_velocity", SR)
    assert reports.survivor_rows(document)[0]["file"] == "0000_p060_C4_v100.wav"


def test_each_reduction_axis_reads_before_and_after(reduced_root: Path) -> None:
    rows = reports.reduction_rows(reports.read_reduced(reduced_root, _INSTRUMENT).reduction)

    assert [(row["before"], row["after"]) for row in rows] == [(8, 3), (12, 5), (18, 1.5)]


def test_a_recording_shorter_than_its_material_states_the_shortfall(reduced_root: Path) -> None:
    rows = reports.recording_rows(reports.read_reduced(reduced_root, _INSTRUMENT).reduction)

    assert [row["covers"] for row in rows] == [True, False]
    assert [row["shortfall_s"] for row in rows] == [0.0, 1.5]


def test_the_shortlist_names_every_axis_each_encoding_asks_for(reduced_root: Path) -> None:
    rows = reports.shortlist_rows(reports.read_reduced(reduced_root, _INSTRUMENT).reduction)

    assert rows[0]["encodings"] == "22k/16/l0 11k/8c/l1"  # c marks compression, l<n> the loop candidate
    assert rows[1]["encodings"] == "11k/8/t"  # t marks the trimmed sample, stored around no loop
    assert rows[0]["useful_rate_hz"] == 12_346


def test_every_loop_a_pitch_may_be_stored_around_is_reported_with_what_it_costs(reduced_root: Path) -> None:
    rows = reports.loop_rows(reports.read_reduced(reduced_root, _INSTRUMENT).reduction)

    assert [row["pitch"] for row in rows] == [60, 60]  # the pitch offering no candidate contributes no row
    assert [row["choice"] for row in rows] == [0, 1]
    assert rows[0]["length_s"] == 0.5
    assert (rows[0]["seam"], rows[0]["timbre_db"]) == (1.25, 3.5)


def test_auditions_open_with_the_recording_the_rest_are_judged_against(reduced_root: Path) -> None:
    assert reports.audition_pitches(reduced_root, _INSTRUMENT) == ["p060_C4"]
    clips = reports.auditions(reduced_root, _INSTRUMENT, "p060_C4")
    assert [clip.label for clip in clips] == ["reference", "r11025_d8_c", "r22050_d16"]
    assert all(clip.path.is_file() for clip in clips)


def test_a_root_no_reduce_run_touched_offers_no_auditions(tmp_path: Path) -> None:
    assert reports.audition_pitches(tmp_path, _INSTRUMENT) == []


# --- reading the allocation back -----------------------------------------------------------------------


def test_only_the_strategies_that_left_a_plan_are_offered(instrument_dir: Path, tmp_path: Path) -> None:
    assert reports.available_strategies(instrument_dir) == ["ungrouped", "grouped"]
    assert reports.available_strategies(tmp_path / "nothing") == []


def test_an_ungrouped_item_holds_the_one_pitch_it_was_recorded_at(instrument_dir: Path) -> None:
    rows = reports.plan_item_rows(reports.read_plan(plan_paths(instrument_dir, "ungrouped")))

    assert (rows[0]["keys"], rows[0]["span"], rows[0]["note"]) == ("60", 1, "C4")
    assert (rows[0]["rate_hz"], rows[0]["depth"], rows[0]["comp"], rows[0]["loop"]) == (11_025, 8, "on", "on")
    assert rows[0]["decay_to"] == 0.2  # the share of the loop's level the note is played down to


def test_a_grouped_item_holds_the_whole_zone_its_representative_serves(instrument_dir: Path) -> None:
    rows = reports.plan_item_rows(reports.read_plan(plan_paths(instrument_dir, "grouped")))

    assert (rows[0]["keys"], rows[0]["span"], rows[0]["note"]) == ("60-66", 7, "D#4")


def test_both_strategies_read_through_the_same_cells(instrument_dir: Path) -> None:
    """One table shape for both plans, so a budget moved between them stays readable column for column."""
    ungrouped = reports.plan_item_rows(reports.read_plan(plan_paths(instrument_dir, "ungrouped")))
    grouped = reports.plan_item_rows(reports.read_plan(plan_paths(instrument_dir, "grouped")))

    assert list(ungrouped[0]) == list(grouped[0])


def test_every_item_names_the_velocity_band_it_was_stored_for(instrument_dir: Path) -> None:
    """A layered plan keeps a row per band for one key, so the band is what tells those rows apart."""
    rows = reports.plan_item_rows(reports.read_plan(plan_paths(instrument_dir, "grouped")))

    assert rows[0]["band"] == _LAYER


def test_the_instrument_rows_price_each_written_instrument(instrument_dir: Path) -> None:
    rows = reports.instrument_rows(reports.read_plan(plan_paths(instrument_dir, "grouped")))

    assert [row["id"] for row in rows] == [0]
    assert (rows[0]["name"], rows[0]["band"], rows[0]["keys"], rows[0]["samples"]) == (_INSTRUMENT, _LAYER, 7, 1)


def test_the_budget_row_states_what_the_plan_spent(instrument_dir: Path) -> None:
    row = reports.budget_rows(reports.read_plan(plan_paths(instrument_dir, "ungrouped")))[0]

    assert (row["strategy"], row["items"]) == ("ungrouped", 1)
    assert row["spent_pct"] == pytest.approx(50.0, abs=0.1)


def test_a_metric_with_no_finite_value_behind_it_is_named_rather_than_dropped(instrument_dir: Path) -> None:
    """``write_json`` writes a non-finite reading as null, so the table says so instead of showing a number."""
    metrics = reports.read_metrics(plan_paths(instrument_dir, "ungrouped"))
    row = reports.event_rows(metrics, ComparedNote(_LAYER, "p060_C4"))[0]

    assert row["loudness_delta_lu"] == "n/a"
    assert row["snr_db"] == 20.0
    assert row["mrstft"] == 0.5


def test_per_note_rows_carry_each_pitch_s_share_of_the_objective(instrument_dir: Path) -> None:
    rows = reports.note_metric_rows(reports.read_metrics(plan_paths(instrument_dir, "ungrouped")))

    assert [(row["note"], row["objective"], row["classes"]) for row in rows] == [("C4", 2.5, 1)]


def test_events_of_a_pitch_no_note_covers_come_back_empty(instrument_dir: Path) -> None:
    unplayed = ComparedNote(_LAYER, "p099_D#7")
    assert reports.event_rows(reports.read_metrics(plan_paths(instrument_dir, "ungrouped")), unplayed) == []


def test_the_ab_pair_resolves_to_the_two_files_that_were_written(instrument_dir: Path) -> None:
    paths = plan_paths(instrument_dir, "ungrouped")
    assert reports.compared_notes(paths) == [ComparedNote(_LAYER, "p060_C4")]
    reference, rendered = reports.comparison(paths, ComparedNote(_LAYER, "p060_C4"))
    assert reference.is_file() and rendered.is_file()


def test_a_strategy_that_wrote_no_pairs_offers_none(instrument_dir: Path) -> None:
    assert reports.compared_notes(plan_paths(instrument_dir, "grouped")) == []


def test_absent_optional_artifacts_read_as_absent(instrument_dir: Path) -> None:
    paths = plan_paths(instrument_dir, "ungrouped")
    assert reports.module_render(paths) is None
    assert reports.stored_samples(paths) == []
    assert reports.infeasible_reason(paths) is None
    assert reports.report_text(paths).startswith("Instrument")


def test_an_infeasible_strategy_reads_back_the_reason_it_left(instrument_dir: Path) -> None:
    paths = plan_paths(instrument_dir, "grouped")
    paths.infeasible.write_text("grouped allocation is infeasible at this budget:\ntoo small\n", encoding="utf-8")

    assert "too small" in (reports.infeasible_reason(paths) or "")
