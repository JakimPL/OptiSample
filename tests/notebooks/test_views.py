from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod import BitDepth
from trackmod.module.storage import Storage

from notebooks.utils import degrade, loading, views
from optisample.config.spectral import SpectralConfig
from optisample.metrics import CompositeFidelity
from optisample.model import Manifest
from optisample.optimize.plans.budget import populated_instrument_bytes

SR = 44_100

Demo = tuple[Path, Manifest]


def test_sample_row_has_expected_shape(demo: Demo, spectral_config: SpectralConfig, storage: Storage) -> None:
    _, manifest = demo
    sample = loading.get_instrument(manifest, "strings").samples[0]
    signal, sample_rate = loading.load_signal(sample)
    row = views.sample_row(sample, signal, sample_rate, spectral_config, storage)
    assert row["frames"] == signal.size
    assert row["dur_s"] == pytest.approx(signal.size / sample_rate)
    assert float(row["kib_16"]) > float(row["kib_8"]) > 0.0


def test_material_rows_weight_is_count_times_duration(demo: Demo) -> None:
    _, manifest = demo
    rows = views.material_rows(loading.get_instrument(manifest, "strings"))
    assert rows
    for row in rows:
        assert row["weight"] == pytest.approx(float(row["count"]) * float(row["dur_s"]))


def test_stored_bytes_matches_the_formats_cost_table(storage: Storage) -> None:
    depth = BitDepth.SIXTEEN
    assert views.stored_bytes([1000, 2000], depth, storage) == (
        storage.sample_bytes(frames=1000, depth=depth)
        + storage.sample_bytes(frames=2000, depth=depth)
        + populated_instrument_bytes(storage)
    )


def test_budget_summary_flags_over_budget(demo: Demo, storage: Storage) -> None:
    _, manifest = demo
    strings = loading.get_instrument(manifest, "strings")
    frame_counts = [loading.load_signal(sample)[0].size for sample in strings.samples]
    summary = views.budget_summary(strings, frame_counts, storage)
    assert summary["n_samples"] == len(strings.samples)
    assert float(summary["over_ratio_16"]) > float(summary["over_ratio_8"]) > 0.0
    assert isinstance(summary["fits_16"], bool)


def test_compare_identical_is_transparent(
    composite: CompositeFidelity, tone: Callable[..., NDArray[np.float64]]
) -> None:
    signal = tone()
    row = views.compare(signal, signal.copy(), SR, "identical", composite)
    assert float(row["fidelity"]) == pytest.approx(0.0, abs=1e-6)
    assert float(row["snr_db"]) > 100.0


def test_compare_ranks_16bit_above_8bit(composite: CompositeFidelity, tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone()
    row16 = views.compare(signal, degrade.quantize(signal, 16), SR, "16-bit", composite)
    row8 = views.compare(signal, degrade.quantize(signal, 8), SR, "8-bit", composite)
    assert float(row16["fidelity"]) < float(row8["fidelity"])


def test_compare_isolates_pure_level_change_to_loudness(
    composite: CompositeFidelity, tone: Callable[..., NDArray[np.float64]]
) -> None:
    signal = tone()
    row = views.compare(signal, degrade.gain(signal, 0.3), SR, "quieter", composite)
    assert float(row["fidelity"]) == pytest.approx(0.0, abs=1e-3)
    assert abs(float(row["loudness_dLU"])) > 5.0


def test_compare_specs_labels_align(composite: CompositeFidelity, tone: Callable[..., NDArray[np.float64]]) -> None:
    signal = tone()
    specs = degrade.smoke_specs()
    rows = views.compare_specs(signal, SR, specs, composite)
    assert [row["candidate"] for row in rows] == [spec.label for spec in specs]
