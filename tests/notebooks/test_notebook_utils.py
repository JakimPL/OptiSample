"""Tests for the notebook helper modules (the notebook itself is validated by running it)."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from notebooks.utils import audioio, degrade, loading, views, viz
from optisample.dsp.spectral import band_energy

SR = 44_100


def tone(freq: float = 440.0, duration: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    times = np.arange(int(duration * SR)) / SR
    return amplitude * np.sin(2.0 * np.pi * freq * times)


@pytest.fixture(scope="module")
def demo(tmp_path_factory, config):
    root = tmp_path_factory.mktemp("demo")
    manifest = loading.load(loading.ensure_demo_manifest(root, config.synth))
    return root, manifest


# --- loading ---------------------------------------------------------------


def test_ensure_demo_manifest_is_idempotent(tmp_path, synth_config):
    first = loading.ensure_demo_manifest(tmp_path, synth_config)
    second = loading.ensure_demo_manifest(tmp_path, synth_config)
    assert first == second
    assert first.exists()


def test_instrument_and_sample_selection(demo):
    _, manifest = demo
    ids = loading.instrument_ids(manifest)
    assert ids == ["strings", "piano"]

    strings = loading.get_instrument(manifest, "strings")
    labels = loading.sample_labels(strings)
    assert len(labels) == len(strings.samples)
    # Round-trip: a label resolves back to the very sample it names.
    assert loading.sample_label(loading.get_sample(strings, labels[0])) == labels[0]


def test_unknown_lookups_raise(demo):
    _, manifest = demo
    with pytest.raises(KeyError):
        loading.get_instrument(manifest, "nope")
    with pytest.raises(KeyError):
        loading.get_sample(loading.get_instrument(manifest, "piano"), "p1 v1 c0")


def test_load_signal_matches_manifest_sample(demo):
    _, manifest = demo
    sample = loading.get_instrument(manifest, "strings").samples[0]
    signal, sample_rate = loading.load_signal(sample)
    assert sample_rate == SR
    assert signal.ndim == 1 and signal.size > 0


# --- audioio ---------------------------------------------------------------


def test_to_wav_bytes_roundtrips_and_normalizes():
    signal = tone(amplitude=0.05)  # quiet input
    data = audioio.to_wav_bytes(signal, SR)
    assert data[:4] == b"RIFF"
    back, sample_rate = sf.read(io.BytesIO(data), dtype="float64")
    assert sample_rate == SR
    assert back.shape[0] == signal.size
    assert float(np.max(np.abs(back))) == pytest.approx(0.95, abs=0.02)


def test_to_wav_bytes_without_normalization_keeps_level():
    signal = tone(amplitude=0.2)
    back, _ = sf.read(io.BytesIO(audioio.to_wav_bytes(signal, SR, normalize=False)), dtype="float64")
    assert float(np.max(np.abs(back))) == pytest.approx(0.2, abs=0.01)


# --- degrade ---------------------------------------------------------------


def test_quantize_snaps_to_step_grid():
    signal = tone()
    quantized = degrade.quantize(signal, 8)
    step = 2.0 / (2**8)
    residual = quantized / step
    assert np.allclose(residual, np.round(residual))
    assert np.unique(quantized).size <= 2**8


def test_lowpass_removes_high_frequency_energy():
    signal = tone(freq=8000.0)
    filtered = degrade.lowpass(signal, SR, 3000.0)
    assert band_energy(filtered, SR, 3000.0, SR / 2.0) < 1e-6 * band_energy(signal, SR, 3000.0, SR / 2.0)


def test_gain_scales_amplitude():
    signal = tone()
    assert np.allclose(degrade.gain(signal, 0.25), 0.25 * signal)


@pytest.mark.parametrize("antialias", [True, False])
def test_resample_roundtrip_preserves_length(antialias):
    signal = tone()
    out = degrade.resample_roundtrip(signal, SR, 22_050, antialias=antialias)
    assert out.size == signal.size


def test_naive_resample_aliases_more_than_antialiased():
    # A tone above the downsampled Nyquist aliases badly without a filter.
    signal = tone(freq=16_000.0)
    clean = degrade.resample_roundtrip(signal, SR, 22_050, antialias=True)
    aliased = degrade.resample_roundtrip(signal, SR, 22_050, antialias=False)
    err_clean = float(np.mean((signal - clean) ** 2))
    err_aliased = float(np.mean((signal - aliased) ** 2))
    assert err_aliased > err_clean


def test_degrade_spec_labels_and_dispatch():
    signal = tone()
    specs = {
        "8-bit": degrade.DegradeSpec(kind="quantize", bits=8),
        "lowpass 3 kHz": degrade.DegradeSpec(kind="lowpass", cutoff_hz=3000.0),
        "gain x0.3": degrade.DegradeSpec(kind="gain", factor=0.3),
        "resample 22050 Hz": degrade.DegradeSpec(kind="resample", target_sr=22_050),
    }
    for expected_label, spec in specs.items():
        assert spec.label == expected_label
        assert degrade.apply(spec, signal, SR).size == signal.size


def test_smoke_specs_cover_the_p1_set():
    labels = [spec.label for spec in degrade.smoke_specs()]
    assert labels == ["16-bit", "8-bit", "lowpass 3 kHz", "gain x0.3"]


# --- views -----------------------------------------------------------------


def test_sample_row_has_expected_shape(demo, spectral_config):
    _, manifest = demo
    sample = loading.get_instrument(manifest, "strings").samples[0]
    signal, sample_rate = loading.load_signal(sample)
    row = views.sample_row(sample, signal, sample_rate, spectral_config)
    assert row["frames"] == signal.size
    assert row["dur_s"] == pytest.approx(signal.size / sample_rate)
    assert float(row["kib_16"]) > float(row["kib_8"]) > 0.0


def test_material_rows_weight_is_count_times_duration(demo):
    _, manifest = demo
    rows = views.material_rows(loading.get_instrument(manifest, "strings"))
    assert rows
    for row in rows:
        assert row["weight"] == pytest.approx(float(row["count"]) * float(row["dur_s"]))


def test_stored_bytes_matches_size_model():
    from optisample.metrics import SampleSize
    from optisample.metrics.size import INSTRUMENT_HEADER_BYTES

    assert views.stored_bytes([1000, 2000], 16) == (
        SampleSize(1000, 16).total_bytes + SampleSize(2000, 16).total_bytes + INSTRUMENT_HEADER_BYTES
    )


def test_budget_summary_flags_over_budget(demo):
    _, manifest = demo
    strings = loading.get_instrument(manifest, "strings")
    frame_counts = [loading.load_signal(sample)[0].size for sample in strings.samples]
    summary = views.budget_summary(strings, frame_counts)
    assert summary["n_samples"] == len(strings.samples)
    assert float(summary["over_ratio_16"]) > float(summary["over_ratio_8"]) > 0.0
    assert isinstance(summary["fits_16"], bool)


def test_compare_identical_is_transparent(composite):
    signal = tone()
    row = views.compare(signal, signal.copy(), SR, "identical", composite)
    assert float(row["fidelity"]) == pytest.approx(0.0, abs=1e-6)
    assert float(row["snr_db"]) > 100.0


def test_compare_ranks_16bit_above_8bit(composite):
    signal = tone()
    row16 = views.compare(signal, degrade.quantize(signal, 16), SR, "16-bit", composite)
    row8 = views.compare(signal, degrade.quantize(signal, 8), SR, "8-bit", composite)
    assert float(row16["fidelity"]) < float(row8["fidelity"])


def test_compare_isolates_pure_level_change_to_loudness(composite):
    signal = tone()
    row = views.compare(signal, degrade.gain(signal, 0.3), SR, "quieter", composite)
    assert float(row["fidelity"]) == pytest.approx(0.0, abs=1e-3)
    assert abs(float(row["loudness_dLU"])) > 5.0


def test_compare_specs_labels_align(composite):
    signal = tone()
    specs = degrade.smoke_specs()
    rows = views.compare_specs(signal, SR, specs, composite)
    assert [row["candidate"] for row in rows] == [spec.label for spec in specs]


# --- viz -------------------------------------------------------------------


def test_waveform_and_spectrogram_render_to_png(spectral_config, metrics_config):
    signal = tone()
    figures = (
        viz.waveform_figure(signal, SR),
        viz.spectrogram_figure(
            signal, SR, params=spectral_config.stft, dynamic_range_db=metrics_config.preprocess.dynamic_range_db
        ),
    )
    for figure in figures:
        png = viz.figure_png(figure)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_contribution_bar_totals_match_fidelity(composite, metrics_config):
    signal = tone()
    weights = metrics_config.weights
    rows = views.compare_specs(signal, SR, degrade.smoke_specs(), composite)
    for row in rows:
        stacked = sum(weights.get(key, 0.0) * float(row[key]) for key in viz._CONTRIBUTION_KEYS)
        assert stacked == pytest.approx(float(row["fidelity"]), rel=1e-9, abs=1e-9)
    assert viz.figure_png(viz.contribution_bar(rows, weights))[:8] == b"\x89PNG\r\n\x1a\n"
