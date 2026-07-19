from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optisample.cli import main
from optisample.config.synth import SynthConfig
from optisample.io.audio import read_wav
from optisample.io.manifest import load_manifest
from optisample.synth import Archetype, NoteSpec, generate_demo, midi_to_freq, render_sample

QUICK_RATE = 8_000


def test_midi_to_freq_a4() -> None:
    assert midi_to_freq(69) == pytest.approx(440.0)
    assert midi_to_freq(81) == pytest.approx(880.0)


def test_generate_demo_writes_manifest_and_audio(synth_config: SynthConfig, tmp_path: Path) -> None:
    manifest_path = generate_demo(tmp_path, synth_config, sample_rate=QUICK_RATE)
    assert manifest_path.exists()

    manifest = load_manifest(manifest_path)
    assert {inst.id for inst in manifest.instruments} == {preset.id for preset in synth_config.presets}

    for instrument in manifest.instruments:
        for sample in instrument.samples:
            assert sample.file.exists()
            data, rate = read_wav(sample.file)
            assert rate == QUICK_RATE
            assert data.size > 0
            assert np.all(np.isfinite(data))
            assert np.max(np.abs(data)) > 0.0  # not silent


def test_generate_demo_is_deterministic(synth_config: SynthConfig, tmp_path: Path) -> None:
    a = load_manifest(generate_demo(tmp_path / "a", synth_config, sample_rate=QUICK_RATE, seed=7))
    b = load_manifest(generate_demo(tmp_path / "b", synth_config, sample_rate=QUICK_RATE, seed=7))
    wave_a, _ = read_wav(a.instruments[0].samples[0].file)
    wave_b, _ = read_wav(b.instruments[0].samples[0].file)
    np.testing.assert_array_equal(wave_a, wave_b)


def _note(velocity: int) -> NoteSpec:
    return NoteSpec(pitch=60, velocity=velocity, controller=0.0, duration_s=0.5, sample_rate=QUICK_RATE)


@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_higher_velocity_is_louder(archetype: Archetype, synth_config: SynthConfig) -> None:
    rng = np.random.default_rng(0)
    soft = render_sample(archetype, _note(40), rng, synth_config)
    loud = render_sample(archetype, _note(120), rng, synth_config)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2)))

    assert rms(loud) > rms(soft)


@pytest.mark.parametrize("archetype", ["sustained", "piano"])
def test_higher_velocity_is_brighter(archetype: Archetype, synth_config: SynthConfig) -> None:
    rng = np.random.default_rng(0)
    soft = render_sample(archetype, _note(40), rng, synth_config)
    loud = render_sample(archetype, _note(120), rng, synth_config)

    def centroid(x: np.ndarray) -> float:
        mag = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.size, 1.0 / QUICK_RATE)
        total = float(np.sum(mag))
        return float(np.sum(freqs * mag) / total) if total > 0 else 0.0

    assert centroid(loud) > centroid(soft)


def test_cli_synth(tmp_path: Path) -> None:
    main(["synth", str(tmp_path), "--sample-rate", str(QUICK_RATE), "--seed", "1"])
    assert (tmp_path / "manifest.yaml").exists()
