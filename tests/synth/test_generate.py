from __future__ import annotations

from pathlib import Path

import numpy as np

from optisample.cli import main
from optisample.config.synth import SynthConfig
from optisample.io.audio import read_wav
from optisample.io.manifest import load_manifest
from optisample.synth import generate_demo

QUICK_RATE = 8_000


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


def test_cli_synth(tmp_path: Path) -> None:
    main(["synth", str(tmp_path), "--sample-rate", str(QUICK_RATE), "--seed", "1"])
    assert (tmp_path / "manifest.yaml").exists()
