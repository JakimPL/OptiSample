from __future__ import annotations

from pathlib import Path

import numpy as np

from optisample.cli import main
from optisample.config.synth import SynthConfig
from optisample.io.audio import read_wav
from optisample.io.note_extractor import IngestSettings, load_notes
from optisample.model import ProjectSpec
from optisample.synth import DemoSettings, generate_demo

QUICK_RATE = 8_000


def _first_wav(samples_dir: Path) -> Path:
    return sorted(samples_dir.glob("*.wav"))[0]


def test_generate_demo_writes_notes_and_audio(synth_config: SynthConfig, tmp_path: Path) -> None:
    outputs = generate_demo(tmp_path, synth_config, DemoSettings(sample_rate=QUICK_RATE))
    assert {samples_dir.name for _, samples_dir in outputs} == {preset.id for preset in synth_config.presets}

    for notes_json, samples_dir in outputs:
        assert notes_json.exists()
        settings = IngestSettings(instrument_id=samples_dir.name, budget_kb=64.0, project=ProjectSpec(name="demo"))
        manifest = load_notes(notes_json, samples_dir, settings)
        instrument = manifest.instruments[0]
        assert len(instrument.samples) == len(instrument.material)  # one note is both a sample and an event
        for sample in instrument.samples:
            assert sample.file.exists()
            data, rate = read_wav(sample.file)
            assert rate == QUICK_RATE
            assert data.size > 0
            assert np.all(np.isfinite(data))
            assert np.max(np.abs(data)) > 0.0  # not silent


def test_generate_demo_is_deterministic(synth_config: SynthConfig, tmp_path: Path) -> None:
    a = generate_demo(tmp_path / "a", synth_config, DemoSettings(sample_rate=QUICK_RATE, seed=7))
    b = generate_demo(tmp_path / "b", synth_config, DemoSettings(sample_rate=QUICK_RATE, seed=7))
    wave_a, _ = read_wav(_first_wav(a[0][1]))
    wave_b, _ = read_wav(_first_wav(b[0][1]))
    np.testing.assert_array_equal(wave_a, wave_b)


def test_cli_synth(tmp_path: Path) -> None:
    main(["synth", str(tmp_path), "--sample-rate", str(QUICK_RATE), "--seed", "1"])
    assert sorted(tmp_path.glob("*.notes.json"))  # one .notes.json per preset
