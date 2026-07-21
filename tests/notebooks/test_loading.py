"""Tests for the notebook loading helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from notebooks.utils import loading
from optisample.config import OptiConfig
from optisample.config.synth import SynthConfig
from optisample.model import Manifest

SR = 44_100

Demo = tuple[Path, Manifest]


def test_ensure_demo_manifest_is_idempotent(tmp_path: Path, synth_config: SynthConfig) -> None:
    first = loading.ensure_demo_manifest(tmp_path, synth_config)
    second = loading.ensure_demo_manifest(tmp_path, synth_config)
    assert first == second
    assert first.exists()


def test_instrument_and_sample_selection(demo: Demo, config: OptiConfig) -> None:
    _, manifest = demo
    ids = loading.instrument_ids(manifest)
    assert ids == [preset.id for preset in config.synth.presets]

    strings = loading.get_instrument(manifest, "strings")
    labels = loading.sample_labels(strings)
    assert len(labels) == len(strings.samples)
    # Round-trip: a label resolves back to the very sample it names.
    assert loading.sample_label(loading.get_sample(strings, labels[0])) == labels[0]


def test_unknown_lookups_raise(demo: Demo) -> None:
    _, manifest = demo
    with pytest.raises(KeyError):
        loading.get_instrument(manifest, "nope")
    with pytest.raises(KeyError):
        loading.get_sample(loading.get_instrument(manifest, "piano"), "p1 v1 c0")


def test_load_signal_matches_manifest_sample(demo: Demo) -> None:
    _, manifest = demo
    sample = loading.get_instrument(manifest, "strings").samples[0]
    signal, sample_rate = loading.load_signal(sample)
    assert sample_rate == SR
    assert signal.ndim == 1 and signal.size > 0
