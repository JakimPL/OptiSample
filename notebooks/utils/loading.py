"""Discover / generate a manifest and pull individual samples out of it.

Selection helpers are keyed by short human labels (``"p60 v80 c0"``) so a marimo dropdown, whose
value is a string, can round-trip straight back to the :class:`SourceSample` it names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from optisample.io.audio import read_wav
from optisample.io.manifest import load_manifest
from optisample.model import InstrumentSpec, Manifest, SourceSample
from optisample.synth import default_synth_config, generate_demo

Signal = NDArray[np.float64]


def ensure_demo_manifest(root: Path | str, *, seed: int = 0) -> Path:
    """Return the manifest under ``root``, generating the synthetic demo there if absent."""
    root = Path(root)
    manifest_path = root / "manifest.yaml"
    if manifest_path.exists():
        return manifest_path
    # transitional: phase 9 threads a SynthConfig from the notebook's config selection.
    return generate_demo(root, default_synth_config(), seed=seed)


def load(path: Path | str) -> Manifest:
    """Load and validate a manifest (relative sample paths resolved against its directory)."""
    return load_manifest(path)


def instrument_ids(manifest: Manifest) -> list[str]:
    """The ids of every instrument, in manifest order."""
    return [instrument.id for instrument in manifest.instruments]


def get_instrument(manifest: Manifest, instrument_id: str) -> InstrumentSpec:
    """Look up an instrument by id."""
    for instrument in manifest.instruments:
        if instrument.id == instrument_id:
            return instrument
    raise KeyError(f"no instrument {instrument_id!r}; have {instrument_ids(manifest)}")


def sample_label(sample: SourceSample) -> str:
    """Short, unique-per-instrument label: pitch / velocity / controller."""
    return f"p{sample.pitch} v{sample.velocity} c{sample.controller:g}"


def sample_labels(instrument: InstrumentSpec) -> list[str]:
    """Labels for every sample of an instrument, in manifest order."""
    return [sample_label(sample) for sample in instrument.samples]


def get_sample(instrument: InstrumentSpec, label: str) -> SourceSample:
    """Look up a sample by its :func:`sample_label`."""
    for sample in instrument.samples:
        if sample_label(sample) == label:
            return sample
    raise KeyError(f"no sample {label!r}; have {sample_labels(instrument)}")


def load_signal(sample: SourceSample) -> tuple[Signal, int]:
    """Read a sample's WAV as ``(mono float64 signal, sample_rate)``."""
    return read_wav(sample.file)
