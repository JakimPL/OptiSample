from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from optisample.config.synth import SynthConfig
from optisample.io.audio import read_wav
from optisample.io.note_extractor import IngestSettings, load_notes
from optisample.model import InstrumentSpec, Manifest, ProjectSpec, SourceSample
from optisample.synth import generate_demo

Signal = NDArray[np.float64]

_NOTES_SUFFIX = ".notes.json"
_DEMO_BUDGET_KB = 128.0


def ensure_demo(root: Path | str, config: SynthConfig, *, seed: int = 0) -> Path:
    """Return the demo directory under ``root``, generating the synthetic dataset (from ``config``) if absent."""
    root = Path(root)
    if not sorted(root.glob(f"*{_NOTES_SUFFIX}")):
        generate_demo(root, config, seed=seed)
    return root


def load(demo_dir: Path | str) -> Manifest:
    """Combine every ``.notes.json`` under a demo directory into one manifest (its samples dir is the sibling)."""
    demo_dir = Path(demo_dir)
    project = ProjectSpec(name=demo_dir.name)
    instruments: list[InstrumentSpec] = []
    for notes_json in sorted(demo_dir.glob(f"*{_NOTES_SUFFIX}")):
        instrument_id = notes_json.name[: -len(_NOTES_SUFFIX)]
        settings = IngestSettings(instrument_id=instrument_id, budget_kb=_DEMO_BUDGET_KB, project=project)
        manifest = load_notes(notes_json, demo_dir / instrument_id, settings)
        instruments.append(manifest.instruments[0])
    return Manifest(project=project, instruments=instruments)


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
    """Short, unique-per-instrument label: the sample's render-indexed WAV stem (e.g. ``0007_p60_v100``)."""
    return sample.file.stem


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
