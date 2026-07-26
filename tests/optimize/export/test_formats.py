"""What holds across every tracker format the exporter writes, and what each one decides for itself."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from optisample.config.optimize import SweepConfig
from optisample.config.render import RenderConfig
from optisample.config.tracker import TrackerFormat
from optisample.io.render import openmpt123_available, render_module
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import InstrumentPlan
from tests.optimize.export.demo import SR
from trackmod.module.protocol import TrackerModule

requires_openmpt = pytest.mark.skipif(not openmpt123_available(), reason="openmpt123 not installed")

_FORMATS = tuple(TrackerFormat)

_ABOVE_XM_PITCH = 108  # the first key past FastTracker 2's eight octaves, which Impulse Tracker still numbers

_RMS_TOLERANCE = 0.002
_SILENCE = 1e-6


def rms(audio: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(audio**2)))


def peak(audio: NDArray[np.float64]) -> float:
    """The loudest frame, reading a render that stops here as silent."""
    return float(np.max(np.abs(audio), initial=0.0))


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_every_format_writes_the_plan_it_is_given(
    tracker_format: TrackerFormat, build: Callable[..., tuple[InstrumentPlan, TrackerModule]]
) -> None:
    plan, module = build(None, tracker_format)
    assert module.violations() == ()
    assert len(module.song.samples) == len(plan.pitches)
    assert module.size().total == len(module.to_bytes())  # the size model accounts for every byte written


@pytest.mark.parametrize("tracker_format", _FORMATS)
def test_every_format_names_its_file_after_itself(
    tracker_format: TrackerFormat,
    tmp_path: Path,
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]],
) -> None:
    _, module = build(None, tracker_format)
    path = tmp_path / f"module{module.extension}"
    module.save(path)
    assert path.suffix == f".{tracker_format}"
    assert path.stat().st_size == module.size().total


@requires_openmpt
def test_both_formats_play_the_same_music(
    build: Callable[..., tuple[InstrumentPlan, TrackerModule]], render_config: RenderConfig
) -> None:
    """One plan written two ways sounds the same, within what each format's pitch lattice allows.

    Impulse Tracker stores a sample's rate outright while FastTracker 2 reaches it by transposing the
    key, so the two land on pitches a fraction of a cent apart and the waveforms drift out of phase over
    a long note. The level each key sounds at is what the two formats agree on. The renders run to
    different lengths because the canonical Impulse Tracker pattern floor pads its song with rows the
    material never reaches, which play as trailing silence.
    """
    _, it_module = build(None, TrackerFormat.IT)
    _, xm_module = build(None, TrackerFormat.XM)
    it_audio, it_rate = render_module(it_module, render_config)
    xm_audio, xm_rate = render_module(xm_module, render_config)
    played = min(it_audio.size, xm_audio.size)

    assert (it_rate, xm_rate) == (render_config.sample_rate, render_config.sample_rate)
    assert abs(rms(it_audio[:played]) - rms(xm_audio[:played])) < _RMS_TOLERANCE
    assert peak(it_audio[played:]) < _SILENCE  # the longer render only carries the padding rows
    assert peak(xm_audio[played:]) < _SILENCE


def test_a_pitch_above_a_formats_keyboard_is_refused_by_that_format(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    as_format: Callable[[TrackerFormat | None], ExportContext],
    piano_note: Callable[..., NDArray[np.float64]],
) -> None:
    """The top of the shared keyboard is Impulse Tracker's alone; FastTracker 2 stops two octaves lower."""
    audio = {(_ABOVE_XM_PITCH, 100): piano_note(_ABOVE_XM_PITCH, 100, seed=1)}
    material = [NoteEvent(pitch=_ABOVE_XM_PITCH, velocity=100, duration_s=0.4)]
    instrument = InstrumentSpec(
        id="top",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("top.wav"), pitch=_ABOVE_XM_PITCH, velocity=100)],
        material=material,
    )
    settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depths=(16, 8)))
    plan = optimize_instrument(instrument, audio, SR, settings)

    module = build_module(plan, audio, SR, material, as_format(TrackerFormat.IT))
    assert module.violations() == ()
    with pytest.raises(ValueError, match=f"MIDI note {_ABOVE_XM_PITCH} is outside the XM key range"):
        build_module(plan, audio, SR, material, as_format(TrackerFormat.XM))
