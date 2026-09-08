from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from trackmod import TrackerModule

from optisample.config.optimize import SweepConfig
from optisample.config.render import RenderConfig
from optisample.config.tracker import TrackerFormat
from optisample.io.render import openmpt123_available, render_module
from optisample.io.tracker.target import ExportTarget
from optisample.io.tracker.voices import routed_voices
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.export import build_module
from optisample.optimize.export.context import ExportContext
from optisample.optimize.orchestrate import optimize_instrument
from optisample.optimize.orchestrate.settings import OptimizeSettings
from optisample.optimize.plans import InstrumentPlan
from optisample.optimize.tasks import StoredRecordings
from tests.optimize.export.demo import SR
from tests.optimize.export.test_material import song_cells

Recordings = Callable[..., StoredRecordings]

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
    assert len(routed_voices(module.song).samples) == len(plan.pitches)
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
    recordings: Recordings,
) -> None:
    """The top of the shared keyboard is Impulse Tracker's alone; FastTracker 2 stops two octaves lower."""
    audio = {SampleKey(_ABOVE_XM_PITCH, 100): piano_note(_ABOVE_XM_PITCH, 100, seed=1)}
    material = [NoteEvent(pitch=_ABOVE_XM_PITCH, velocity=100, duration_s=0.4)]
    instrument = InstrumentSpec(
        id="top",
        budget_kb=64.0,
        samples=[SourceSample(file=Path("top.wav"), pitch=_ABOVE_XM_PITCH, velocity=100)],
        material=material,
    )
    settings = optimize_settings(sweep=sweep(rates=(44_100, 11_025), depth=16))
    plan = optimize_instrument(instrument, recordings(audio, SR), settings)

    module = build_module(plan, recordings(audio, SR), material, as_format(TrackerFormat.IT))
    assert module.violations() == ()
    with pytest.raises(ValueError, match=f"MIDI note {_ABOVE_XM_PITCH} is outside the XM key range"):
        build_module(plan, recordings(audio, SR), material, as_format(TrackerFormat.XM))


_WIDE_KEYS = tuple(range(60, 80))  # twenty keys, more samples than one FastTracker 2 instrument owns
_WIDE_BUDGET_KB = 512.0  # room for a recording per key, so the plan stores every one of them
_SHORT_NOTE_S = 0.1


def _wide_instrument() -> InstrumentSpec:
    """An instrument recorded and played across twenty keys, at one dynamic."""
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=_SHORT_NOTE_S) for pitch in _WIDE_KEYS]
    return InstrumentSpec(
        id="wide",
        budget_kb=_WIDE_BUDGET_KB,
        samples=[SourceSample(file=Path(f"{pitch}.wav"), pitch=pitch, velocity=100) for pitch in _WIDE_KEYS],
        material=material,
    )


def test_a_plan_holding_more_samples_than_an_xm_instrument_owns_is_written_as_several(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    as_format: Callable[[TrackerFormat | None], ExportContext],
    retarget: Callable[[TrackerFormat], ExportTarget],
    piano_note: Callable[..., NDArray[np.float64]],
    recordings: Recordings,
) -> None:
    """FastTracker 2 numbers sixteen samples inside an instrument, so a wider plan is cut into slots."""
    instrument = _wide_instrument()
    audio = {SampleKey(pitch, 100): piano_note(pitch, 100, dur=_SHORT_NOTE_S, seed=pitch) for pitch in _WIDE_KEYS}
    settings = optimize_settings(sweep=sweep(rates=(11_025,), depth=8, dither=False))
    plan = optimize_instrument(instrument, recordings(audio, SR), settings)
    per_instrument = retarget(TrackerFormat.XM).max_samples_per_instrument

    module = build_module(plan, recordings(audio, SR), instrument.material, as_format(TrackerFormat.XM))

    assert len(plan.sample_units()) == len(_WIDE_KEYS)  # every key kept its own recording
    assert module.violations() == ()
    assert len(routed_voices(module.song).instruments) == 2
    assert all(len(written.samples) <= per_instrument for written in routed_voices(module.song).instruments)


def test_one_instrument_holds_the_whole_plan_where_the_format_numbers_samples_freely(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    as_format: Callable[[TrackerFormat | None], ExportContext],
    piano_note: Callable[..., NDArray[np.float64]],
    recordings: Recordings,
) -> None:
    """Impulse Tracker lets an instrument reach the whole sample table, so the same plan stays one."""
    instrument = _wide_instrument()
    audio = {SampleKey(pitch, 100): piano_note(pitch, 100, dur=_SHORT_NOTE_S, seed=pitch) for pitch in _WIDE_KEYS}
    settings = optimize_settings(sweep=sweep(rates=(11_025,), depth=8, dither=False))
    plan = optimize_instrument(instrument, recordings(audio, SR), settings)

    module = build_module(plan, recordings(audio, SR), instrument.material, as_format(TrackerFormat.IT))

    assert module.violations() == ()
    assert len(routed_voices(module.song).instruments) == 1
    assert len(routed_voices(module.song).instruments[0].samples) == len(_WIDE_KEYS)


def test_every_note_of_a_cut_plan_names_the_instrument_its_key_resolves_to(
    optimize_settings: Callable[..., OptimizeSettings],
    sweep: Callable[..., SweepConfig],
    as_format: Callable[[TrackerFormat | None], ExportContext],
    retarget: Callable[[TrackerFormat], ExportTarget],
    piano_note: Callable[..., NDArray[np.float64]],
    recordings: Recordings,
) -> None:
    """Each slot owns a run of keys, and the pattern plays a note through the slot owning its own."""
    instrument = _wide_instrument()
    audio = {SampleKey(pitch, 100): piano_note(pitch, 100, dur=_SHORT_NOTE_S, seed=pitch) for pitch in _WIDE_KEYS}
    settings = optimize_settings(sweep=sweep(rates=(11_025,), depth=8, dither=False))
    plan = optimize_instrument(instrument, recordings(audio, SR), settings)
    per_instrument = retarget(TrackerFormat.XM).max_samples_per_instrument

    module = build_module(plan, recordings(audio, SR), instrument.material, as_format(TrackerFormat.XM))

    played = [cell for cell in song_cells(module) if cell.instrument is not None]
    assert [cell.instrument for cell in played] == [
        0 if index < per_instrument else 1 for index in range(len(_WIDE_KEYS))
    ]
    for cell in played:
        assignment = routed_voices(module.song).instruments[cell.instrument].assignment(cell.note)
        assert (
            assignment is not None
            and assignment.sample in routed_voices(module.song).instruments[cell.instrument].samples
        )
