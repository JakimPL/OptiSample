from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from optisample.carrier.instrument import CarrierInstrument, carrier_instrument
from optisample.carrier.source import CarrierSource
from optisample.carrier.store import CarrierSettings
from optisample.config import OptiConfig
from optisample.config.tracker import TrackerConfig, TrackerFormat
from optisample.dsp.level import gain_to_db
from optisample.dsp.loop import Loop
from optisample.dsp.resample import resample_to
from optisample.io.tracker.envelope import sounding_gain
from optisample.io.tracker.target import ExportTarget, export_target
from optisample.metrics.base import Signal
from tests.carrier.conftest import ROOT_PITCH, SR, TEMPO, Sourcer
from trackmod.spec.levels import MAX_VOLUME

Settings = Callable[..., CarrierSettings]

_INSTRUMENT_ID = "Piano"
_PLAYED_BACK_DB = 15.0  # how far the written pair's own error stays under the recording it reproduces
_LOOP = Loop(start=SR // 4, end=SR // 2)


def _played_back(built: CarrierInstrument, index: int, settings: CarrierSettings) -> tuple[Signal, Signal]:
    """What the module puts out for one sample at its own key, beside the recording it stands for."""
    carrier = built.stored[index]
    stored = carrier.stored
    gain = built.gains[index]
    envelope = built.unit.instrument.volume_envelope
    loudest = max(held.playback_gain for held in built.stored)
    played = sounding_gain(envelope, tempo=settings.grid.tempo, frames=stored.frames, sample_rate=stored.sample_rate)
    out = stored.pcm * played * (gain / MAX_VOLUME) * loudest
    reference = resample_to(carrier.source.recording, carrier.source.sample_rate, stored.sample_rate)[: stored.frames]
    return np.asarray(out, dtype=np.float64), np.asarray(reference, dtype=np.float64)


def _build(sources: Sequence[CarrierSource], settings: CarrierSettings) -> CarrierInstrument:
    return carrier_instrument(sources, instrument_id=_INSTRUMENT_ID, name=_INSTRUMENT_ID, settings=settings)


def test_the_written_pair_plays_the_recordings_back(source: Sourcer, carrier_settings: Settings) -> None:
    """The claim the whole route rests on: waveform times envelope times step is the recording again."""
    settings = carrier_settings()
    sources = [source(pitch=ROOT_PITCH + step, decay_db=30.0, loop=_LOOP) for step in (0, 2, 4)]
    built = _build(sources, settings)

    for index in range(len(sources)):
        out, reference = _played_back(built, index, settings)
        error = gain_to_db(np.sqrt(np.mean((out - reference) ** 2)) / np.sqrt(np.mean(reference**2)))
        assert error < -_PLAYED_BACK_DB


def test_a_shallower_depth_stores_half_the_bytes(source: Sourcer, carrier_settings: Settings) -> None:
    """Halving the depth is what doubles what a budget buys, which is the trade the carrier affords."""
    sources = [source(pitch=ROOT_PITCH + step, loop=_LOOP) for step in (0, 4)]
    deep = _build(sources, carrier_settings(depth=16))
    shallow = _build(sources, carrier_settings(depth=8))

    assert shallow.stored_bytes * 2 == deep.stored_bytes


def test_every_key_the_format_numbers_is_answered(source: Sourcer, carrier_settings: Settings) -> None:
    """A routing is widened from the keys the set was recorded at, so no key falls through."""
    settings = carrier_settings()
    built = _build([source(pitch=ROOT_PITCH), source(pitch=ROOT_PITCH + 12)], settings)
    keymap = built.unit.instrument.keymap
    target = settings.target

    answered = [keymap[target.key(pitch).value] for pitch in range(target.min_pitch, target.max_pitch + 1)]
    assert all(assignment is not None for assignment in answered)


def test_a_key_between_two_recordings_plays_the_nearer_one(source: Sourcer, carrier_settings: Settings) -> None:
    """Each stretch of keyboard goes to whichever recording is closer, which is what widening settles."""
    settings = carrier_settings()
    built = _build([source(pitch=ROOT_PITCH), source(pitch=ROOT_PITCH + 12)], settings)
    keymap = built.unit.instrument.keymap
    target = settings.target

    assert keymap[target.key(ROOT_PITCH + 1).value].sample == 0
    assert keymap[target.key(ROOT_PITCH + 11).value].sample == 1


def test_the_sample_asking_for_most_takes_the_top_step(source: Sourcer, carrier_settings: Settings) -> None:
    """The balance a flat instrument loses is restored on the step the format keeps beside each sample."""
    built = _build([source(peak=0.4), source(pitch=ROOT_PITCH + 3, peak=0.02)], carrier_settings())

    assert max(built.gains) == MAX_VOLUME
    assert built.gains[0] > built.gains[1]


def test_a_format_pinning_its_gain_carries_the_balance_in_the_waveform(
    source: Sourcer,
    carrier_settings: Settings,
    config: OptiConfig,
) -> None:
    """FastTracker 2 keeps no per-sample multiplier, so every sample there reports the same top step."""
    written = export_target(
        TrackerConfig.model_validate({**config.export.tracker.model_dump(), "format": TrackerFormat.XM})
    )
    built = _build(
        [source(peak=0.4), source(pitch=ROOT_PITCH + 3, peak=0.02)],
        carrier_settings(written=written),
    )

    assert set(built.gains) == {MAX_VOLUME}


def test_the_instrument_writes_as_a_standalone_file(source: Sourcer, carrier_settings: Settings) -> None:
    """What a set is for: a file a tracker loads, holding every sample and the curve above them."""
    settings = carrier_settings()
    built = _build([source(pitch=ROOT_PITCH + step, loop=_LOOP) for step in (0, 5)], settings)
    file = settings.target.instrument_file(built.unit)

    assert file.violations() == ()
    assert len(file.to_bytes()) == file.size().total


def _crest_db(signal: Signal) -> float:
    """How far a stored waveform's peak stands over the body it holds, which is depth spent on nothing."""
    return gain_to_db(float(np.max(np.abs(signal))) / float(np.sqrt(np.mean(signal**2))))


def test_what_the_shared_curve_had_no_room_for_stops_setting_the_stored_peak(
    source: Sourcer, carrier_settings: Settings
) -> None:
    """What the second pass is for: a take standing apart spends less of its depth on its own crest.

    The set is stored end to end here, since a waveform wrapped on an early region never reaches the
    stretch the burst stands in.
    """
    sources = [source(pitch=ROOT_PITCH, spike_db=18.0)] + [source(pitch=ROOT_PITCH + step) for step in (2, 4)]
    held = _build(sources, carrier_settings(depth=8))
    plain = _build(sources, carrier_settings(depth=8, ratio=1.0))

    assert _crest_db(held.stored[0].stored.pcm) < _crest_db(plain.stored[0].stored.pcm)


def test_a_set_the_curve_already_states_is_stored_as_the_one_pass_answer(
    source: Sourcer, carrier_settings: Settings
) -> None:
    """The hold opens over the body, so recordings a shared curve follows closely reach the encoder untouched."""
    sources = [source(pitch=ROOT_PITCH + step, loop=_LOOP) for step in (0, 2, 4)]
    held = _build(sources, carrier_settings(depth=8))
    plain = _build(sources, carrier_settings(depth=8, ratio=1.0))

    for one, other in zip(held.stored, plain.stored):
        assert one.stored.pcm == pytest.approx(other.stored.pcm)


def test_the_bytes_a_set_is_written_as_reproduce(source: Sourcer, carrier_settings: Settings) -> None:
    """One seeded generator advances once per source, so a set written twice lands identically."""
    sources = [source(pitch=ROOT_PITCH + step, loop=_LOOP) for step in (0, 3)]
    first = _build(sources, carrier_settings(depth=8))
    again = _build(sources, carrier_settings(depth=8))

    for left, right in zip(first.stored, again.stored):
        assert left.stored.pcm == pytest.approx(right.stored.pcm)
