from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from trackmod import BitDepth

from optisample.carrier.instrument import written_shape
from optisample.carrier.store import CarrierSettings, store_carrier
from optisample.config import OptiConfig
from optisample.dsp.loop import Loop
from optisample.io.tracker.envelope import NO_ENVELOPE, carried_signal
from optisample.io.tracker.target import ExportTarget
from tests.carrier.conftest import ROOT_PITCH, SR, TEMPO, Sourcer, level_range_db, levels_db

Settings = Callable[..., CarrierSettings]

_LOOP = Loop(start=SR // 4, end=SR // 2)


def _written(source_set: list, settings: CarrierSettings) -> object:
    return written_shape(source_set, target=settings.target, grid=settings.grid).envelope


def test_a_set_carrying_no_curve_keeps_its_recording(source: Sourcer) -> None:
    """There is no level to hand over, so the waveform stays the recording it was taken from."""
    one = source()

    assert carried_signal(one.recording, NO_ENVELOPE, tempo=TEMPO, sample_rate=SR) == pytest.approx(one.recording)


def test_dividing_by_the_written_curve_flattens_the_level(
    source: Sourcer,
    carrier_settings: Settings,
    config: OptiConfig,
) -> None:
    """The whole of the carrier idea: the decline moves to the envelope and the waveform holds timbre."""
    settings = carrier_settings()
    one = source(decay_db=36.0)
    envelope = _written([one], settings)
    flattened = carried_signal(one.recording, envelope, tempo=TEMPO, sample_rate=SR)

    recorded = level_range_db(levels_db(one.recording, SR, ROOT_PITCH, config))
    stored = level_range_db(levels_db(flattened, SR, ROOT_PITCH, config))
    assert stored < recorded


def test_the_flattened_waveform_stays_finite_under_a_curve_that_silences(
    source: Sourcer,
    carrier_settings: Settings,
) -> None:
    """A moment the curve silences plays as silence, so the division is taken against the softest step."""
    settings = carrier_settings()
    one = source(decay_db=90.0, seconds=3.0)
    flattened = carried_signal(one.recording, _written([one], settings), tempo=TEMPO, sample_rate=SR)

    assert np.all(np.isfinite(flattened))


def test_a_source_naming_a_loop_is_stored_around_it(source: Sourcer, carrier_settings: Settings) -> None:
    """The loop a source carries is the region its waveform wraps, and the storage stops at its end."""
    settings = carrier_settings(rate=SR)
    one = source(loop=_LOOP)
    stored = store_carrier(one, NO_ENVELOPE, settings=settings, rng=np.random.default_rng(0)).stored

    assert stored.loop is not None
    assert stored.frames == stored.loop.end


def test_a_source_naming_no_loop_is_stored_as_the_span_it_plays(
    source: Sourcer,
    carrier_settings: Settings,
) -> None:
    """A recording the loop stage settled nothing over keeps what it plays rather than wrapping."""
    stored = store_carrier(source(), NO_ENVELOPE, settings=carrier_settings(), rng=np.random.default_rng(0)).stored

    assert stored.loop is None


@pytest.mark.parametrize("depth", [BitDepth.EIGHT, BitDepth.SIXTEEN])
def test_the_waveform_is_kept_at_the_depth_asked_for(
    source: Sourcer, carrier_settings: Settings, depth: BitDepth
) -> None:
    """Depth is the axis a flattened waveform buys its bytes back on, so it is stored as asked."""
    stored = store_carrier(
        source(),
        NO_ENVELOPE,
        settings=carrier_settings(depth=depth),
        rng=np.random.default_rng(0),
    ).stored

    assert stored.depth == depth


def test_the_waveform_is_kept_at_the_rate_asked_for(source: Sourcer, carrier_settings: Settings) -> None:
    """A stored rate is the caller's, so a set lands on one clock whatever it was analyzed at."""
    stored = store_carrier(
        source(),
        NO_ENVELOPE,
        settings=carrier_settings(rate=11_025),
        rng=np.random.default_rng(0),
    ).stored

    assert stored.sample_rate == 11_025


def test_the_stored_waveform_reports_what_playback_owes_it(source: Sourcer, carrier_settings: Settings) -> None:
    """Storing hot spends the whole depth on one recording, and playback gives that scaling back."""
    carrier = store_carrier(source(peak=0.05), NO_ENVELOPE, settings=carrier_settings(), rng=np.random.default_rng(0))

    assert carrier.playback_gain == pytest.approx(carrier.stored.playback_gain)
    assert carrier.playback_gain < 1.0
