from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from trackmod.core.envelopes.envelope import Envelope

from optisample.carrier.source import CarrierSource
from optisample.carrier.store import CarrierSettings
from optisample.config.loop import EnvelopeConfig
from optisample.dsp.dynamics import held_back
from optisample.dsp.envelope import LevelReading, Signal, decompose, level_reading
from optisample.io.tracker.envelope import carried_signal
from optisample.music import midi_to_freq


def source_reading(source: CarrierSource, config: EnvelopeConfig) -> LevelReading:
    """The weighting ``source`` has every level of it read under, formed from its own pitch and rate."""
    return level_reading(source.sample_rate, config, midi_to_freq(source.root_pitch))


def compressed_source(source: CarrierSource, gain: Signal, reading: LevelReading) -> CarrierSource:
    """``source`` restated at the level ``gain`` left it, its key, loops and playing time as they stand.

    The split is taken again over the held-back recording, so the level a shape is fitted from next is the
    one the material now holds and the pair still multiplies back to what it is written from.
    """
    return replace(source, decomposition=decompose(source.recording * gain, reading))


def _held_gain(
    source: CarrierSource,
    envelope: Envelope | None,
    reading: LevelReading,
    *,
    settings: CarrierSettings,
) -> Signal:
    """What one source is held back by, read off the waveform the written curve would leave it as."""
    flattened = carried_signal(
        source.recording,
        envelope,
        tempo=settings.grid.tempo,
        sample_rate=source.sample_rate,
    )
    return held_back(flattened, reading, settings.compression)


def held_sources(
    sources: Sequence[CarrierSource],
    envelope: Envelope | None,
    *,
    settings: CarrierSettings,
) -> tuple[CarrierSource, ...]:
    """Every source restated at the level ``envelope`` left in it, held back before any of it is stored.

    Splitting a recording into level and carrier is already a compression of infinite ratio, exact but
    unwritable: the curve carrying its inverse is shared across a whole band, turns through a few dozen
    corners on a tick and amplitude grid, and stops at the quietest step a format still sounds. What that
    written curve has no room to state stays in the waveform, where peak normalization reads it as the
    crest and the depth is spent on it. This pass reads exactly that remainder and holds it back at a ratio
    the format does carry (:func:`~optisample.dsp.dynamics.held_back`), which leaves the set at a level the
    shape fitted next follows more closely and the waveforms stored under it spending their grids on
    timbre.
    """
    readings = [source_reading(source, settings.config.envelope) for source in sources]
    return tuple(
        compressed_source(source, _held_gain(source, envelope, reading, settings=settings), reading)
        for source, reading in zip(sources, readings)
    )
