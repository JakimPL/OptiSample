from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from optisample.carrier.source import CarrierSource
from optisample.config.codec import EncodeConfig
from optisample.config.dynamics import HoldConfig
from optisample.dsp.surrogate import EncodeContext, EncodingParams, StoredSample, encode
from optisample.io.tracker.envelope import EnvelopeGrid, carried_signal
from optisample.io.tracker.target import ExportTarget
from trackmod.core.envelopes.envelope import Envelope


@dataclass(frozen=True)
class StoredCarrier:
    """One source stored as the waveform a tracker keeps, beside the recording it was taken from.

    ``stored`` is the encoder's own answer -- the resampled, looped and quantized span, and how hot it was
    stored -- and ``source`` travels with it so a caller reports what each waveform holds and how far the
    envelope beside it reached.
    """

    source: CarrierSource
    stored: StoredSample

    @property
    def playback_gain(self) -> float:
        """What playback multiplies this waveform by to sound at the level it was stored from."""
        return self.stored.playback_gain


@dataclass(frozen=True)
class CarrierSettings:
    """What writing a set of recordings as carrier samples is carried out with.

    ``grid`` states the clock and the two grids a volume envelope is written on, ``params`` the rate and
    depth every waveform is stored at, and ``config`` how a span is resampled, wrapped and quantized on the
    way there. ``compression`` is the curve the level a written envelope has no room to state is held back
    along (:func:`~optisample.carrier.compress.held_sources`). ``seed`` starts the one generator the set's
    dither is drawn from, so the bytes a set is written as reproduce.
    """

    target: ExportTarget
    grid: EnvelopeGrid
    params: EncodingParams
    config: EncodeConfig
    compression: HoldConfig
    seed: int


def store_carrier(
    source: CarrierSource,
    envelope: Envelope | None,
    *,
    settings: CarrierSettings,
    rng: np.random.Generator,
) -> StoredCarrier:
    """``source`` stored the way a tracker keeps it: the carrier, at the rate, depth and loop asked for.

    The waveform reaches the encoder already divided by what the envelope plays it down by
    (:func:`~optisample.io.tracker.envelope.carried_signal`), so the span it resamples, wraps and quantizes
    is the flattened one -- which is what spends the whole of a shallow grid on timbre rather than on a
    decline the envelope states anyway. Which loop is stored is the source's own choice, so the params a
    caller states hold for every source of a set while each keeps the region settled over it.
    """
    return StoredCarrier(
        source=source,
        stored=encode(
            carried_signal(
                source.recording,
                envelope,
                tempo=settings.grid.tempo,
                sample_rate=source.sample_rate,
            ),
            source.sample_rate,
            replace(settings.params, loop_index=source.loop_index),
            EncodeContext(
                root_pitch=source.root_pitch,
                config=settings.config,
                settled=source.loops,
                rng=rng,
            ),
        ),
    )
