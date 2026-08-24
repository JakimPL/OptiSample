from collections.abc import Sequence
from typing import Final

from optisample.config.dynamic_axis import DynamicAxis, DynamicAxisConfig
from optisample.keys import Identified
from optisample.model import InstrumentSpec, Manifest, NoteEvent, SourceSample

_ONE_POSITION: Final = 1  # positions an axis carrying no dynamics leaves every recording at


def dynamic_of(identified: Identified, config: DynamicAxisConfig) -> int:
    """How loudly ``identified`` was played, read off the axis its instrument's dynamics travel on.

    A controller average is time-weighted and so continuous, while a dynamic is one of the 128 positions
    a tracker writes, which is what rounding settles.

    Raises:
        ValueError: when the axis names a controller the note carries no average for, which is a run
            configured for a library the material was not recorded with.
    """
    if config.axis is DynamicAxis.VELOCITY:
        return identified.velocity

    average = identified.cc_averages.get(config.controller)
    if average is None:
        raise ValueError(
            f"dynamics read from controller {config.controller}, which the material does not carry; "
            f"its notes hold controllers {sorted(identified.cc_averages)}"
        )

    return round(average)


def _sample_on_axis(sample: SourceSample, config: DynamicAxisConfig) -> SourceSample:
    """``sample`` reporting the dynamic it was recorded at as its velocity."""
    return SourceSample.model_validate({**sample.model_dump(), "velocity": dynamic_of(sample, config)})


def _event_on_axis(event: NoteEvent, config: DynamicAxisConfig) -> NoteEvent:
    """``event`` reporting the dynamic it is played at as its velocity."""
    return NoteEvent.model_validate({**event.model_dump(), "velocity": dynamic_of(event, config)})


def _instrument_on_axis(instrument: InstrumentSpec, config: DynamicAxisConfig) -> InstrumentSpec:
    """``instrument`` with both of its sides read on the configured axis.

    The recordings and the material are keyed against one another all through the run -- a played note is
    routed to the recording nearest it in dynamic -- so both are read the same way or neither is.

    Raises:
        ValueError: when the axis leaves every recording at one position, which is a run reading a
            controller the library holds still.
    """
    samples = [_sample_on_axis(sample, config) for sample in instrument.samples]
    _states_dynamics(samples, instrument.id, config)
    return InstrumentSpec.model_validate(
        {
            **instrument.model_dump(),
            "samples": samples,
            "material": [_event_on_axis(event, config) for event in instrument.material],
        }
    )


def _states_dynamics(samples: Sequence[SourceSample], instrument_id: str, config: DynamicAxisConfig) -> None:
    """Hold the chosen axis to one the recordings actually vary along.

    A library that tracks a controller without playing it -- a percussion map recorded with the wheel
    down, most often -- answers every note at one position, and reading dynamics there would hand every
    key the same volume without anything failing. That is the one misconfiguration the material can be
    asked about, so it is asked here rather than heard later.

    Raises:
        ValueError: when every recording sits at one position of the axis.
    """
    positions = {sample.velocity for sample in samples}
    if len(positions) > _ONE_POSITION:
        return

    raise ValueError(
        f"{instrument_id} holds controller {config.controller} at {positions.pop()} on every one of its "
        f"{len(samples)} recordings, so it carries no dynamics; read this instrument on another axis"
    )


def on_axis(manifest: Manifest, config: DynamicAxisConfig) -> Manifest:
    """``manifest`` with every recording and every played note carrying the dynamic its axis states.

    A tracker pattern cell holds one dynamic column, so an instrument whose dynamics travel on a
    controller has that controller written there. Taking the reading once, at the way in, is what lets
    every stage afterwards work on an axis that varies with how loudly the material was played: the
    velocity map anchors on it, the layers cut it, deduplication keeps one recording per position of it,
    and a played note resolves to the recording nearest it along it.

    The controller average each note was recorded at stays on the note
    (:attr:`~optisample.model.SourceSample.cc_averages`), so a dataset a stage writes states the reading
    it was keyed on and reading it back on the same axis answers the same way.
    """
    if config.axis is DynamicAxis.VELOCITY:
        return manifest

    return Manifest.model_validate(
        {
            **manifest.model_dump(),
            "instruments": [_instrument_on_axis(instrument, config) for instrument in manifest.instruments],
        }
    )
