from collections.abc import Sequence
from typing import Final

from optisample.dsp.surrogate import StoredSample
from optisample.dsp.timebase import tick_seconds
from optisample.model import NoteEvent
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.envelope import NO_ENVELOPE, shared_decay, volume_envelope
from optisample.optimize.export.material import CHANNELS, Voicing, material_patterns
from optisample.optimize.export.samples import plan_samples
from optisample.optimize.layers.slots import ONE_SLOT, InstrumentSlot, SlotLayout, plan_slots
from optisample.optimize.plans import SINGLE_LAYER, StrategyPlan
from optisample.optimize.tasks import StoredRecordings
from trackmod.core.envelopes.envelope import Envelope
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import Keymap
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.module.protocol import TrackerModule

_NAME_CHARS: Final = 22  # the narrowest instrument-name field a target format keeps, FastTracker 2's
_SHORTEST_ID: Final = 1  # instrument-id characters a name keeps however long the axes it states are


def _axes(layout: SlotLayout, index: int) -> str:
    """The axes the plan split, as the suffix telling one written instrument from its siblings apart.

    A name states each axis the plan actually split: the velocity band once several layers are stored,
    and the stretch of keyboard once a band is written as several instruments. One layer written whole is
    told apart by nothing, so it answers with an empty suffix and carries the instrument's own name.
    """
    slot = layout.slots[index]
    stated = []
    if layout.layers.count > SINGLE_LAYER:
        stated.append(slot.band.label)

    if len(layout.layer_slots(slot.layer)) > ONE_SLOT:
        stated.append(slot.span)

    return " ".join(stated)


def instrument_name(instrument_id: str, layout: SlotLayout, index: int) -> str:
    """What the tracker's instrument list calls one written instrument.

    The whole name fits the narrowest field a target format keeps for it, and the axes the plan split
    (:func:`_axes`) are what the room is kept for, so a list read in the tracker states which dynamics
    and which keys play through which instrument.
    """
    axes = _axes(layout, index)
    if not axes:
        return instrument_id[:_NAME_CHARS]

    return f"{instrument_id[: max(_SHORTEST_ID, _NAME_CHARS - len(axes) - 1)]} {axes}"


def slot_envelope(
    slot: InstrumentSlot,
    stored: Sequence[StoredSample],
    context: ExportContext,
) -> Envelope | None:
    """The volume curve one written instrument plays every voice it starts down by.

    A looped sample holds one level for as long as a note is held, so the decline the recording made past
    that point lives in the envelope rather than the waveform. The envelope belongs to the instrument and
    the slot holds several samples, so one shape answers for all of them
    (:func:`~optisample.optimize.export.envelope.shared_decay`).
    """
    shared = shared_decay([stored[index].decay for index in slot.samples])
    if shared is None:
        return NO_ENVELOPE

    return volume_envelope(
        shared,
        tick_s=tick_seconds(context.playback.tempo),
        release_s=context.envelope.release_s,
        last_tick=context.target.max_envelope_tick,
    )


def _slot_instruments(
    plan: StrategyPlan,
    layout: SlotLayout,
    keymaps: Sequence[Keymap],
    stored: Sequence[StoredSample],
    context: ExportContext,
) -> tuple[Instrument, ...]:
    """One instrument per written slot, so a note's dynamic and pitch name the one it plays."""
    return tuple(
        Instrument(
            name=instrument_name(plan.instrument_id, layout, index),
            keymap=keymap,
            volume_envelope=slot_envelope(layout.slots[index], stored, context),
        )
        for index, keymap in enumerate(keymaps)
    )


def build_song(
    plan: StrategyPlan,
    recordings: StoredRecordings,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> Song:
    """Assemble the format-agnostic song either strategy's plan describes.

    The plan supplies the stored samples and the keys they serve; the format decides how many instruments
    those samples are written as (:func:`~optisample.optimize.layers.slots.pack_slots`); the material
    supplies the patterns that audition them. Everything else -- the song name, the single channel, the
    clock -- is the same for both strategies, so it lives here once.
    """
    layout = plan_slots(plan, context.target)
    planned = plan_samples(plan, layout, recordings, context)
    voicing = Voicing(layout=layout, velocity_map=plan.velocity_map)
    patterns, order = material_patterns(material, voicing, context.playback, context.target)
    return Song(
        name=plan.instrument_id,
        channels=CHANNELS,
        patterns=patterns,
        order=order,
        instruments=_slot_instruments(plan, layout, planned.keymaps, planned.stored, context),
        samples=planned.samples,
        playback=Playback(speed=context.playback.speed, tempo=context.playback.tempo),
    )


def build_module(
    plan: StrategyPlan,
    recordings: StoredRecordings,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> TrackerModule:
    """Assemble a complete module from either strategy's plan, bound to the target tracker format."""
    return context.target.bind(build_song(plan, recordings, material, context))
