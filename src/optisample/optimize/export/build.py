from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from optisample.dsp.level import Clock, Level, curve_level, loudest_db, written_level
from optisample.dsp.trajectory import SharedTrajectory
from optisample.io.tracker.envelope import NO_ENVELOPE, shape_nodes, volume_envelope
from optisample.model import NoteEvent
from optisample.optimize.export.carriers import plan_trajectories
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.material import CHANNELS, Voicing, material_patterns
from optisample.optimize.export.samples import PlannedSamples, plan_samples
from optisample.optimize.export.voices import NO_SHAPE, PlannedVoices, PlayedVoices, instrument_shapes
from optisample.optimize.layers.slots import ONE_SLOT, SlotLayout, plan_slots
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


def slot_level(shape: SharedTrajectory | None) -> Level | None:
    """The level one written instrument plays every voice it starts down by.

    A looped sample holds one level for as long as a note is held, so the decline the recording made past
    that point lives in the envelope rather than the waveform. The envelope belongs to the instrument and
    the slot answers several keys, so ``shape`` is the one trajectory fitted to all of them
    (:func:`~optisample.optimize.export.voices.instrument_shape`), carried here as the level the algebra
    composes. It stands on the played clock, which is the clock a tracker walks its envelope on.
    """
    if shape is NO_SHAPE:
        return NO_SHAPE

    return curve_level(shape.curve, Clock.PLAYED)


def slot_envelope(level: Level | None, context: ExportContext, *, reference_db: float) -> Envelope | None:
    """``level`` written onto the format's own grid, against the reference every instrument shares.

    A volume envelope only attenuates, so the level is stated relative to ``reference_db``
    (:func:`~optisample.dsp.level.written.written_level`) before it reaches the format
    (:func:`~optisample.io.tracker.envelope.volume_envelope`). An instrument carrying no level leaves its
    voices at the level their own waveforms hold.
    """
    if level is NO_SHAPE:
        return NO_ENVELOPE

    return volume_envelope(written_level(level, reference_db=reference_db), context.envelope_grid)


def _slot_instruments(
    plan: StrategyPlan,
    layout: SlotLayout,
    keymaps: Sequence[Keymap],
    envelopes: Sequence[Envelope | None],
) -> tuple[Instrument, ...]:
    """One instrument per written slot, so a note's dynamic and pitch name the one it plays."""
    return tuple(
        Instrument(
            name=instrument_name(plan.instrument_id, layout, index),
            keymap=keymap,
            volume_envelope=envelopes[index],
        )
        for index, keymap in enumerate(keymaps)
    )


@dataclass(frozen=True)
class WrittenVoices:
    """What a plan is written as: the stored samples, and the curve each instrument plays them down by."""

    planned: PlannedSamples
    envelopes: tuple[Envelope | None, ...]


def _written(shapes: Sequence[SharedTrajectory | None], context: ExportContext) -> tuple[Envelope | None, ...]:
    """Each shape written onto the format's grid, all against the one level the plan states as unity.

    Taking the reference across the whole plan is what keeps two velocity layers exactly as far apart as
    their gains and recordings put them: each shape stated against its own peak would move the pair by
    exactly the difference between those peaks. A plan whose instruments carry no shape names unity itself,
    there being nothing to stand under.
    """
    levels = tuple(slot_level(shape) for shape in shapes)
    reference_db = loudest_db([level for level in levels if level is not NO_SHAPE])
    return tuple(slot_envelope(level, context, reference_db=reference_db) for level in levels)


def _carried(
    plan: StrategyPlan,
    layout: SlotLayout,
    recordings: StoredRecordings,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> WrittenVoices:
    """The carrier order: fit each instrument's curve from its recordings, write it, then store what it leaves.

    The shape is settled before anything is encoded (:func:`~optisample.optimize.export.carriers.plan_trajectories`),
    so each waveform can be its recording divided by the gain that written curve applies. Nothing iterates:
    the level the curve cannot state is exactly what stays in the waveform.
    """
    shapes = plan_trajectories(
        layout,
        PlannedVoices(recordings=recordings, material=material, velocity_map=plan.velocity_map),
        nodes=shape_nodes(context.target.envelope_point_bound),
    )
    envelopes = _written(shapes, context)
    return WrittenVoices(
        planned=plan_samples(plan, layout, recordings, context, envelopes=envelopes),
        envelopes=envelopes,
    )


def _levelled(
    plan: StrategyPlan,
    layout: SlotLayout,
    recordings: StoredRecordings,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> WrittenVoices:
    """The order a run has always used: store each recording as it was played, then fit what its level leaves.

    The stored waveform carries its own decline, so the curve an instrument plays is fitted to the residual
    the encoder and the written levels left behind
    (:func:`~optisample.optimize.export.voices.instrument_shapes`).
    """
    planned = plan_samples(plan, layout, recordings, context, envelopes=(NO_ENVELOPE,) * layout.count)
    sources = PlayedVoices(
        recordings=recordings,
        material=material,
        velocity_map=plan.velocity_map,
        gains=planned.gains,
    )
    shapes = instrument_shapes(
        layout,
        planned.stored,
        sources,
        nodes=shape_nodes(context.target.envelope_point_bound),
    )
    return WrittenVoices(planned=planned, envelopes=_written(shapes, context))


def written_voices(
    plan: StrategyPlan,
    layout: SlotLayout,
    recordings: StoredRecordings,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> WrittenVoices:
    """The stored samples a plan is written as, beside the curve each instrument plays them down by.

    Which way round the two settle is what ``context.carrier`` states, and every caller re-encoding a plan
    goes through here, so the bytes a module carries and the bytes an artifact reports are the same bytes.
    """
    if context.carrier:
        return _carried(plan, layout, recordings, material, context)

    return _levelled(plan, layout, recordings, material, context)


def build_song(
    plan: StrategyPlan,
    recordings: StoredRecordings,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> Song:
    """Assemble the format-agnostic song either strategy's plan describes.

    The plan supplies the stored samples and the keys they serve; the format decides how many instruments
    those samples are written as (:func:`~optisample.optimize.layers.slots.pack_slots`); the material
    supplies the patterns that audition them and, beside the recordings, the trajectory each instrument's
    own envelope is fitted to. Which way round those two settle is what ``context.carrier`` states:
    :func:`_carried` fits the curve first and stores what it leaves, :func:`_levelled` stores the recording
    and fits what its level leaves. Everything else -- the song name, the single channel, the clock -- is
    the same for both strategies, so it lives here once.
    """
    layout = plan_slots(plan, context.target)
    written = written_voices(plan, layout, recordings, material, context)
    voicing = Voicing(layout=layout, velocity_map=plan.velocity_map)
    patterns, order = material_patterns(material, voicing, context.playback, context.target)
    return Song(
        name=plan.instrument_id,
        channels=CHANNELS,
        patterns=patterns,
        order=order,
        instruments=_slot_instruments(plan, layout, written.planned.keymaps, written.envelopes),
        samples=written.planned.samples,
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
