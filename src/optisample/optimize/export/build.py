from collections.abc import Sequence

from optisample.model import NoteEvent
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.material import CHANNELS, Voicing, material_patterns
from optisample.optimize.export.samples import plan_samples
from optisample.optimize.layers.bands import VelocityLayers
from optisample.optimize.plans import SINGLE_LAYER, StrategyPlan
from optisample.optimize.tasks import AudioMap
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import Keymap
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.module.protocol import TrackerModule


def instrument_name(instrument_id: str, layers: VelocityLayers, layer: int) -> str:
    """What the tracker's instrument list calls one written layer.

    A split storing several layers names each by the velocity band it answers for, so the list states
    which dynamics play through which. One layer answers every dynamic, so it carries the instrument's
    own name.
    """
    if layers.count == SINGLE_LAYER:
        return instrument_id

    return f"{instrument_id} {layers.bands[layer].label}"


def _layer_instruments(plan: StrategyPlan, keymaps: Sequence[Keymap]) -> tuple[Instrument, ...]:
    """One instrument per velocity layer, in band order, so a note's dynamic names the one it plays."""
    return tuple(
        Instrument(name=instrument_name(plan.instrument_id, plan.layers, layer), keymap=keymap)
        for layer, keymap in enumerate(keymaps)
    )


def build_song(
    plan: StrategyPlan,
    audio: AudioMap,
    sample_rate: int,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> Song:
    """Assemble the format-agnostic song either strategy's plan describes.

    The plan supplies the stored samples, the keys they serve and the velocity layers they are written
    as; the material supplies the patterns that audition them. Everything else -- the song name, the
    single channel, the clock -- is the same for both strategies, so it lives here once.
    """
    samples, keymaps = plan_samples(plan, audio, sample_rate, context)
    voicing = Voicing(layers=plan.layers, velocity_map=plan.velocity_map)
    patterns, order = material_patterns(material, voicing, context.playback, context.target)
    return Song(
        name=plan.instrument_id,
        channels=CHANNELS,
        patterns=patterns,
        order=order,
        instruments=_layer_instruments(plan, keymaps),
        samples=samples,
        playback=Playback(speed=context.playback.speed, tempo=context.playback.tempo),
    )


def build_module(
    plan: StrategyPlan,
    audio: AudioMap,
    sample_rate: int,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> TrackerModule:
    """Assemble a complete module from either strategy's plan, bound to the target tracker format."""
    return context.target.bind(build_song(plan, audio, sample_rate, material, context))
