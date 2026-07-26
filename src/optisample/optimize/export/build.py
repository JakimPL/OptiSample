from collections.abc import Sequence

from optisample.model import NoteEvent
from optisample.optimize.export.context import ExportContext
from optisample.optimize.export.material import CHANNELS, material_patterns
from optisample.optimize.export.samples import plan_samples
from optisample.optimize.plans import StrategyPlan
from optisample.optimize.tasks import AudioMap
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.module.protocol import TrackerModule


def build_song(
    plan: StrategyPlan,
    audio: AudioMap,
    sample_rate: int,
    material: Sequence[NoteEvent],
    context: ExportContext,
) -> Song:
    """Assemble the format-agnostic song either strategy's plan describes.

    The plan supplies the stored samples and the keys they serve; the material supplies the patterns
    that audition them. Everything else -- the song and instrument name, the single channel, the clock
    -- is the same for both strategies, so it lives here once.
    """
    samples, keymap = plan_samples(
        plan.instrument_id,
        plan.sample_units(),
        audio,
        sample_rate,
        context,
    )
    patterns, order = material_patterns(
        material,
        plan.velocity_map,
        context.playback,
        context.target,
    )
    return Song(
        name=plan.instrument_id,
        channels=CHANNELS,
        patterns=patterns,
        order=order,
        instruments=(Instrument(name=plan.instrument_id, keymap=keymap),),
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
