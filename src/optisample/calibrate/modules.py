from typing import Final

from optisample.calibrate.context import NoteProbe
from optisample.config.render import PlaybackConfig
from optisample.dsp.surrogate import StoredSample
from optisample.dsp.timebase import row_seconds
from optisample.io.tracker.target import ExportTarget
from optisample.music import sounded_note
from optisample.optimize.export.material import CHANNELS
from trackmod.core.instruments.instrument import Instrument
from trackmod.core.instruments.keymap import KeyAssignment, routed_keymap
from trackmod.core.patterns.builder import PatternBuilder
from trackmod.core.patterns.cell import Cell
from trackmod.core.samples.sample import Sample
from trackmod.core.songs.order import OrderList
from trackmod.core.songs.playback import Playback
from trackmod.core.songs.song import Song
from trackmod.module.protocol import TrackerModule

DEFAULT_NAME: Final = "calib"

_CHANNEL: Final = 0
_INSTRUMENT: Final = 0
_SAMPLE: Final = 0
_ROW_MARGIN: Final = 2
_ORDERS: Final = 1


def _probe_rows(
    probe: NoteProbe,
    seconds_per_row: float,
    target: ExportTarget,
) -> int:
    """A pattern tall enough to outlast the probe, kept within the heights the format accepts."""
    held = int(probe.duration_s / seconds_per_row) + _ROW_MARGIN
    return min(target.max_rows, max(target.min_rows, held))


def single_note_module(
    stored: StoredSample,
    probe: NoteProbe,
    playback: PlaybackConfig,
    target: ExportTarget,
    *,
    name: str = DEFAULT_NAME,
) -> TrackerModule:
    """Wrap ``stored`` in a minimal one-sample, one-note module that plays ``probe`` from the start.

    The probed key is routed to the sample sounding the note that transposes it from the recording's
    own pitch, so triggering it shifts pitch exactly as the surrogate does. The note is held for the
    whole pattern, which outlasts ``probe.duration_s`` -- callers trim the render to the duration they
    asked for.
    """
    key = target.key(probe.pitch)
    seconds_per_row = row_seconds(playback.speed, playback.tempo)
    builder = PatternBuilder(
        rows=_probe_rows(probe, seconds_per_row, target),
        channels=CHANNELS,
    )
    builder.place(
        0,
        _CHANNEL,
        Cell(note=key, instrument=_INSTRUMENT, volume=probe.volume),
    )
    keymap = routed_keymap(
        {
            key: KeyAssignment(
                sample=_SAMPLE,
                note=sounded_note(
                    key,
                    target.key(stored.root_pitch),
                ),
            )
        },
    )
    song = Song(
        name=name,
        channels=CHANNELS,
        patterns=(builder.build(),),
        order=OrderList.sequential(_ORDERS),
        instruments=(Instrument(name=name, keymap=keymap),),
        samples=(Sample(name=name, pcm=stored.pcm, rate=stored.sample_rate, depth=stored.depth),),
        playback=Playback(speed=playback.speed, tempo=playback.tempo),
    )
    return target.bind(song)
