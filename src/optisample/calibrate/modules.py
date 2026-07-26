from typing import Final

from optisample.calibrate.context import NoteProbe
from optisample.dsp.surrogate import StoredSample
from optisample.io.it_writer import (
    MAX_ROWS,
    ITCell,
    ITInstrument,
    ITModule,
    ITPattern,
    ITPlayback,
    ITSample,
    identity_note_map,
)
from optisample.optimize.export import c5speed_for_pitch, row_seconds

_ROW_MARGIN: Final = 2


def single_note_module(
    stored: StoredSample, probe: NoteProbe, playback: ITPlayback, *, name: str = "calib"
) -> ITModule:
    """Wrap ``stored`` in a minimal one-sample, one-note module that plays ``probe`` from the start.

    The sample's ``C5Speed`` is set so its root key plays at the natural rate; triggering ``probe.pitch``
    then transposes exactly as the surrogate does. The note is held (no cut) and the pattern is sized to
    outlast ``probe.duration_s`` -- callers trim the render to the duration they asked for.
    """
    seconds_per_row = row_seconds(playback)
    rows = min(MAX_ROWS, int(probe.duration_s / seconds_per_row) + _ROW_MARGIN)
    sample = ITSample(
        name=name,
        pcm=stored.pcm,
        depth_bits=stored.depth_bits,
        c5speed=c5speed_for_pitch(stored.sample_rate, stored.root_pitch),
    )
    instrument = ITInstrument(name=name, note_map=identity_note_map({probe.pitch: 1}))
    pattern = ITPattern(rows=rows, cells=((0, 0, ITCell(note=probe.pitch, instrument=1, volume=probe.volume)),))
    return ITModule(
        name=name, samples=(sample,), instruments=(instrument,), patterns=(pattern,), orders=(0,), playback=playback
    )
