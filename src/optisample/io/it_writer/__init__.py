"""Hand-rolled uncompressed Impulse Tracker (``.IT``) writer.

No mature Python IT writer exists, so this subpackage serializes the uncompressed IT format directly
from the ITTECH.TXT specification. It is deliberately *domain-agnostic*: it knows about samples,
instruments, patterns and byte layout, not about the optimizer -- the plan->module bridge lives in
:mod:`optisample.optimize.export`.

Layout produced (all multi-byte fields little-endian)::

    [192-B file header][orders + 0xFF][instr offsets][sample offsets][pattern offsets]
    [554-B instrument headers][80-B sample headers][packed patterns][sample PCM]

It emits mono signed 8/16-bit PCM with optional forward loops, disabled envelopes, and one instrument
per identity note map (each key plays its own dedicated sample at natural rate via a per-sample
``C5Speed``). IT214 compression is out of scope (GPL-licensing-gated); the optimizer's byte model is
the uncompressed size regardless.

The implementation is split by record: :mod:`~optisample.io.it_writer.constants` (format primitives),
:mod:`~optisample.io.it_writer.samples`, :mod:`~optisample.io.it_writer.instruments`,
:mod:`~optisample.io.it_writer.patterns`, and :mod:`~optisample.io.it_writer.module` (body assembly).
"""

from optisample.io.it_writer.constants import MAX_ROWS, NOTE_CUT, TICKS_PER_ROW_BASE, require_it_note
from optisample.io.it_writer.instruments import ITInstrument, identity_note_map
from optisample.io.it_writer.module import ITModule, write_it, write_it_module
from optisample.io.it_writer.patterns import ITCell, ITPattern, ITPlayback, it_playback
from optisample.io.it_writer.samples import ITSample

__all__ = [
    "MAX_ROWS",
    "NOTE_CUT",
    "TICKS_PER_ROW_BASE",
    "require_it_note",
    "ITCell",
    "ITInstrument",
    "ITModule",
    "ITPattern",
    "ITPlayback",
    "ITSample",
    "identity_note_map",
    "it_playback",
    "write_it",
    "write_it_module",
]
