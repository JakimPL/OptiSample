from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from optisample.config.reduce import TrimConfig
from optisample.dsp.levels import peak_amplitude
from optisample.metrics.base import Signal
from optisample.model import InstrumentSpec, NoteEvent
from optisample.optimize.reduce.keys import SampleKey

_SILENT: Final = 0  # a recording reaching the floor nowhere holds no content to keep


def content_frames(signal: Signal, floor: float) -> int:
    """Frames of ``signal`` up to and including the last one reaching ``floor``.

    Where a recording stops carrying anything: past the last frame at or above the floor the decay stays
    under it, so a note recorded into a long silence is measured by the stretch that sounds.
    """
    reaching = np.flatnonzero(np.abs(np.asarray(signal, dtype=np.float64)) >= floor)
    return int(reaching[-1]) + 1 if reaching.size else _SILENT


def carries_signal(signal: Signal, floor: float) -> bool:
    """Whether ``signal`` reaches ``floor`` anywhere, which is what a recording does to earn a slot."""
    return peak_amplitude(signal) >= floor


def trim_recording(signal: Signal, sample_rate: int, config: TrimConfig) -> Signal:
    """``signal`` cut to the span worth storing: its own content, bounded by the configured length.

    Both bounds shorten from the end, so the kept span runs to whichever comes first -- the frame the
    decay falls under :attr:`~optisample.config.reduce.TrimConfig.tail_floor`, or
    :attr:`~optisample.config.reduce.TrimConfig.max_length_s`. What survives is what every later stage
    reads: the reference the objective scores a note against, and the clip a plan may store for it.
    """
    length_limit = round(config.max_length_s * sample_rate)
    return signal[: min(content_frames(signal, config.tail_floor), length_limit)]


@dataclass(frozen=True)
class RecordingScreen:
    """What admitting the recordings cost: the ones left out, and the material that left unplayable.

    ``silenced`` names the survivors whose peak stayed under
    :attr:`~optisample.config.reduce.TrimConfig.silence_floor`, so the dataset holds no audio for them.
    ``unplayable`` are the pitches those losses stripped of every recording, and ``dropped_notes`` how
    many played notes went with them, which is what the material lost.
    """

    silenced: tuple[SampleKey, ...]
    unplayable: tuple[int, ...]
    dropped_notes: int

    @property
    def admitted_everything(self) -> bool:
        """Whether every survivor carried signal, leaving the material exactly as it was ingested."""
        return not self.silenced


NO_SCREEN: Final = RecordingScreen(silenced=(), unplayable=(), dropped_notes=0)


@dataclass(frozen=True)
class ScreenedInstrument:
    """An instrument narrowed to the notes its kept recordings answer for, beside what that removed."""

    instrument: InstrumentSpec
    screen: RecordingScreen


def _dropped(material: Sequence[NoteEvent], recorded: Collection[int]) -> list[NoteEvent]:
    """The notes played at pitches no kept recording covers, which nothing can be scored against."""
    return [event for event in material if event.pitch not in recorded]


def screen_instrument(
    instrument: InstrumentSpec,
    recorded: Collection[int],
    silenced: Sequence[SampleKey],
) -> ScreenedInstrument:
    """``instrument`` playing only the pitches ``recorded`` holds audio for, and what that cost.

    Dropping a recording for carrying no signal can leave a pitch with nothing to serve it. The notes
    there have no reference to be scored against and no sample to sound through, so they leave the
    material with the recording, and the screen states how many did.

    Raises:
        ValueError: when the screen leaves the instrument with no playable note at all, which means
            every recording it listed was silent.
    """
    playable = [event for event in instrument.material if event.pitch in recorded]
    if not playable:
        raise ValueError(f"instrument {instrument.id!r} has no recording carrying signal for any pitch it plays")

    dropped = _dropped(instrument.material, recorded)
    return ScreenedInstrument(
        instrument=instrument.model_copy(update={"material": playable}),
        screen=RecordingScreen(
            silenced=tuple(silenced),
            unplayable=tuple(sorted({event.pitch for event in dropped})),
            dropped_notes=sum(event.count for event in dropped),
        ),
    )
