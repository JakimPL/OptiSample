from __future__ import annotations

from dataclasses import dataclass

from optisample.dsp.envelope import Decomposition, Signal
from optisample.dsp.surrogate import SettledLoops
from optisample.keys import SampleKey


@dataclass(frozen=True)
class CarrierSource:
    """One recording a carrier instrument is written from, held as what it sounds like times how loud it is.

    ``decomposition`` is that split (:func:`~optisample.dsp.envelope.decompose`), which is what lets the
    level be handed to a volume envelope while the stored waveform keeps the timbre and spends its whole
    grid on it. ``loops`` holds the regions the loop stage settled over the recording, in the order an
    encoding indexes them, and ``loop_index`` names the one this sample is stored around -- left unset for a
    recording whose material a loop stands in for nowhere, which is stored as the span it plays.

    ``weight`` is the playing time the material gives this recording's key, which is the say it carries in
    the one shape its instrument plays every voice down by.
    """

    key: SampleKey
    root_pitch: int
    sample_rate: int
    decomposition: Decomposition
    loops: SettledLoops
    loop_index: int | None
    weight: float

    @property
    def frames(self) -> int:
        """How many frames the recording holds, which both halves of its split state."""
        return int(self.decomposition.level.size)

    @property
    def duration_s(self) -> float:
        """How long the recording sounds for, at the rate it was read at."""
        return self.frames / self.sample_rate

    @property
    def recording(self) -> Signal:
        """The recording itself, put back together from the split it is carried as."""
        return self.decomposition.recombined()
