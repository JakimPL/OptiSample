from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from optisample.artifacts.documents.sample import (
    PAYLOAD_DTYPE,
    ProvenanceRecord,
    SampleDocument,
    read_sample,
    sample_decomposition,
    sample_document,
    sample_loops,
)
from optisample.artifacts.serialize import write_msgpack
from optisample.config import load_config
from optisample.dsp.envelope import Decomposition, decompose, level_reading
from optisample.dsp.level import Clock, read_level, unit_level
from optisample.dsp.loop import Loop, LoopQuality
from optisample.dsp.series import Readings
from optisample.dsp.surrogate import SettledLoop
from optisample.keys import SampleKey
from optisample.loop.settle import StoredLoop
from optisample.music import midi_to_freq
from tests.conftest import TEST_CONFIG_DIR

SR = 22_050
_PITCH = 60
_FRAMES = 4_096
_EDGE = 200  # frames read off each end to compare how far a curve falls across the note
_DECAY_RATE = 40.0  # nepers a second, steep enough that the tone all but dies inside the fixture's span
_QUALITY = LoopQuality(seam_step=0.4, level_drift_db=1.25, spectral_distance=3.5)
_PROVENANCE = ProvenanceRecord(stage="loop", instrument_id="pad", index=3, source="0003_C-4_v100.wav")


def _falloff(curve: NDArray[np.float64]) -> float:
    """How far a curve comes down across the note: its opening peak over its closing one."""
    return float(np.max(np.abs(curve[:_EDGE])) / np.max(np.abs(curve[-_EDGE:])))


@pytest.fixture
def key() -> SampleKey:
    return SampleKey(_PITCH, 100)


@pytest.fixture
def signal() -> NDArray[np.float64]:
    """A tone that decays steeply, so the level carries movement the carrier is normalised against."""
    times = np.arange(_FRAMES, dtype=np.float64) / SR
    return np.asarray(np.exp(-_DECAY_RATE * times) * np.sin(2.0 * np.pi * midi_to_freq(_PITCH) * times))


@pytest.fixture
def split(signal: NDArray[np.float64]) -> Decomposition:
    return decompose(signal, level_reading(SR, load_config(TEST_CONFIG_DIR).loop.envelope, midi_to_freq(_PITCH)))


@pytest.fixture
def offered() -> tuple[StoredLoop, ...]:
    """Two offers, one playing out flat and one falling on a ramp, cheapest stored span first."""
    return (
        StoredLoop(
            settled=SettledLoop(loop=Loop(start=512, end=1_024), level=unit_level(Clock.RECORDED)), quality=_QUALITY
        ),
        StoredLoop(
            settled=SettledLoop(
                loop=Loop(start=512, end=2_048),
                level=read_level(
                    Readings(values=np.asarray([0.0, -12.0]), seconds=np.asarray([0.1, 0.9])), Clock.RECORDED
                ),
            ),
            quality=_QUALITY,
        ),
    )


@pytest.fixture
def document(split: Decomposition, offered: tuple[StoredLoop, ...], key: SampleKey) -> SampleDocument:
    return sample_document(split, offered, key=key, sample_rate=SR, provenance=_PROVENANCE)


@pytest.fixture
def written(document: SampleDocument, tmp_path: Path) -> Path:
    path = tmp_path / "held.sample"
    write_msgpack(path, document)
    return path


def test_a_container_states_the_identity_and_the_rate_its_frames_are_counted_in(document: SampleDocument) -> None:
    """A container travels on its own, so what it holds is readable without the tree it was written in."""
    assert (document.key, document.root_pitch) == ("p060_C4_v100", _PITCH)
    assert (document.sample_rate, document.frames) == (SR, _FRAMES)
    assert document.provenance == _PROVENANCE


def test_the_pair_a_container_holds_puts_the_recording_back_together(
    written: Path, signal: NDArray[np.float64]
) -> None:
    """Level times carrier is the recording as the stage analysed it, which is what makes the split a carrier."""
    recombined = sample_decomposition(read_sample(written)).recombined()

    assert recombined == pytest.approx(signal, abs=1e-6)


def test_the_carrier_a_container_holds_stands_at_one_level_where_the_recording_falls_away(
    document: SampleDocument, signal: NDArray[np.float64]
) -> None:
    """Spending the whole depth of a stored grid on the waveform is the reason to carry the carrier."""
    assert _falloff(signal) > 100.0
    assert _falloff(sample_decomposition(document).carrier) < 4.0


def test_the_loops_a_container_states_read_back_as_the_encoder_receives_them(
    written: Path, offered: tuple[StoredLoop, ...]
) -> None:
    """The stage's decision travels with the very audio it was measured over, in the order params index it."""
    assert sample_loops(read_sample(written)) == tuple(stored.settled for stored in offered)


def test_a_recording_the_stage_found_no_loop_for_is_carried_stating_none(split: Decomposition, key: SampleKey) -> None:
    assert sample_document(split, (), key=key, sample_rate=SR, provenance=_PROVENANCE).loops == []


def test_a_payload_holding_other_than_the_frames_stated_is_refused(document: SampleDocument) -> None:
    """A truncated curve reads as a short one, so the frame count is checked rather than trusted."""
    truncated = {**document.model_dump(), "level": document.level[: -PAYLOAD_DTYPE.itemsize]}

    with pytest.raises(ValidationError, match="frames of p060_C4_v100 ask for"):
        SampleDocument.model_validate(truncated)
