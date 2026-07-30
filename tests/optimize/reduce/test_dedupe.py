from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from optisample.config.loop import LoopConfig
from optisample.config.reduce import DedupeKey, ReduceConfig
from optisample.io.audio import write_wav
from optisample.keys import SampleKey
from optisample.model import InstrumentSpec, NoteEvent, SourceSample
from optisample.optimize.reduce.dedupe import required_duration_s, select_recordings
from optisample.progress import NO_PROGRESS

SR = 22_050
NOTE_S = 0.02  # short enough that the loop floor, not the transposition headroom, sets the requirement
PADDING_S = 0.5  # what every split in the padding test pads a recording by in total
UNDER_FLOOR_S = 0.1  # how far a length is set under the floor where the test asks for one that falls short

ReduceFactory = Callable[..., ReduceConfig]


@pytest.fixture
def wav(tmp_path: Path) -> Callable[[str, float], Path]:
    """Factory: an all-zero WAV of a stated length, so only its header matters to dedup."""

    def _wav(name: str, duration_s: float) -> Path:
        path = tmp_path / name
        write_wav(path, np.zeros(round(duration_s * SR), dtype=np.float64), SR)
        return path

    return _wav


def instrument(samples: Sequence[SourceSample], material: Sequence[NoteEvent]) -> InstrumentSpec:
    return InstrumentSpec(id="piano", budget_kb=64.0, samples=list(samples), material=list(material))


def one_note(pitch: int = 60, velocity: int = 100, duration_s: float = NOTE_S) -> list[NoteEvent]:
    return [NoteEvent(pitch=pitch, velocity=velocity, duration_s=duration_s)]


# --- the coverage rule ----------------------------------------------------------------------------


def test_required_duration_scales_the_longest_note_by_the_transposition_headroom(
    reduce: ReduceFactory, loop_config: LoopConfig
) -> None:
    config = reduce(dedupe={"transposition_headroom_semitones": 12}, trim={"max_length_s": 30.0})
    # An octave up runs the sample at twice the speed, so it has to hold twice the seconds.
    assert required_duration_s(4.0, config, loop_config.geometry) == pytest.approx(8.0)


def test_required_duration_never_falls_below_what_loop_detection_needs(
    reduce: ReduceFactory, loop_config: LoopConfig, loop_floor_s: float
) -> None:
    config = reduce(dedupe={"transposition_headroom_semitones": 0})
    assert required_duration_s(0.01, config, loop_config.geometry) == pytest.approx(loop_floor_s)


def test_no_headroom_asks_only_for_the_note_itself(reduce: ReduceFactory, loop_config: LoopConfig) -> None:
    config = reduce(dedupe={"transposition_headroom_semitones": 0})
    assert required_duration_s(4.0, config, loop_config.geometry) == pytest.approx(4.0)


def test_the_trim_bounds_what_a_recording_is_ever_asked_to_hold(reduce: ReduceFactory, loop_config: LoopConfig) -> None:
    """A recording is only asked for the span the trim keeps, however long the note it serves runs."""
    config = reduce(dedupe={"transposition_headroom_semitones": 12}, trim={"max_length_s": 5.0})
    assert required_duration_s(4.0, config, loop_config.geometry) == pytest.approx(5.0)


def test_the_loop_floor_holds_the_requirement_up_through_the_trim(
    reduce: ReduceFactory, loop_config: LoopConfig, loop_floor_s: float
) -> None:
    """Loop detection needs its room whatever the trim says, so the floor outranks a shorter bound."""
    config = reduce(trim={"max_length_s": loop_floor_s / 2})
    assert required_duration_s(4.0, config, loop_config.geometry) == pytest.approx(loop_floor_s)


# --- which recording survives ---------------------------------------------------------------------


def test_keeps_the_shortest_recording_that_covers_the_material(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path], loop_floor_s: float
) -> None:
    lengths = {
        "0000_long.wav": 1.5,
        "0001_enough.wav": loop_floor_s + 0.1,
        "0002_short.wav": loop_floor_s - UNDER_FLOOR_S,
    }
    samples = [SourceSample(file=wav(name, length), pitch=60, velocity=100) for name, length in lengths.items()]

    selections = select_recordings(instrument(samples, one_note()), reduce(), loop_config.geometry, NO_PROGRESS)

    assert len(selections) == 1
    kept = selections[0]
    assert kept.sample.file.name == "0001_enough.wav"  # the least material that still covers the note
    assert kept.considered == 3
    assert kept.covers_material


def test_keeps_the_longest_recording_when_none_covers_the_material(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path], loop_floor_s: float
) -> None:
    samples = [
        SourceSample(file=wav("0000_shorter.wav", loop_floor_s / 3), pitch=60, velocity=100),
        SourceSample(file=wav("0001_longer.wav", loop_floor_s / 2), pitch=60, velocity=100),
    ]

    kept = select_recordings(instrument(samples, one_note()), reduce(), loop_config.geometry, NO_PROGRESS)[0]

    assert kept.sample.file.name == "0001_longer.wav"  # as much of the note as was ever recorded
    assert not kept.covers_material
    assert kept.duration_s < kept.required_duration_s


@pytest.mark.parametrize(("lead_in_s", "trail_out_s"), [(PADDING_S, 0.0), (0.0, PADDING_S), (0.2, 0.3)])
def test_the_padding_around_a_note_is_removed_before_a_recording_is_measured(
    reduce: ReduceFactory,
    loop_config: LoopConfig,
    wav: Callable[[str, float], Path],
    loop_floor_s: float,
    lead_in_s: float,
    trail_out_s: float,
) -> None:
    """A recording is ranked on the span from its onset to its release, which is all of it that is stored."""
    recorded_s = loop_floor_s + PADDING_S - UNDER_FLOOR_S  # padding aside, a span the floor outruns
    padded = SourceSample(
        file=wav("0000_padded.wav", recorded_s),
        pitch=60,
        velocity=100,
        lead_in_s=lead_in_s,
        trail_out_s=trail_out_s,
    )

    kept = select_recordings(instrument([padded], one_note()), reduce(), loop_config.geometry, NO_PROGRESS)[0]

    assert kept.duration_s == pytest.approx(recorded_s - lead_in_s - trail_out_s, abs=1e-3)
    assert not kept.covers_material  # the padding is not part of the note


@pytest.mark.parametrize(("lead_in_s", "trail_out_s"), [(0.5, 0.0), (0.0, 0.5), (0.3, 0.3)])
def test_padding_longer_than_the_recording_measures_as_empty(
    reduce: ReduceFactory,
    loop_config: LoopConfig,
    wav: Callable[[str, float], Path],
    lead_in_s: float,
    trail_out_s: float,
) -> None:
    clipped = SourceSample(
        file=wav("0000_clipped.wav", 0.2),
        pitch=60,
        velocity=100,
        lead_in_s=lead_in_s,
        trail_out_s=trail_out_s,
    )

    kept = select_recordings(instrument([clipped], one_note()), reduce(), loop_config.geometry, NO_PROGRESS)[0]

    assert kept.duration_s == 0.0


def test_a_pitch_the_material_never_plays_only_has_to_satisfy_the_loop_floor(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path], loop_floor_s: float
) -> None:
    unplayed = SourceSample(file=wav("0000_unplayed.wav", loop_floor_s + 0.1), pitch=72, velocity=100)
    played = SourceSample(file=wav("0001_played.wav", 2.0), pitch=60, velocity=100)

    selections = select_recordings(
        instrument([unplayed, played], one_note()), reduce(), loop_config.geometry, NO_PROGRESS
    )

    by_pitch = {selection.key.pitch: selection for selection in selections}
    assert by_pitch[72].required_duration_s == pytest.approx(loop_floor_s)
    assert by_pitch[72].covers_material


# --- deterministic tie-breaking ---------------------------------------------------------------------


def test_equally_suitable_recordings_break_the_tie_on_the_render_index(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path]
) -> None:
    samples = [
        SourceSample(file=wav("0005_later.wav", 1.0), pitch=60, velocity=100),
        SourceSample(file=wav("0002_earlier.wav", 1.0), pitch=60, velocity=100),
    ]

    kept = select_recordings(instrument(samples, one_note()), reduce(), loop_config.geometry, NO_PROGRESS)[0]

    assert kept.sample.file.name == "0002_earlier.wav"  # listing order does not decide it


def test_a_filename_without_a_render_index_ranks_after_one_that_has_it(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path]
) -> None:
    samples = [
        SourceSample(file=wav("anonymous.wav", 1.0), pitch=60, velocity=100),
        SourceSample(file=wav("0009_indexed.wav", 1.0), pitch=60, velocity=100),
    ]

    kept = select_recordings(instrument(samples, one_note()), reduce(), loop_config.geometry, NO_PROGRESS)[0]

    assert kept.sample.file.name == "0009_indexed.wav"


def test_selections_come_back_in_key_order(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path]
) -> None:
    samples = [
        SourceSample(file=wav("0000_high.wav", 1.0), pitch=67, velocity=100),
        SourceSample(file=wav("0001_loud.wav", 1.0), pitch=60, velocity=110),
        SourceSample(file=wav("0002_quiet.wav", 1.0), pitch=60, velocity=40),
    ]
    material = [NoteEvent(pitch=pitch, velocity=100, duration_s=NOTE_S) for pitch in (60, 67)]

    selections = select_recordings(instrument(samples, material), reduce(), loop_config.geometry, NO_PROGRESS)

    assert [selection.key for selection in selections] == [SampleKey(60, 40), SampleKey(60, 110), SampleKey(67, 100)]


# --- what the key collapses -------------------------------------------------------------------------


def test_a_pitch_only_key_collapses_every_velocity_into_one_survivor(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path], loop_floor_s: float
) -> None:
    samples = [
        SourceSample(file=wav("0000_quiet.wav", 1.5), pitch=60, velocity=40),
        SourceSample(file=wav("0001_loud.wav", loop_floor_s + 0.1), pitch=60, velocity=110),
    ]
    config = reduce(dedupe={"key": DedupeKey.PITCH})

    selections = select_recordings(instrument(samples, one_note()), config, loop_config.geometry, NO_PROGRESS)

    assert len(selections) == 1
    assert selections[0].key == SampleKey(60, 110)  # the survivor still reports the velocity it was recorded at
    assert selections[0].considered == 2


def test_a_cc_key_keeps_timbral_variants_apart(
    reduce: ReduceFactory, loop_config: LoopConfig, wav: Callable[[str, float], Path]
) -> None:
    samples = [
        SourceSample(file=wav("0000_dark.wav", 1.0), pitch=60, velocity=100, cc_averages={1: 0.0}),
        SourceSample(file=wav("0001_bright.wav", 1.0), pitch=60, velocity=100, cc_averages={1: 100.0}),
    ]
    overrides = {"key": DedupeKey.PITCH_VELOCITY_CC, "cc_quantum": 10.0}

    variants = select_recordings(
        instrument(samples, one_note()), reduce(dedupe=overrides), loop_config.geometry, NO_PROGRESS
    )
    collapsed = select_recordings(instrument(samples, one_note()), reduce(), loop_config.geometry, NO_PROGRESS)

    assert [selection.key.cc for selection in variants] == [((1, 0),), ((1, 10),)]
    assert len(collapsed) == 1  # the default key does not distinguish the two takes
