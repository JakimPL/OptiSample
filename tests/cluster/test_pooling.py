from __future__ import annotations

import numpy as np
import pytest

from optisample.cluster.pooling import anchor_stack, frames_spanning, pooled_rows, window_mean

HOP_S = 0.01
DEPTHS = (0.0, 6.0, 12.0, 24.0)


def _falling(frames: int, *, decay_db: float, rising: int = 0) -> np.ndarray:
    """A level curve climbing over ``rising`` frames and then falling ``decay_db`` over the rest."""
    climb = np.linspace(-decay_db, 0.0, rising, endpoint=False, dtype=np.float64)
    fall = np.linspace(0.0, -decay_db, frames - rising, dtype=np.float64)
    return np.concatenate((climb, fall))


def test_a_stretch_covers_at_least_one_frame_of_the_series_it_is_read_on() -> None:
    assert frames_spanning(0.1, HOP_S) == 10
    assert frames_spanning(1e-9, HOP_S) == 1


def test_a_window_reads_the_rows_it_covers() -> None:
    rows = np.arange(20.0).reshape(5, 4)

    assert np.allclose(window_mean(rows, start=1, end=3), rows[1:3].mean(axis=0))


def test_a_window_reaching_past_either_end_reads_the_material_that_is_there() -> None:
    rows = np.arange(20.0).reshape(5, 4)

    assert np.allclose(window_mean(rows, start=-4, end=2), rows[:2].mean(axis=0))
    assert np.allclose(window_mean(rows, start=3, end=99), rows[3:].mean(axis=0))


def test_a_window_landing_on_no_rows_reads_zeros() -> None:
    rows = np.arange(20.0).reshape(5, 4)

    assert np.allclose(window_mean(rows, start=9, end=12), np.zeros(4))
    assert np.allclose(window_mean(np.zeros((0, 4)), start=0, end=3), np.zeros(4))


def test_each_anchor_sits_where_the_level_first_stands_that_far_under_the_peak() -> None:
    stack = anchor_stack(_falling(100, decay_db=40.0), DEPTHS, HOP_S)

    assert stack.peak_frame == 0
    assert stack.depths == len(DEPTHS)
    assert list(stack.reached) == [True, True, True, True]
    assert list(stack.frames) == [0, 15, 30, 60]
    assert np.allclose(stack.seconds, np.asarray([0, 15, 30, 60]) * HOP_S)


def test_the_search_opens_at_the_peak_so_an_anchor_is_a_moment_of_the_decline() -> None:
    """A climb into the peak stands under it too, and the anchors read the fall rather than the rise."""
    stack = anchor_stack(_falling(120, decay_db=40.0, rising=20), DEPTHS, HOP_S)

    assert stack.peak_frame == 20
    assert stack.attack_s == pytest.approx(20 * HOP_S)
    assert all(frame >= stack.peak_frame for frame in stack.frames)


def test_a_recording_short_of_a_depth_holds_the_frame_it_last_reached() -> None:
    """A note falling into its own floor holds the shape it had there, which is the honest fill."""
    stack = anchor_stack(_falling(100, decay_db=15.0), DEPTHS, HOP_S)

    assert list(stack.reached) == [True, True, True, False]
    assert stack.frames[-1] == stack.frames[-2]


def test_a_recording_reaching_no_depth_past_its_peak_holds_the_peak() -> None:
    stack = anchor_stack(_falling(50, decay_db=2.0), (6.0, 12.0), HOP_S)

    assert not stack.reached.any()
    assert list(stack.frames) == [0, 0]
    assert stack.attack_s == 0.0


def test_a_level_curve_holding_no_readings_anchors_to_nothing() -> None:
    with pytest.raises(ValueError, match="at least one level reading"):
        anchor_stack(np.zeros(0), DEPTHS, HOP_S)


def test_each_anchor_reads_the_stretch_of_material_centred_on_it() -> None:
    rows = np.arange(400.0).reshape(100, 4)
    stack = anchor_stack(_falling(100, decay_db=40.0), DEPTHS, HOP_S)
    pooled = pooled_rows(rows, stack, span=6)

    assert pooled.shape == (len(DEPTHS), 4)
    assert np.allclose(pooled[2], rows[27:33].mean(axis=0))
    assert np.allclose(pooled[0], rows[0:3].mean(axis=0))  # the first anchor reads what sits past it


def test_truncating_a_falling_note_leaves_every_anchor_it_still_reaches_where_it_stood() -> None:
    """This is what makes a two-second take and an eight-second take of one note comparable."""
    level = _falling(200, decay_db=60.0)
    rows = np.arange(800.0).reshape(200, 4)
    whole = anchor_stack(level, DEPTHS, HOP_S)
    cut = anchor_stack(level[:70], DEPTHS, HOP_S)
    shared = whole.reached & cut.reached

    assert list(cut.reached) == [True, True, True, False]
    assert np.array_equal(whole.frames[shared], cut.frames[shared])
    assert np.allclose(pooled_rows(rows, whole, span=6)[shared], pooled_rows(rows[:70], cut, span=6)[shared])
