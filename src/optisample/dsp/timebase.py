from trackmod.spec.clock import TICK_DENOMINATOR, TICK_NUMERATOR


def seconds_to_frames(duration_s: float, sample_rate: int) -> int:
    """Frame count spanning ``duration_s`` at ``sample_rate`` (rounded to nearest, floored at zero)."""
    return max(0, round(duration_s * sample_rate))


def row_seconds(speed: int, tempo: int) -> float:
    """Seconds one pattern row lasts: ``speed`` ticks, each ``5 / (2 * tempo)`` seconds long.

    Both tracker formats run on this one clock, so everything that lays material out in rows -- the
    exporter and the calibration probe alike -- reads a row's duration from here.
    """
    return speed * TICK_NUMERATOR / (TICK_DENOMINATOR * tempo)
