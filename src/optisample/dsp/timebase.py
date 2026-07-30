from trackmod.spec.clock import TICK_DENOMINATOR, TICK_NUMERATOR


def seconds_to_frames(duration_s: float, sample_rate: int) -> int:
    """Frame count spanning ``duration_s`` at ``sample_rate`` (rounded to nearest, floored at zero)."""
    return max(0, round(duration_s * sample_rate))


def tick_seconds(tempo: int) -> float:
    """Seconds one tracker tick lasts: ``5 / (2 * tempo)``.

    The tick is the unit both formats count envelope breakpoints in, so an envelope written for one tempo
    runs at another rate under a different one. Everything placing a breakpoint in time reads its length
    from here, which is what lets the tempo an envelope was fitted at be stated once and travel with it.
    """
    return TICK_NUMERATOR / (TICK_DENOMINATOR * tempo)


def row_seconds(speed: int, tempo: int) -> float:
    """Seconds one pattern row lasts: ``speed`` ticks, each :func:`tick_seconds` long.

    Both tracker formats run on this one clock, so everything that lays material out in rows -- the
    exporter and the calibration probe alike -- reads a row's duration from here.
    """
    return speed * tick_seconds(tempo)
