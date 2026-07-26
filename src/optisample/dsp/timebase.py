def seconds_to_frames(duration_s: float, sample_rate: int) -> int:
    """Frame count spanning ``duration_s`` at ``sample_rate`` (rounded to nearest, floored at zero)."""
    return max(0, round(duration_s * sample_rate))
