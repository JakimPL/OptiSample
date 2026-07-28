from typing import Final

DEFAULT_SEED: Final = 137  # the run entropy every stage dithers from when a caller states none of its own
SURROGATE_SEED: Final = 0  # what a proxy encode dithers from, so a clip prices the same in any order
