"""The byte-budget unit conversion the reports and the budget split share.

What a record or a stored sample costs is a tracker-format fact, so it is read from that format's
``trackmod`` storage table; what remains here is the KiB scale the budget is expressed in.
"""

import math
from typing import Final

_BYTES_PER_KIB: Final = 1024


def bytes_to_kib(num_bytes: float) -> float:
    return num_bytes / _BYTES_PER_KIB


def kib_to_bytes(kib: float) -> int:
    return math.ceil(kib * _BYTES_PER_KIB)
