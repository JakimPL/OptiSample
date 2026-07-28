import math
from typing import Final

_BYTES_PER_KIB: Final = 1024


def bytes_to_kib(num_bytes: float) -> float:
    return num_bytes / _BYTES_PER_KIB


def kib_to_bytes(kib: float) -> int:
    return math.ceil(kib * _BYTES_PER_KIB)
