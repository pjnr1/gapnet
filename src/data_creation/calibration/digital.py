from typing import Annotated, TypeVar, Iterable
from typing_tools.annotations import check_annotations
from typing_tools.annotation_checkers import ValueRange

import numpy as np

Number = TypeVar('Number', float, int)
T = TypeVar('T', float, Iterable[float], np.ndarray)


@check_annotations
def quantise(x: T, n_bits: Annotated[int, ValueRange(min_value=2)] = 16) -> T:
    """

    @param x:
        range to quantise (expected values are in range [-1.0, 1.0])
    @param n_bits:
        Number of bits (at least 2 is required)
    @return:
    """
    bit_multiplier = n_bits**2 / 2
    y = x
    if isinstance(x, np.ndarray):
        y = x.copy()
    y *= bit_multiplier
    y = np.floor(y)
    y /= bit_multiplier

    return y
