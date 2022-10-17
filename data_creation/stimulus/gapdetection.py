import numpy as np
from typing import Annotated, List, Protocol

GapLengthRange = Annotated[List[float], 2]


class GapLengthGenerator(Protocol):
    """
    Protocol signature for ramp functions
    """

    def __call__(self, n: int, r: GapLengthRange) -> np.ndarray: ...


def linear_gap_length_generator(n, r) -> np.ndarray:
    return np.linspace(r[0], r[1], n)


def log_gap_length_generator(n, r) -> np.ndarray:
    x = np.linspace(1.0, 41.0, n)

    # generate
    gaps = np.power(10, (x / 20.0))

    # normalise
    gaps -= gaps[0]
    gaps /= gaps.max()

    # map
    gaps *= r[1] - r[0]
    gaps += r[0]

    return gaps


def get_gap_lengths(n, r=None, generator: GapLengthGenerator = linear_gap_length_generator):
    """
    Generate n gap lengths between r[0] and r[1] using a GapLengthGenerator
    default generator is linear_gap_length_generator

    @param n:
        number of gap lengths
    @param r:
        range of the gap lengths
    @param generator:
        generator function
    @return:
        list of gap lengths
    """
    if r is None:
        r = [1.0, 100.0]

    return generator(n, r)
