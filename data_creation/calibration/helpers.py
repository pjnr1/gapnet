import numpy as np


def normalise_array(x: np.ndarray) -> np.ndarray:
    """
    Helper function to normalise array between 0.0 and 1.0

    @param x:
        the array
    @return:
        the normalised array
    """
    x -= x[0]
    x /= x.max(initial=0)
    return x
