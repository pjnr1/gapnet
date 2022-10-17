"""
Selection of ramping functions. These functions are not meant to be interfaced with by the user, for generating ramps,
one should use the function C{ramp(...)} located in ramp.py

Note: a ramp_func-function must have comply with following:
 - the signature must be C{Callable[t:np.ndarray, gamma_t:float, width:float]->np.ndarray}
 - the function name ends with C{_ramp_func}
"""
import numpy as np


def linear_ramp_func(t: np.ndarray,
                     gamma_t: float,
                     width: float) -> np.ndarray:
    """
    Creates a linear ramp from 0.0 to 1.0 for the full duration of t

    Note that this ramp function ignores gamma_t and width

    @param t:
        time vector
    @param gamma_t:
        _unused_
    @param width:
        _unused_

    @return:
    """
    return np.linspace(0, 1, t.shape[0])


def cosine_squared_ramp_func(t: np.ndarray,
                             gamma_t: float,
                             width: float) -> np.ndarray:
    """
    Creates a cosine squared ramp centered in gamma_t with defined width, relative to the time-vector t:

    M{ramp = (cos(S{pi} * (t - (gamma_t - width / 2) + width) / (2 * width))^2}

    @param t:
        time vector
    @param gamma_t:
        center of ramp
    @param width:
        width of ramp

    @return:
    """
    x = t - (gamma_t - width / 2)
    return np.power(np.cos(np.pi * (x + width) / (2 * width)), 2)


def _cosine_sum(t: np.ndarray,
                gamma_t: float,
                width: float,
                a0: float) -> np.ndarray:
    """
    Private helper used for cosine-sum ramps, i.e. Hann and Hamming

    Computed from:

      - a0 - (1 - a0)cos(S{pi} * x / width)

    @param t:
        time vector
    @param gamma_t:
        center of ramp
    @param width:
        width of ramp
    @param a0:
        Offset parameter, see function description for details
    @return:
    """
    x = t - (gamma_t - width / 2)
    return a0 - (1.0 - a0) * np.cos(np.pi * x / width)


def hann_ramp_func(t: np.ndarray,
                   gamma_t: float,
                   width: float) -> np.ndarray:
    """
    Ramp based on the Hann window function

    @param t:
        time vector
    @param gamma_t:
        center of ramp
    @param width:
        width of ramp
    @return:
    """
    return _cosine_sum(t=t, gamma_t=gamma_t, width=width, a0=0.5)


def hamming_ramp_func(t: np.ndarray,
                      gamma_t: float,
                      width: float) -> np.ndarray:
    """
    Ramp based on the Hamming window function

    @param t:
        time vector
    @param gamma_t:
        center of ramp
    @param width:
        width of ramp
    @return:
    """
    return _cosine_sum(t=t, gamma_t=gamma_t, width=width, a0=25.0 / 46.0)


def gaussian_ramp_func(t: np.ndarray,
                       gamma_t: float,
                       width: float) -> np.ndarray:
    """
    Ramp function based on the gaussian gate used by
    U{Schneider and Hamstra, 1999 <https://asa.scitation.org/doi/10.1121/1.427062>}

    @param t:
        time vector
    @param gamma_t:
        center of ramp
    @param width:
        width of ramp
    @return:
    """
    gamma_t += width / 2
    width /= 4
    return np.exp(-((t - gamma_t) ** 2) / (2 * np.power(width, 2)))
