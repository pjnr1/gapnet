"""
Ramp interface functions
"""
from functools import partial
import numpy as np
import operator
from typing import Protocol, List, Annotated

from .functions import linear_ramp_func

GammaTPair = Annotated[List[float], 2]
"""
Type specification for gamma_t pair, used in functions with onset and offset
"""


class RampFunction(Protocol):
    """
    Protocol signature for ramp functions
    """
    def __call__(self, t: np.ndarray, gamma_t: float, width: float) -> np.ndarray: ...


def ramp(t: np.ndarray,
         gamma_t: float = 0.0,
         width: float = 2.5e-3,
         ramp_function: RampFunction = linear_ramp_func,
         invert: bool = False,
         dtype: np.dtype = None) -> np.ndarray:
    """
    Create an on-ramp centered at `gamma_t` using `t` as the time-vector. The output will match the shape of `t`.
    The function to create the ramp is passed with ramp_function. The function must be defined as::

     def <ramp_function>(
         t [numpy array, (cut from `t`)],
         gamma_t [float, (temporal center of the ramp)]
         width [float, (width of the ramp)]) -> [numpy array of same size as input]

    The numpy array parsed to the function fits the dimension from the cutout::

     (t < (gamma_t + width / 2)) & (t >= (gamma_t - width / 2))


    @param t:
        time vector
    @param gamma_t:
        center position of the ramp (in the same unit as the time vector)
    @param width:
        width of the ramp (in same unit as the time vector)
    @param ramp_function:
        the function to create the ramp
    @param invert:
        whether to invert the output (ramp on or ramp off)
    @param dtype:
        data type of output vector, defaults to the datatype of t
    @return:
        vector of the same shape as t

    """

    def _within_ramp_range(x):
        return (x < (gamma_t + width / 2)) & (x >= (gamma_t - width / 2))

    def _above_range(x):
        return x > (gamma_t - width / 2)

    if dtype is None:
        dtype = t.dtype

    out = np.zeros(t.shape, dtype=dtype)
    out[_above_range(t)] = np.ones(out[_above_range(t)].shape[0])
    out[_within_ramp_range(t)] = ramp_function(t=t[_within_ramp_range(t)],
                                               gamma_t=gamma_t,
                                               width=width)

    if invert:
        out[_within_ramp_range(t)] = np.flip(out[_within_ramp_range(t)])
        out[~_within_ramp_range(t)] = 1 - out[~_within_ramp_range(t)]

    return out


def ramp_onoff(t: np.ndarray,
               gamma_t: GammaTPair = None,
               width: float = 2.5e-3,
               ramp_function: RampFunction = linear_ramp_func,
               invert: bool = False,
               dtype: np.dtype = None) -> np.ndarray:
    """
    Takes two time points (gamma_t) and generates an onset+offset-ramp, by generating two ramps over the time-vector
    (see ramp) and combining by multiplying (ramp1(t) * ramp2(t)) or addition, when invert=True (ramp1(t)+ramp2(t))

    @param t:
        time vector
    @param gamma_t:
        center positions of the onset/offset-ramps (in the same unit as the time vector)
    @param width:
        width of the ramp (in same unit as the time vector)
    @param ramp_function:
        the function to create the ramp
    @param invert:
        whether to invert the output
    @param dtype:
        data type of output vector, defaults to the datatype of t
    @return:
        vector of the same shape as t

    """
    if gamma_t is None:
        gamma_t = [-0.001, 0.001]
    i = [False, True]

    if invert:
        i = [True, False]

    # Partially set ramp arguments for readability
    _ramp = partial(ramp,
                    width=width,
                    ramp_function=ramp_function,
                    dtype=dtype)

    _op = operator.add if invert else operator.mul

    return _op(_ramp(t=t, gamma_t=gamma_t[0], invert=i[0]),
               _ramp(t=t, gamma_t=gamma_t[1], invert=i[1]))


def ramp_onoff_with_gap(t,
                        stim_gamma_t: GammaTPair = None,
                        gap_gamma_t: GammaTPair = None,
                        width: float = 2.5e-3,
                        ramp_function: RampFunction = linear_ramp_func,
                        dtype: np.dtype = None):
    """
    Combines two instances of ramp_onoff to create a "gate" for some signal with onset and offset- and a gap in between.

    @param t:
        time vector
    @param gap_gamma_t   :
        center positions of the onset/offset-ramps of the gap (in the same unit as the time vector)
    @param stim_gamma_t  :
        center positions of the onset/offset-ramps of the signal (in the same unit as the time vector)
    @param width         :
        width of the ramp (in same unit as the time vector)
    @param ramp_function :
        the function to create the ramp
    @param dtype         :
        data type of output vector, defaults to the datatype of t
    @return:
    """
    if stim_gamma_t is None:
        stim_gamma_t = [0.1, 0.6]

    _ramp = partial(ramp_onoff,
                    width=width,
                    ramp_function=ramp_function,
                    dtype=dtype)

    # Create stimulus ramps / gate
    stimulus_gate = _ramp(t=t,
                          gamma_t=stim_gamma_t)

    if gap_gamma_t is None:
        # Return early with just the stimulus, if no gap is specified
        return stimulus_gate

    # Create gap ramps / gate (if any)
    gap_gate = _ramp(t=t,
                     gamma_t=gap_gamma_t,
                     invert=True)

    return stimulus_gate * gap_gate
