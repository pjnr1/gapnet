"""
Gap detection experiment ment for training machines!

This particular experiment setup mimics the code convention for the other experiments, but is intended for creating the
training gap-in-babble training data.

It's slightly different from the other generators, as it returns both a gap and no-gap condition.
This is needed in training for forcing the model to learn features of the gap, rather than random occuring cues
from the generation process

"""
import os
from typing import Annotated, List, Union

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample_poly

from data_creation.ramp.ramp import ramp_onoff, RampFunction
from data_creation.ramp.functions import hann_ramp_func
from data_creation.stimulus.babble import generate_babble
from data_creation.time.time import get_sampling_frequency
from data_creation.time.time import start_and_duration_to_gamma_t
from data_creation.time.time import position_and_length_to_gamma_t
from typing_tools.annotation_checkers import MaxLen


def generate_stimulus(t,
                      babble_path: str,
                      n_talkers: int,
                      gap_position: float,
                      gap_duration: float,
                      noise_start: float = 0.1,
                      noise_duration: float = 0.5,
                      ramp_width: Union[float, Annotated[List[float], MaxLen(2)]] = 2.5e-3,
                      ramp_function: RampFunction = hann_ramp_func) -> (int, np.ndarray, np.ndarray):
    """

    @param t:
        time vector
    @param babble_path:
        path to either a pre-made babble-noise file or folder containing single-speaker recordings
    @param n_talkers:
        number of talkers, i.e. number of files to include from the folder containing single-speaker recordings
    @param gap_position:
        center position of the gap
    @param gap_duration:
        length of the gap
    @param noise_start:
        onset of the babble noise
    @param noise_duration:
        length of the babble noise
    @param ramp_width:
        width of both onset/offset and gap ramps if type=float, if type=List[float], first and second index are used for
        onset/offset and gap ramps respectively
    @param ramp_function:
        ramp function for generating the ramps (see L{data_creation.ramp})
    @return:
        start index of the babble cut, babble without gap, babble with gap
    """
    # Get sampling frequency from time-vector
    t_fs = get_sampling_frequency(None, t)

    # Get ramp widths
    if isinstance(ramp_width, list):
        ramp_width_babble = ramp_width[0]
        ramp_width_gap = ramp_width[1]
    else:
        ramp_width_babble = ramp_width
        ramp_width_gap = ramp_width

    # Load from either file or generate from folder
    fs = None
    data = None
    if os.path.isfile(babble_path):
        fs, data = read(babble_path)
    if os.path.isdir(babble_path):
        fs, data = generate_babble(babble_path, n_talkers)

    # Resample to match time vector sampling frequency
    if fs != t_fs:
        data = resample_poly(data, t_fs, fs)
    babble = data

    # Generate cut indices for the babble sound
    start_idx = np.random.randint(0, len(babble) - len(t))
    babble = babble[start_idx:start_idx + len(t)]

    # Ramp onset and offset
    babble *= ramp_onoff(t,
                         gamma_t=start_and_duration_to_gamma_t(noise_start,
                                                               noise_duration),
                         width=ramp_width_babble,
                         ramp_function=ramp_function)

    # Create version with gap
    babble_gap = babble * ramp_onoff(t,
                                     gamma_t=position_and_length_to_gamma_t(gap_position,
                                                                            gap_duration),
                                     width=ramp_width_gap,
                                     ramp_function=ramp_function,
                                     invert=True)

    return start_idx, babble, babble_gap
