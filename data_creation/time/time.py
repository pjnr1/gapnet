import numpy as np
from typing import List, Union


def generate_time_vector(start: float, end: float, fs: int) -> np.ndarray:
    """
    Generates an array with time points at sampling rate fs with start- and end-time

    @arg start:
        start time in seconds
    @arg end:
        end time in seconds (non-inclusive)
    @arg fs:
        sampling rate
    @return:
        numpy array with time points
    """
    return np.arange(fs * start, fs * end) / fs


def start_and_duration_to_gamma_t(start: float, duration: float) -> List[float]:
    """
    Small helper to get start- and end-time from start and duration

    @arg start:
    @arg duration:
    @return:
        list with two indices; start and end
    """
    return [start, start + duration]


def get_sampling_frequency(fs: Union[int, None], t: Union[np.ndarray, List[float]]) -> int:
    """
    Compute sampling frequency from t when fs is None, else simply returns fs

    @arg fs:
        sampling frequency
    @arg t:
        time vector
    @return:

    """
    if fs is None:
        return np.abs(1.0 / (t[1] - t[0]))
    return fs
