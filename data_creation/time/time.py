import numpy as np
from typing import Annotated, List, Union

from typing_tools.annotation_checkers import ExactLen


def generate_time_vector(start: float, end: float, fs: int) -> np.ndarray:
    """
    Generates an array with time points at sampling rate fs with start- and end-time

    @param start:
        start time in seconds
    @param end:
        end time in seconds (non-inclusive)
    @param fs:
        sampling rate
    @return:
        numpy array with time points
    """
    return np.arange(fs * start, fs * end) / fs


def start_and_duration_to_gamma_t(start: float, duration: float) -> Annotated[List[float], ExactLen(2)]:
    """
    Small helper to get start- and end-time from start and duration

    @param start:
    @param duration:
    @return:
        list with two indices; start and end
    """
    return [start, start + duration]


def position_and_length_to_gamma_t(position: float, length: float) -> Annotated[List[float], ExactLen(2)]:
    """
    Small helper to get start- and end-time from start and duration

    @param position:
    @param length:
    @return:
        list with two indices; start and end
    """
    return [position - (length / 2), position + (length / 2)]


def get_sampling_frequency(fs: Union[int, None], t: Union[np.ndarray, List[float]]) -> int:
    """
    Compute sampling frequency from t when fs is None, else simply returns fs

    @param fs:
        sampling frequency
    @param t:
        time vector
    @return:

    """
    if fs is None:
        return np.abs(1.0 / (t[1] - t[0]))
    return fs
