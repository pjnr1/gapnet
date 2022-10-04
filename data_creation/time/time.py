import numpy as np
from typing import List


def generate_time_vector(start: float, end: float, fs: int) -> np.ndarray:
    return np.arange(fs * start, fs * end) / fs


def start_and_duration_to_gamma_t(start: float, duration: float) -> List[float]:
    return [start, start + duration]


def get_sampling_frequency(fs: int, t: List[float]) -> int:
    if fs is None:
        return np.abs(1.0 / (t[1] - t[0]))
    return fs
