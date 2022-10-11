from typing import Protocol, List, Annotated, Union

import numpy as np

from data_creation.time.time import get_sampling_frequency


class WhiteNoiseFunction(Protocol):
    def __call__(self, t: int) -> np.ndarray: ...


def uniform_white_noise(n: int) -> np.ndarray:
    return np.random.random(n) - 0.5


def normal_distributed_white_noise(n: int) -> np.ndarray:
    return np.random.randn(n)


def white_noise_n(n: int, generator: WhiteNoiseFunction = uniform_white_noise) -> np.ndarray:
    return generator(n)


def white_noise(t, generator: WhiteNoiseFunction = uniform_white_noise) -> np.ndarray:
    return white_noise_n(t.shape[0], generator)


def broadband_noise_n(fs: Union[float, int] = 1e6,
                      amp: float = 0.1,
                      duration: float = 1.0,
                      freq_range: List[float] = None,
                      dtype: type = float) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / float(fs))
    return broadband_noise(t, amp, freq_range, dtype)


def broadband_noise(t: np.ndarray,
                    amp: float = 0.1,
                    freq_range: Annotated[List[float], 2] = None,
                    dtype: type = float) -> np.ndarray:
    n_t = t.shape[0]
    fs = get_sampling_frequency(None, t)
    if freq_range is None:
        freq_range = np.array([1/8, 3/8]) * fs
    if dtype is None:
        dtype = t.dtype

    # Frequencies for broadband noise
    f = np.fft.fftfreq(n_t, t[1] - t[0])  # Get frequencies
    f = f[(f > 0) & (f >= freq_range[0]) & (f <= freq_range[1])]  # Filter frequencies
    f = 2 * np.pi * f  # Convert from omega to f (multiplying by 2pi)
    n_f = f.shape[0]  # Get number of frequencies

    # Generator random phases for each frequency
    phase = 2 * np.pi * np.random.rand(n_f)

    # Pre-allocate signal vector
    signal = np.zeros(n_t, dtype=dtype)

    # Generate noise by summing sine-waves with the randomly generated phases
    for i in range(n_f):
        signal += np.sin(f[i] * t + phase[i])

    signal *= amp
    # signal /= np.max(np.abs(signal)) + eps

    return signal
