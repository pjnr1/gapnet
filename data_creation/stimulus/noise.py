from typing import Protocol, List

import numpy as np


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


def broadband_noise_n(fs: float = 1e6,
                      amp: float = 0.1,
                      duration: float = 1.0,
                      eps: float = 1e-10,
                      freq_range: List[float] = None) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / fs)
    return broadband_noise(t, amp, eps, freq_range)


def broadband_noise(t: np.ndarray,
                    amp: float = 0.1,
                    eps: float = 1e-10,
                    freq_range: List[float] = None,
                    dtype: type = float) -> np.ndarray:
    if freq_range is None:
        freq_range = [200, 20000]
    if dtype is None:
        dtype = t.dtype
    n_t = t.shape[0]
    duration = t[-1] - t[0]

    # Frequencies for broadband noise
    freqs = 2 * np.pi * np.arange(freq_range[0], freq_range[1], 1.0 / duration)
    n_freqs = freqs.shape[0]

    # Generator random phases for each frequency
    phase = 2 * np.pi * np.random.rand(n_freqs)

    # Pre-allocate signal vector
    signal = np.zeros(n_t, dtype=dtype)

    # Generate noise by summing sine-waves with the randomly generated phases
    for i in range(n_freqs):
        signal += np.sin(freqs[i] * t + phase[i])

    signal *= amp
    signal /= np.max(np.abs(signal)) + eps

    return signal
