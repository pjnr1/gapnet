from typing import Protocol, List, Annotated, Union

import numpy as np

from data_creation.calibration.transform import a2spl
from data_creation.calibration.spectrum import normalise_by_spectrum_level
from data_creation.time.time import get_sampling_frequency
from typing_tools.annotation_checkers import ExactLen


class WhiteNoiseFunction(Protocol):
    def __call__(self, t: int) -> np.ndarray: ...


def uniform_white_noise_generator(n: int) -> np.ndarray:
    return np.random.random(n) - 0.5


def normal_distributed_white_noise_generator(n: int) -> np.ndarray:
    return np.random.randn(n)


def white_noise_n(n: int,
                  generator: WhiteNoiseFunction = uniform_white_noise_generator,
                  dtype: type = float) -> np.ndarray:
    return generator(n).astype(dtype=dtype)


def white_noise(t: np.ndarray,
                amp: float = 0.1,
                generator: WhiteNoiseFunction = uniform_white_noise_generator,
                dtype: type = float) -> np.ndarray:
    """
    Generate white noise of the length of the time vector

    Example::

        >>> from data_creation.time.time import generate_time_vector
        >>> t = generate_time_vector(0.0, 1.0, 1000)
        >>> x = white_noise(t, 0.5)

    @param t:
        time vector
    @param amp:
        amplitude
    @param generator:
        generator function (most fulfill the L{WhiteNoiseFunction}.protocol)
    @param dtype:
        datatype
    @return:
        noise signal
    """
    x = white_noise_n(t.shape[0], generator, dtype=dtype)
    x = normalise_by_spectrum_level(t, x, a2spl(amp), None)
    return x


def broadband_noise_n(fs: Union[float, int] = 1e6,
                      amp: float = 0.1,
                      duration: float = 1.0,
                      freq_range: List[float] = None,
                      dtype: type = float) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / float(fs))
    return broadband_noise(t, amp, freq_range, dtype)


def broadband_noise(t: np.ndarray,
                    amp: float = 0.1,
                    freq_range: Annotated[List[float], ExactLen(2)] = None,
                    dtype: type = float) -> np.ndarray:
    """

    @param t:
        time vector
    @param amp:
        amplitude
    @param freq_range:
        desired frequency range
    @param dtype:
        desired datatype
    @return:
        noise signal
    """
    n_t = t.shape[0]
    fs = get_sampling_frequency(None, t)
    if freq_range is None:
        freq_range = np.array([1 / 8, 3 / 8]) * fs
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

    return signal
