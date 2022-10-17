"""
Functions for calibration using spectrum

"""
from typing import Tuple, Union, Annotated

import numpy as np

from data_creation.calibration.transform import a2spl, spl2a
from data_creation.time.time import get_sampling_frequency


def get_spectrum(t: np.ndarray,
                 x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectrum and show positive frequency space in SPL

    @param t:
        time vector
    @param x:
        signal
    @return:
        Tuple(frequency vector, spectrum)
    """
    return get_spectrum_n(t.shape[-1], x, get_sampling_frequency(None, t))


def get_spectrum_n(n: Union[float, int],
                   x: np.ndarray,
                   fs: Union[float, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectrum and show positive frequency space in SPL

    @param n:
        length in samples
    @param x:
        signal
    @param fs:
        sampling frequency (only used for frequency vector)
    @return:
        Tuple(frequency vector, spectrum)
    """
    x_cut = len(x) // 2
    f = (np.fft.fftfreq(n, 1 / fs))[:x_cut]
    X = np.abs((np.fft.fft(x)))[:x_cut]
    X /= len(x)  # normalise by number of samples and sampling frequency
    X *= 2  # Compensate for one-sided
    X = a2spl(X)
    return f, X


def spectrum_level(f: np.ndarray,
                   X: np.ndarray,
                   fr: Union[None, Annotated[list[float], 2]]) -> Union[float, np.ndarray]:
    """
    Computes the spectrum level within a given frequency range

    @param f:
        frequency vector
    @param X:
        spectrum vector
    @param fr:
        frequency range to compute, if None, the full spectrum is used
    @return:
    """
    if fr is None:
        return np.mean(X)

    return np.mean(X[(f >= fr[0]) & (f <= fr[1])])


def normalise_by_spectrum_level(t: np.ndarray,
                                x: np.ndarray,
                                target_level: float,
                                frequency_range=None) -> np.ndarray:
    """
    Normalises a given signal by its spectrum level within a defined frequency range. If

    @param t:
        time vector
    @param x:
        signal
    @param target_level:
        target spectrum level in dB SPL
    @param frequency_range:
        frequency range for spectrum level calculation
    @return:
        normalised signal
    """
    f, X = get_spectrum(t, x * spl2a(0))
    sl = spectrum_level(f, X, frequency_range)
    return x * spl2a(target_level - sl)
