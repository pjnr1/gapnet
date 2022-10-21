"""
Functions for calibration using spectrum

"""
from __future__ import annotations

from typing import Tuple, Union, Annotated

import scipy.fft

from typing_tools.annotation_checkers import PathExists
from typing_tools.annotations import check_annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from data_creation.calibration.transform import a2spl, spl2a
from data_creation.time.time import get_sampling_frequency


def get_spectrum(t: np.ndarray,
                 x: np.ndarray,
                 get_abs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectrum and show positive frequency space in SPL

    @param t:
        time vector
    @param x:
        signal
    @param get_abs:
        whether to return the absolute value of the spectrum or not
    @return:
        Tuple(frequency vector, spectrum)
    """
    return get_spectrum_fs(x, get_sampling_frequency(None, t), get_abs)


def get_spectrum_fs(x: np.ndarray,
                    fs: Union[float, int],
                    get_abs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spectrum and show positive frequency space in SPL

    @param x:
        signal
    @param fs:
        sampling frequency (only used for frequency vector)
    @param get_abs:
        whether to return the absolute value of the spectrum or not
    @return:
        Tuple(frequency vector, spectrum)
    """
    n = len(x)
    f = (np.fft.fftfreq(n, 1 / fs))
    X = np.fft.fft(x)
    if get_abs:
        X = np.abs(X)
    X /= n  # normalise by number of samples and sampling frequency
    X = a2spl(X)
    return f, X


def get_signal(x: np.ndarray) -> np.ndarray:
    """
    Reverse of get_spectrum
    @param x:
    @return:
    """
    x = spl2a(x)
    x /= 2
    x *= len(x)
    x = np.fft.ifft(x)
    return x


def get_impulse_response_from_spectrum(f: np.ndarray,
                                       x: np.ndarray):
    assert(len(f) == len(x))
    # Convert to amplitude from dB
    x = spl2a(x)

    # if spectrum is one-sided
    if min(f) >= 0:
        x *= 2 * (len(x) - 1)
        x = np.fft.irfft(x)
    else:
        x *= len(x)
        x = np.fft.ifft(x)
    return x


def spectrum_level_from_signal(t: np.ndarray,
                               x: np.ndarray,
                               frequency_range: Union[None, Annotated[list[float], 2]] = None) -> float | np.ndarray:
    f"""
    Combines C{get_spectrum} and C{spectrum_level} for a complete function to compute the spectrum level of a given
    signal
    
    @param t: 
        time vector
    @param x: 
        signal
    @param frequency_range: 
        frequency range to compute spectrum level from, if None, the full spectrum is used
    @return: 
    """
    f, spectrum = get_spectrum(t=t,
                               x=x,
                               get_abs=True)
    return spectrum_level(f=f,
                          spectrum=spectrum,
                          frequency_range=frequency_range)


def spectrum_level_from_signal_fs(x: np.ndarray,
                                  fs: Union[float, int],
                                  frequency_range: Union[None, Annotated[list[float], 2]] = None) -> float | np.ndarray:
    f"""
    Combines C{get_spectrum_fs} and C{spectrum_level} for a complete function to compute the spectrum level of a given
    signal

    @param x: 
        signal
    @param fs:
        sampling frequency
    @param frequency_range: 
        frequency range to compute spectrum level from, if None, the full spectrum is used
    @return: 
    """
    f, spectrum = get_spectrum_fs(x=x,
                                  fs=fs,
                                  get_abs=True)
    return spectrum_level(f=f,
                          spectrum=spectrum,
                          frequency_range=frequency_range)


def spectrum_level(f: np.ndarray,
                   spectrum: np.ndarray,
                   frequency_range: Union[None, Annotated[list[float], 2]]) -> Union[float, np.ndarray]:
    """
    Computes the spectrum level within a given frequency range

    @param f:
        frequency vector
    @param spectrum:
        spectrum vector
    @param frequency_range:
        frequency range to compute spectrum level from, if None, the full spectrum is used
    @return:
    """
    if frequency_range is None:
        return np.mean(spectrum)

    # If only positive axis is used for computing, compensate by doubling the spectrum level
    sl = np.mean(spectrum[(f >= frequency_range[0]) & (f <= frequency_range[1])])
    if frequency_range[0] > 0 and frequency_range[1] > 0:
        sl *= 2
    return sl


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


def normalise_by_spectrum_level_fs(x: np.ndarray,
                                   fs: Union[float, int],
                                   target_level: float,
                                   frequency_range=None) -> np.ndarray:
    """
    Normalises a given signal by its spectrum level within a defined frequency range. If

    @param x:
        signal
    @param fs:
        sampling frequency
    @param target_level:
        target spectrum level in dB SPL
    @param frequency_range:
        frequency range for spectrum level calculation
    @return:
        normalised signal
    """
    f, X = get_spectrum_fs(x * spl2a(0), fs)
    sl = spectrum_level(f, X, frequency_range)
    return x * spl2a(target_level - sl)


@check_annotations
def get_spectrum_from_path(t: np.ndarray, path: Annotated[str, PathExists]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    fs = get_sampling_frequency(None, t)
    f = np.fft.rfftfreq(len(t), 1 / fs)
    X = interp1d(x=df['f'].to_numpy(),
                 y=df['x'].to_numpy(),
                 fill_value='extrapolate')(f)
    X -= max(X)  # Normalise to zero dB
    return f, X
