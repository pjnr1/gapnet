"""
Gap detection experiment by Moore et al., 1993
U{DOI: 10.1121/1.406815 <https://asa.scitation.org/doi/10.1121/1.406815>}

Stimulus used is puretone with background masker


They used the Khron-Hite 3550 filter for lowpassing the noise, which has a B{24 dB/Octave} slope
U{source <https://www.valuetronics.com/product/3550-krohn-hite-filter-used>}.

Signal was ramped with cosine squared ramp with 50ms rise/fall

"""
import os

import numpy as np

from data_creation.calibration.transform import spl2a
from data_creation.filters.filters import butterworth_lowpass
from data_creation.stimulus.noise import white_noise
from data_creation.time.time import get_sampling_frequency
from data_creation.calibration.spectrum import get_spectrum
from data_creation.calibration.spectrum import get_signal
from data_creation.calibration.spectrum import get_spectrum_from_path
from data_creation.calibration.digital import quantise
from data_creation.stimulus.puretone import puretone_syncedphase
from data_creation.ramp.ramp import ramp_onoff_with_gap


SIGNAL_LEVEL = 80
"""
Level of the sinusoid used in the original paper (they always used 80 dB SPL)
"""

SIGNAL_FREQUENCY = 1e3
"""
Frequency of the sinusoid (they always used 1kHz)
"""

SIGNAL_TO_NOISE_RATIO = 34
"""
Level difference between signal and noise in dB SPL
"""

SIGNAL_DURATION = 400e-3
"""
Target length of the signal (see paper for details)
"""

NOISE_SPECTRUM_PATH = os.path.join(os.path.dirname(__file__),
                                   'data/moore_et_al_1988__noise_profile_1khz.csv')
"""
Path to spectrum shape of the masking noise
"""


def generate_stimulus(t,
                      gap_length: float = 0,
                      signal_start: float = 200e-3) -> np.ndarray:
    """

    @param t:
        time-vector
    @param gap_length:
        length of the gap,
    @param signal_start:
        time-point for starting the signal

    @return:
    """

    # Generate signal
    fs = get_sampling_frequency(None, t)
    signal = spl2a(SIGNAL_LEVEL) * puretone_syncedphase(t, signal_start, SIGNAL_FREQUENCY)
    signal = quantise(signal, 12)

    # Generate gate
    signal_duration_minus_length_ms = 1e3 * (SIGNAL_DURATION - gap_length)
    stim_lengths_ms = [round(signal_duration_minus_length_ms / 2) * 2 / 2,
                       signal_duration_minus_length_ms / 2]
    stim_lengths = [x * 1e-3 for x in stim_lengths_ms]
    stim_gamma_t = [signal_start,
                    signal_start + stim_lengths[0] + gap_length + stim_lengths[1]]

    gap_gamma_t = None
    if gap_length > 0:
        gap_gamma_t = [signal_start + stim_lengths[0],
                       signal_start + stim_lengths[0] + gap_length]

    gate = ramp_onoff_with_gap(t,
                               stim_gamma_t=stim_gamma_t,
                               gap_gamma_t=gap_gamma_t,
                               width=0.0)

    # Generate noise
    f, noise_profile = get_spectrum_from_path(t,
                                              NOISE_SPECTRUM_PATH)
    noise_profile = np.concatenate((noise_profile[:-1], np.flip(noise_profile.conj()[1:])))
    _, X_noise = get_spectrum(t,
                              white_noise(t, spl2a(SIGNAL_LEVEL - SIGNAL_TO_NOISE_RATIO)),
                              get_abs=False)
    noise = get_signal(X_noise + noise_profile)
    noise = np.abs(noise)

    # Generate mixture
    mixture = gate * signal + noise
    mixture = butterworth_lowpass(mixture, 4e3, fs, order=16)  # order of 16 approximates 100 dB / octave cutoff

    return mixture
