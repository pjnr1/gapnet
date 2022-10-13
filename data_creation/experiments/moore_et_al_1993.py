"""
Gap detection experiment by Moore et al., 1993
U{DOI: 10.1121/1.406815 <https://asa.scitation.org/doi/10.1121/1.406815>}

Stimulus used is puretone with background masker


They used the Khron-Hite 3550 filter for lowpassing the noise, which has a B{24 dB/Octave} slope
U{source <https://www.valuetronics.com/product/3550-krohn-hite-filter-used>}.

Signal was ramped with cosine squared ramp with 50ms rise/fall

"""
import numpy as np

from data_creation.calibration.transform import spl2a
from data_creation.filters.filters import butterworth_lowpass
from data_creation.ramp.functions import cosine_squared_ramp_func
from data_creation.ramp.ramp import ramp_onoff
from data_creation.stimulus.noise import white_noise
from data_creation.stimulus.puretone import puretone_randomphase
from data_creation.time.time import get_sampling_frequency


SIGNAL_LEVELS = [25, 40, 55, 70, 85]
"""
Levels of the sinusoid used in the original paper
"""

SIGNAL_FREQUENCIES = [100, 200, 400, 800, 1000, 2000]
"""
Frequencies of the sinusoid used in the original paper
"""

SIGNAL_TO_NOISE_RATIO = 40
"""
Level difference between signal and noise in dB SPL
"""


def generate_stimulus(t,
                      sinusoid_frequency: float = SIGNAL_FREQUENCIES[0],
                      sinusoid_level: float = SIGNAL_LEVELS[0],
                      gap_length: float = 0,
                      signal_start: float = 100e-3,
                      signal_length: float = 400e-3) -> np.ndarray:
    """

    @arg t:
    @arg sinusoid_frequency:
    @arg sinusoid_level:
    @arg gap_length:
    @arg signal_start:
    @arg signal_length:
    @return:
    """
    fs = get_sampling_frequency(None, t)
    signal = spl2a(sinusoid_level) * puretone_randomphase(t, sinusoid_frequency)
    if gap_length > 0:
        gap_center = signal_start + 200e-3
        gamma_t = [gap_center - gap_length / 2, gap_center + gap_length / 2]
        signal *= ramp_onoff(t,
                             gamma_t=gamma_t,
                             width=1e-3,
                             ramp_function=cosine_squared_ramp_func,
                             invert=True)

    stimulus_ramp = ramp_onoff(t,
                               gamma_t=[signal_start, signal_start + signal_length],
                               width=50e-3,  # 40 ms ramp
                               ramp_function=cosine_squared_ramp_func)

    noise = white_noise(t, spl2a(sinusoid_level - SIGNAL_TO_NOISE_RATIO))
    noise = butterworth_lowpass(noise, 3000, fs, order=4)  # 4th order butterworth filter translates to 24 dB/octave

    return stimulus_ramp * (signal + noise)
