"""
Generate stimuli similar to Experiment I in Shailer And Mooore, 1983
U{DOI: ???/??? <FIXME>}
"""
import numpy as np

SIGNAL_FREQUENCIES = 1e3 * np.array([
    0.4,
    0.8,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.5,
    8.0,
])
"""
Center frequencies used in the experiment
"""

SPECTRAL_WIDTH = 0.5
"""
Width of the signal is defined as::
    0.5 * center-frequency
"""

SIGNAL_TO_NOISE_RATIO = 5
"""
Level difference between signal and noise, in dB spectrum level
"""

NOISE_LEVEL = 20
"""
Spectrum level in dB RMS of the masker
"""

def generate_stimulus(t: np.ndarray,
                      ) -> np.ndarray:
    pass