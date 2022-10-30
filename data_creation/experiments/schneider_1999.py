"""
Gap detection experiment by Schneider, 1999
U{DOI: 10.1121/1.427062 <https://asa.scitation.org/doi/abs/10.1121/1.427062>}

"""
import numpy as np

from data_creation.calibration.transform import spl2a
from data_creation.ramp.gaussian_gate import generate_gaussian_gate_length
from data_creation.stimulus.noise import white_noise

SIGNAL_FREQUENCY = 2e3
"""
Frequency for the sinusoid used in the paper
"""

SIGNAL_LEVEL = 90
"""
90 dB SPL
"""

MARKER_DURATIONS = np.array([0.83,
                             2.5,
                             5.0,
                             40.,
                             200.,
                             500.]) * 1e-3
"""
The durations of the marker (in seconds) used in the paper
"""


def generate_stimulus(t: np.ndarray,
                      sinusoid_frequency: float = SIGNAL_FREQUENCY,
                      stimulus_start: float = 200e-3,
                      marker_duration: float = 2e-3,
                      gap_length: float = 1e-3,
                      with_gap: bool = True):
    """

    @param t:
        time-vector
    @param sinusoid_frequency:
        frequency of the target signal (see L{SIGNAL_FREQUENCY} for frequency used in the paper)
    @param stimulus_start:
        when to start the experiment
    @param marker_duration:
        duration of preceding marker
    @param gap_length:
        duration of the gap
    @param with_gap:
        produce the stimulus with or without a gap
    @return:
    """
    sigma = 0.0005
    sinusoid = np.cos(2 * np.pi * sinusoid_frequency * (t + stimulus_start))
    sinusoid *= spl2a(SIGNAL_LEVEL)
    if with_gap:
        mask = generate_gaussian_gate_length(t=t,
                                             gamma_t=stimulus_start,
                                             length=marker_duration,
                                             sigma=sigma)
        mask += generate_gaussian_gate_length(t=t,
                                              gamma_t=stimulus_start + marker_duration + gap_length + sigma,
                                              length=marker_duration,
                                              sigma=sigma)
    else:
        mask = generate_gaussian_gate_length(t=t,
                                             gamma_t=stimulus_start,
                                             length=2 * marker_duration + sigma,
                                             sigma=sigma)

    # TODO masking noise?

    return sinusoid * mask, mask
