import numpy as np

from data_creation.stimulus.puretone import generate_gaussian_gate_length


def generate_stimulus(t,
                      f=2e3,
                      stimulus_start=0.01,
                      marker_duration=0.002,
                      gap_duration=0.001,
                      with_gap=True):
    sigma = 0.0005
    sinosoid = np.cos(2 * np.pi * f * (t + stimulus_start))
    if with_gap:
        mask = generate_gaussian_gate_length(t=t,
                                             gamma_t=stimulus_start,
                                             length=marker_duration,
                                             sigma=sigma)
        mask += generate_gaussian_gate_length(t=t,
                                              gamma_t=stimulus_start + marker_duration + gap_duration + sigma,
                                              length=marker_duration,
                                              sigma=sigma)
    else:
        mask = generate_gaussian_gate_length(t=t,
                                             gamma_t=stimulus_start,
                                             length=2 * marker_duration + gap_duration + sigma,
                                             sigma=sigma)
    return sinosoid * mask
