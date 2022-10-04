"""
Gap detection experiment by Zeng et al., 2005,
U{DOI: 10.1152/jn.00985.2004 <https://journals.physiology.org/doi/full/10.1152/jn.00985.2004>}

Stimulus used is broadband noise
"""
from ..ramp.ramp import ramp_onoff_with_gap
from ..ramp.ramp import RampFunction
from ..time.time import start_and_duration_to_gamma_t, get_sampling_frequency
from ..stimulus.noise import white_noise, WhiteNoiseFunction, uniform_white_noise
from ..filters.filters import linear_bandpass_filter


def generate_stimulus(t,
                      noise_bandwidth=None,
                      gap_start=None,
                      gap_duration=None,
                      ramp_duration=2.5e-3,
                      noise_start=0.1,
                      noise_duration=0.5,
                      fs=None,
                      ramp_function: RampFunction = linear_bandpass_filter,
                      noise_function: WhiteNoiseFunction = uniform_white_noise):
    if noise_bandwidth is None:
        noise_bandwidth = [20, 16e3]
    if gap_start is None and gap_duration is not None:
        gap_start = noise_start + (noise_duration / 2) - (gap_duration / 2)
    fs = get_sampling_frequency(fs, t)

    # Generate noise
    signal = white_noise(t=t, generator=noise_function)

    # Create stimulus ramps / gate
    signal *= ramp_onoff_with_gap(t,
                                  stim_gamma_t=start_and_duration_to_gamma_t(start=noise_start,
                                                                             duration=noise_duration),
                                  gap_gamma_t=start_and_duration_to_gamma_t(start=gap_start,
                                                                            duration=gap_duration),
                                  width=ramp_duration,
                                  ramp_function=ramp_function)

    return linear_bandpass_filter(signal=signal,
                                  low=noise_bandwidth[0],
                                  high=noise_bandwidth[1],
                                  fs=fs)
