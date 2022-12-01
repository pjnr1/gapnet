"""

"""


def generate_stimulus(t,
                      noise_bandwidth=None,
                      gap_start=None,
                      gap_duration=None,
                      ramp_duration=2.5e-3,
                      noise_start=0.1,
                      noise_duration=0.5,
                      fs=None,
                      ramp_function: RampFunction = linear_bandpass_filter,
                      noise_function: WhiteNoiseFunction = uniform_white_noise_generator):
    pass