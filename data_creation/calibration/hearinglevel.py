import numpy as np

from scipy.interpolate import interp1d


def get_hearinglevel():
    """
    TODO add reference for hearing levels

    @return:
    """
    std_frequencies = np.array([
        125,
        250,
        500,
        1000,
        2000,
        3000,
        4000,
        6000,
        8000
    ])

    std_levels = np.array([
        22.1,
        1.4,
        3.8,
        0.8,
        -1.5,
        -4,
        -3.8,
        1.4,
        6.8
    ])
    return std_frequencies, std_levels


def get_hearinglevel_interp(f, hl_ref=None):
    """

    @arg f: Frequencies to return for
    @arg hl_ref: Hearing level reference
    @return: Hearing level at frequencies f
    """
    if hl_ref is None:
        std_frequencies, std_levels = get_hearinglevel()
    else:
        std_frequencies, std_levels = hl_ref
    return interp1d(x=np.log10(std_frequencies),
                    y=std_levels,
                    fill_value='extrapolate')(np.log10(f))
