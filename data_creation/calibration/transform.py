import numpy as np

import constants
from hearinglevel import get_hearinglevel_interp


def spl2a(spl):
    """
    Convert SPL dB to amplitude

    @arg spl:
    @return:
    """
    return constants.SPL_REFERENCE_PRESSURE * np.power(10, spl / 20)


def a2spl(a):
    """
    Convert amplitude to SPL dB

    @arg a:
    @return:
    """
    return 20 * np.log10(a / constants.SPL_REFERENCE_PRESSURE)


def hl2spl(frequencies, levels, hl_ref=None):
    """
    Convert dB hearing level (HL) to dB sound pressure level (SPL)

    @arg frequencies:
    @arg levels:
    @arg hl_ref:

    @return:
    """
    return levels + get_hearinglevel_interp(frequencies, hl_ref)


def spl2hl(frequencies, levels, hl_ref=None):
    """
    Convert dB sound pressure level (SPL) to dB hearing level (HL)

    @arg frequencies:
    @arg levels:
    @arg hl_ref:

    @return:
    """
    return levels - get_hearinglevel_interp(frequencies, hl_ref)
