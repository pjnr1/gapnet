import numpy as np

from data_creation.calibration.constants import SPL_REFERENCE_PRESSURE
from data_creation.calibration.hearinglevel import get_hearinglevel_interp


def spl2a(spl):
    """
    Convert SPL dB to amplitude

    @param spl:
    @return:
    """
    return SPL_REFERENCE_PRESSURE * np.power(10, spl / 20)


def a2spl(a):
    """
    Convert amplitude to SPL dB

    @param a:
    @return:
    """
    return 20 * np.log10(a / SPL_REFERENCE_PRESSURE)


def hl2spl(frequencies, levels, hl_ref=None):
    """
    Convert dB hearing level (HL) to dB sound pressure level (SPL)

    @param frequencies:
    @param levels:
    @param hl_ref:

    @return:
    """
    return levels + get_hearinglevel_interp(frequencies, hl_ref)


def spl2hl(frequencies, levels, hl_ref=None):
    """
    Convert dB sound pressure level (SPL) to dB hearing level (HL)

    @param frequencies:
    @param levels:
    @param hl_ref:

    @return:
    """
    return levels - get_hearinglevel_interp(frequencies, hl_ref)
