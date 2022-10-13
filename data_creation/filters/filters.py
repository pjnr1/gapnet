import numpy as np
import scipy.signal as ss


def linear_bandpass_filter(signal, low, high, fs=44.1e3, filter_order=np.power(2, 11)):
    filter_coefficients = ss.firwin(filter_order, [low, high], pass_zero=False, fs=fs)
    return ss.filtfilt(filter_coefficients, [1], signal)


def butterworth_lowpass(signal, cutoff, fs, order=4):
    sos = ss.butter(N=order, Wn=cutoff, fs=fs, output='sos')
    return ss.sosfiltfilt(sos, signal)
