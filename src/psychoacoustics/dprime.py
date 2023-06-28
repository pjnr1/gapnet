"""
TODO write brief introduction to d-prime measure
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import erfinv

from typing import Annotated

from typing_tools.annotation_checkers import ValueRange
from typing_tools.annotations import check_annotations


def norminv(p, mu=0., std=1.0):
    """
    Replica of the Matlab function C{norminv} (Normal inverse cumulative distribution function)

    @param p:
        Probability values at which to evaluate the inverse of the cdf (icdf), specified as a scalar value or an array
        of scalar values, where each element is in the range [0,1].
    @param mu:
        Mean of the normal distribution, specified as a scalar value or an array of scalar values.
    @param std:
        Standard deviation of the normal distribution, specified as a positive scalar value or an array of positive
        scalar values.
    @return:
        returns the inverse of the normal cdf with mean mu and standard deviation sigma, evaluated at the probability
        values in p.
    """
    return norm.ppf(p, loc=mu, scale=std)


@check_annotations
def inverse_cumulative_std_normfunc(p: Annotated[float, ValueRange(0., 1., False, False)]):
    """
    The inverse cumulative standard norm-function.
    This yields the same result as C{norminv}, i.e. C{norminv(x, mu=0., std=1.) == inverse_cumulative_std_normfunc(x)}.

    Equation originates from U{Jones 2016 <http://dx.doi.org/10.1016/j.jmp.2016.04.006>}

    @param p:
        probability values (must be in the range [0, 1])
    @return:
    """
    return np.sqrt(2) * erfinv(2 * p - 1)


def dprime_empirical(hit_rate, false_alarm_rate):
    """
    Compute dprime from hit rate and false-alarm rate

    @param hit_rate:
        Number of "yes" responses given a stimulus with the signal, divided by the number of presentations with signal
    @param false_alarm_rate:
        Number of "yes" responses given a stimulus with the signal, divided by the number of presentations without
        signal
    @return:
    """
    return inverse_cumulative_std_normfunc(hit_rate) - inverse_cumulative_std_normfunc(false_alarm_rate)


def ideal_threshold(signal, noise):
    """
    Computes the ideal threshold between the two distributions, signal and noise.

    We use E(x) as the expected value of x, and estimate this by the mean of the samples provided.

    When the E(noise) samples are larger than the E(signal),
    the ideal threshold returned is E(noise).

    Otherwise, (E(signal) + E(noise)) / 2 is returned, i.e. the mean of the means.

    @param signal:
        responses due to stimuli with signal
    @param noise:
        responses due to stimuli without signal, i.e. noise only
    @return:
        the ideal threshold between the two distributions
    """
    def _mean_function__pandas(x):
        return np.mean(x.to_numpy())

    if isinstance(signal, pd.DataFrame) and isinstance(noise, pd.DataFrame):
        mf = _mean_function__pandas
    else:
        mf = np.mean

    means = [mf(noise),
             mf(signal)]
    # If the mean of the noise distribution is equal or larger, select as ideal threshold
    if means[0] >= means[1]:
        return means[0]
    else:
        # return mean of the two means
        return sum(means) / 2


def dprime_empirical_jones(h, f, threshold=0.5, eps=0.5):
    """
    Algorithm copied from Pete R Jones (supplementary material to U{Jones 2016 <http://dx.doi.org/10.1016/j.jmp.2016.04.006>}

    @param h:
        internal variable responses due to stimuli with signal
    @param f:
        internal variable responses due to stimuli without signal
    @param threshold:
        threshold for determining whether response was a "yes" or "no"
    @param eps:
        correction factor for avoiding infinite results
    @return:
    """
    def _correct_for_inf(x, n):
        """
        Small helper to correct the rate prior to applying C{norminv}

        @param x:
        @param n:
        @return:
        """
        if x == 0: return x + eps
        if x == n: return x - eps
        return x

    n_hits = sum(h >= threshold)
    n_falarms = sum(f >= threshold)

    n_signal = len(h)
    n_noises = len(f)

    if n_signal == 0 or n_noises == 0:
        return -1

    n_hits = _correct_for_inf(n_hits, n_signal)
    n_falarms = _correct_for_inf(n_falarms, n_noises)

    return norminv(n_hits / n_signal) - norminv(n_falarms / n_noises)


def dprime_analytical(mu1, mu2, std1, std2):
    """
    Computes the d-prime of two internal response distributions, e.g. from signal present and signal absent.
    
    It assumes the simplest form of internal response, thus no correlation etc. (see U{Jones 2016 <http://dx.doi.org/10.1016/j.jmp.2016.04.006>} for a thorough 
    tutorial).
    
    @param mu1: 
        Mean of distribution 1
    @param mu2: 
        Mean of distribution 2
    @param std1: 
        Standard deviation of distribution 1
    @param std2: 
        Standard deviation of distribution 1
    @return: 
    """
    return (mu1 - mu2) / np.sqrt((1 / 2) * (std1 + std2))
