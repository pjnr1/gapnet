import timeit
from functools import partial

import numpy as np

from scipy.optimize import fsolve, curve_fit


# TODO consider rewriting as class


def psychometric_func(x, a, b, scale=1., loc=0):
    """
    Computes the value of the psychometric function at x with parameters a (position) and b (width) with scaling factor
    and location/offset

    @arg x:
        position to compute
    @arg a:
        position-parameter of the psychometric function
    @arg b:
        width-parameter of the psychometric function
    @arg scale:
        scale (default is 1)
    @arg loc:
        location (default is 0)
    @return:
        the value of the psychometric function at x
    """
    return loc + scale * (1. / (1. + np.exp(-(x - a) / b)))


def get_psychometric_point(p: float, a: float, b: float, x0: float = None, method: str = 'analytical') -> float:
    """
    Return the point p, e.g. 50%/0.5 point or similar for the psychometric function centered in a with width b.
    This can be computed either analytically or solved using scipy.optimize.fsolve.

    Scale and location are not used in this calculation

    @arg p:
        point to retrieve
    @arg a:
        position-parameter of the psychometric function
    @arg b:
        width-parameter of the psychometric function
    @arg x0:
        if using fsolve, x0 is used as starting point
    @arg method:
        method used for retrieving x
    @return:
        the position for which the psychometric function equals p
    """
    if x0 is None:
        x0 = a

    methods = {
        'solve': fsolve(lambda x: p - partial(psychometric_func, a=a, b=b, scale=1.0, loc=0.)(x),
                        x0=np.array(x0))[0],
        'analytical': a - b * np.log((1 / p) - 1)
    }
    return methods[method]


def fit_psychometric_func(xdata,
                          ydata,
                          with_scale=True,
                          with_loc=True,
                          scale_bounds=None,
                          loc_bounds=None,
                          scale_p0=1.,
                          loc_p0=1.):
    """
    Find the optimal fit of the psychometric function to the given data scipy.optimize.curve_fit

    @arg xdata:
    @arg ydata:
    @arg with_scale:
        enable fitting with a scaling parameter
    @arg with_loc:
        enable fitting with an offset parameter
    @arg scale_bounds:
        boundary conditions for the scaling parameter
    @arg loc_bounds:
        boundary conditions for the location parameter
    @arg scale_p0:
        initial point for the scaling parameter
    @arg loc_p0:
        initial point for the location parameter

    @return:
        List of optimal parameters for the psychometric-function on the input data
    """
    # Set defaults
    if scale_bounds is None:
        scale_bounds = [0., 1.]
    if loc_bounds is None:
        loc_bounds = [0., 1.]

    # boundaries for fitting : (lower, upper)
    bounds = ([0.,  # alpha
               0.],  # beta
              [100.,  # alpha
               20.])  # beta

    # initial position
    p0 = [10.,  # alpha
          1.]  # beta

    # Ensure right parameter is fitted (in case of with_loc=1, with_scale=0
    # if with_scale and with_loc:
    if with_scale and not with_loc:
        f = partial(psychometric_func, loc=0.)
        bounds[0].append(scale_bounds[0])
        bounds[1].append(scale_bounds[1])
        p0.append(scale_p0)
    elif not with_scale and not with_loc:
        f = partial(psychometric_func, scale=1.0, loc=0.)
    elif not with_scale and with_loc:
        f = partial(psychometric_func, scale=1.0)
        bounds[0].append(loc_bounds[0])
        bounds[1].append(loc_bounds[1])
        p0.append(loc_p0)
    else:
        f = psychometric_func
        bounds[0].append(scale_bounds[0])
        bounds[1].append(scale_bounds[1])
        p0.append(scale_p0)
        bounds[0].append(loc_bounds[0])
        bounds[1].append(loc_bounds[1])
        p0.append(loc_p0)

    try:
        p, _ = curve_fit(f=f,
                         xdata=xdata,
                         ydata=ydata,
                         bounds=bounds,
                         p0=p0)
        return p
    except RuntimeError:
        return None
