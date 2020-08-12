"""
This file contains some utilities to report parameter results in a nice way.
"""

"""
For test purposes:

from getdist import loadMCSamples, MCSamples, WeightedSamples
chain = loadMCSamples('./test_chains/DES')
chains = chain
param_names = None
import tensiometer.utilities as utils
import matplotlib.pyplot as plt
import tensiometer.gaussian_tension as gtens
import tensiometer.tensor_eigenvalues as teig
"""

###############################################################################
# initial imports:

import copy
import numpy as np
from getdist import MCSamples

from . import gaussian_tension as gtens

###############################################################################
# utility functions to get mean and 1d mode:


def get_mode1d(chain, param_names, settings=None):
    """
    Utility to compute the peak of the 1d posterior distribution for all
    parameters (parameter 1d mode).
    This depends and relies on the precomputed KDE smoothing so one can
    feed different analysis settings to change that.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: optional choice of parameter names to
        restrict the calculation. Default is all parameters.
    :param settings: optional dictionary with GetDist analysis settings
        to use for the 1d mode calculation.
        This will change the analysis settings of the chain globally.
    :return: an array with the 1d mode.
    """
    # initial check of parameter names:
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = gtens._check_param_names(chain, param_names)
    # KDE settings:
    if settings is not None:
        chain.updateSettings(settings=settings, doUpdate=True)
    # get the maximum probability from the precomputed pdf:
    param_mode = []
    for p in param_names:
        param_mode.append(chain.get1DDensity(p).x[
                          np.argmax(chain.get1DDensity(p).P)])
    param_mode = np.array(param_mode)
    #
    return param_mode


def get_mean(chain, param_names):
    """
    Utility to compute the parameter mean.
    Mostly this is an utility to get the mean from parameter names rather than
    parameter indexes as we would do in GetDist.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: optional choice of parameter names to
        restrict the calculation. Default is all parameters.
    :return: an array with the mean.
    """
    # initial check of parameter names:
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = gtens._check_param_names(chain, param_names)
    # get indexes of parameters:
    _indexes = [chain.index[p] for p in param_names]
    #
    return chain.getMeans(_indexes)

###############################################################################
#





pass
