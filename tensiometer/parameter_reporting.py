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


def get_mode1d(chain, param_names):
    """
    """
    # initial check of parameter names:
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = gtens._check_param_names(chain, param_names)
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
    """
    # initial check of parameter names:
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = gtens._check_param_names(chain, param_names)
    # get indexes of parameters:
    _indexes = [chain.index[p] for p in param_names]
    #
    return chain.getMeans(_indexes)






pass
