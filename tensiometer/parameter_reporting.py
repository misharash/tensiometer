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
import getdist.types as types
from getdist.mcsamples import MCSamplesError
import scipy.interpolate as interpol

from . import gaussian_tension as gtens

###############################################################################
# utility functions to get mean and 1d mode:


def get_mode1d(chain, param_names):
    """
    Utility to compute the peak of the 1d posterior distribution for all
    parameters (parameter 1d mode).
    This depends and relies on the precomputed KDE smoothing so one can
    feed different analysis settings to change that.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: optional choice of parameter names to
        restrict the calculation. Default is all parameters.
    :return: an array with the 1d mode.
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


def get_PJHPD_bounds(chain, param_names, levels):
    """
    Compute some estimate of the global ML confidence interval as described in
    https://arxiv.org/pdf/2007.01844.pdf

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: optional choice of parameter names to
        restrict the calculation. Default is all parameters.
    :param levels: array with confidence levels to compute, i.e. [0.68, 0.95]
    :return: an array with the bounds for each parameter.
    """
    # initial check of parameter names:
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = gtens._check_param_names(chain, param_names)
    # digest levels:
    lev = np.atleast_1d(levels)
    # sort the samples:
    sort_idx = np.argsort(chain.loglikes)
    # cycle over parameters:
    results = []
    for p in param_names:
        # get the parameter index:
        idx = chain.index[p]
        # get the density spline:
        density = chain.get1DDensity(p)
        # normalize:
        param_array = chain.samples[sort_idx, idx]
        norm_factor = interpol.splint(np.amin(param_array),
                                      np.amax(param_array),
                                      density.spl)
        # precompute the integrals:
        lmax = np.amax(lev)
        vmin, vmax, int = [param_array[0]], [param_array[0]], [0]
        for par in param_array:
            if par < vmin[-1] or par > vmax[-1]:
                if par < vmin[-1]:
                    vmin.append(par)
                    vmax.append(vmax[-1])
                if par > vmax[-1]:
                    vmax.append(par)
                    vmin.append(vmin[-1])
                int.append(interpol.splint(vmin[-1],
                                           vmax[-1],
                                           density.spl)/norm_factor)
                if int[-1] > lmax:
                    break
        # evaluate the interpolation of the bounds:
        results.append(np.stack((np.interp(lev, int, vmin),
                                 np.interp(lev, int, vmax))).T)
    #
    return results

###############################################################################
# Parameter table function:


def parameter_table(chain, param_names, use_peak=False, use_best_fit=True,
                    use_PJHPD_bounds=False, ncol=1, **kwargs):
    """
    Generate latex parameter table with summary results.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: optional choice of parameter names to
        restrict the calculation. Default is all parameters.
    :param use_peak: whether to use the peak of the 1d distribution instead of
        the mean. Default is False.
    :param use_best_fit: whether to include the best fit either from explicit
        minimization or sample. Default True.
    :param use_PJHPD_bounds: whether to report PJHPD bounds. Default False.
    :param ncol: number of columns for the table. Default 1.
    :param analysis_settings: optional analysis settings to use.
    :param kwargs: arguments for :class:`~getdist.types.ResultTable`
    :return: a :class:`~getdist.types.ResultTable`
    """
    # initial check of parameter names:
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = gtens._check_param_names(chain, param_names)
    # get parameter indexes:
    param_index = [chain.index[p] for p in param_names]
    # get marge stats:
    marge = copy.deepcopy(chain.getMargeStats())
    # if we want to use the peak substitute the mean in marge:
    if use_peak:
        mode = get_mode1d(chain, param_names)
        for num, ind in enumerate(param_index):
            marge.names[ind].mean = mode[num]
    # get the best fit:
    if use_best_fit:
        # get the best fit object from file or sample:
        try:
            bestfit = chain.getBestFit()
        except MCSamplesError:
            # have to get best fit from samples:
            bfidx = np.argmin(chain.loglikes)
            bestfit = types.BestFit()
            bestfit.names = copy.deepcopy(chain.paramNames.names)
            for nam in bestfit.names:
                nam.best_fit = nam.bestfit_sample
            bestfit.logLike = chain.loglikes[bfidx]
        # if we want PJHPD bounds we have to compute and add them:
        if use_PJHPD_bounds:
            # compute bounds (note have to do for all parameters):
            temp = get_PJHPD_bounds(chain,
                                    param_names=None,
                                    levels=marge.limits)
            # initialize another marge object:
            marge2 = copy.deepcopy(marge)
            for nam in marge2.names:
                # center:
                temp_bf = bestfit.parWithName(nam.name)
                nam.mean = temp_bf.best_fit
                # PJHPD bouds:
                temp_bounds = temp[chain.index[nam.name]]
                for lim, tmp in zip(nam.limits, temp_bounds):
                    lim.twotail = True
                    lim.onetail_lower = False
                    lim.onetail_upper = False
                    lim.lower = tmp[0]
                    lim.upper = tmp[1]
            # prepare the table:
            table = types.ResultTable(ncol, [marge, marge2],
                                      paramList=param_names, **kwargs)
        else:
            marge.addBestFit(bestfit)
            table = types.ResultTable(ncol, [marge],
                                      paramList=param_names, **kwargs)
    else:
        table = types.ResultTable(ncol, [marge],
                                  paramList=param_names, **kwargs)
    #
    return table
