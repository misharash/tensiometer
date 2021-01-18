"""
This file contains the functions and utilities to compute agreement and
disagreement between two different chains with Monte Carlo methods.

For more details on the method implemented see
`arxiv 1806.04649 <https://arxiv.org/pdf/1806.04649.pdf>`_
and `arxiv 1912.04880 <https://arxiv.org/pdf/1912.04880.pdf>`_.


For test purposes:

from getdist import loadMCSamples, MCSamples, WeightedSamples
chain_1 = loadMCSamples('./tensiometer/test_chains/DES')
chain_2 = loadMCSamples('./tensiometer/test_chains/Planck18TTTEEE')
chain_12 = loadMCSamples('./tensiometer/test_chains/Planck18TTTEEE_DES')
chain_prior = loadMCSamples('./tensiometer/test_chains/prior')

import tensiometer.utilities as utils
import matplotlib.pyplot as plt

diff_chain = parameter_diff_chain(chain_1, chain_2, boost=1)
num_params, num_samples = diff_chain.samples.T.shape

param_names = None
scale = None
method = 'brute_force'
feedback=2
n_threads = 1
"""

###############################################################################
# initial imports and set-up:

import os
import time
import gc
from numba import jit
import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples, WeightedSamples
import scipy
from scipy.linalg import sqrtm
from scipy.integrate import simps
from scipy.spatial import cKDTree

from . import utilities as utils

# imports for parallel calculations:
import multiprocessing
import joblib
# number of threads available:
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = multiprocessing.cpu_count()

###############################################################################
# Parameter difference chain:


def parameter_diff_weighted_samples(samples_1, samples_2, boost=1,
                                    indexes_1=None, indexes_2=None):
    """
    Compute the parameter differences of two input weighted samples.
    The parameters of the difference samples are related to the
    parameters of the input samples, :math:`\\theta_1` and
    :math:`\\theta_2` by:

    .. math:: \\Delta \\theta \\equiv \\theta_1 - \\theta_2

    This function does not assume Gaussianity of the chain.
    This functions does assume that the parameter determinations from the two
    chains (i.e. the underlying data sets) are uncorrelated.
    Do not use this function for chains that are correlated.

    :param samples_1: :class:`~getdist.chains.WeightedSamples`
        first input weighted samples with :math:`n_1` samples.
    :param samples_2: :class:`~getdist.chains.WeightedSamples`
        second input weighted samples with :math:`n_2` samples.
    :param boost: (optional) boost the number of samples in the
        difference. By default the length of the difference samples
        will be the length of the longest one.
        Given two samples the full difference samples can contain
        :math:`n_1\\times n_2` samples but this is usually prohibitive
        for realistic chains.
        The boost parameters wil increase the number of samples to be
        :math:`{\\rm boost}\\times {\\rm max}(n_1,n_2)`.
        Default boost parameter is one.
        If boost is None the full difference chain is going to be computed
        (and will likely require a lot of memory and time).
    :param indexes_1: (optional) array with the indexes of the parameters to
        use for the first samples. By default this tries to use all
        parameters.
    :param indexes_2: (optional) array with the indexes of the parameters to
        use for the second samples. By default this tries to use all
        parameters.
    :return: :class:`~getdist.chains.WeightedSamples` the instance with the
        parameter difference samples.
    """
    # test for type, this function assumes that we are working with MCSamples:
    if not isinstance(samples_1, WeightedSamples):
        raise TypeError('Input samples_1 is not of WeightedSamples type.')
    if not isinstance(samples_2, WeightedSamples):
        raise TypeError('Input samples_2 is not of WeightedSamples type.')
    # get indexes:
    if indexes_1 is None:
        indexes_1 = np.arange(samples_1.samples.shape[1])
    if indexes_2 is None:
        indexes_2 = np.arange(samples_2.samples.shape[1])
    # check:
    if not len(indexes_1) == len(indexes_2):
        raise ValueError('The samples do not containt the same number',
                         'of parameters.')
    num_params = len(indexes_1)
    # order the chains so that the second chain is always with less points:
    if (len(samples_1.weights) >= len(samples_2.weights)):
        ch1, ch2 = samples_1, samples_2
        sign = +1.
        ind1, ind2 = indexes_1, indexes_2
    else:
        ch1, ch2 = samples_2, samples_1
        sign = -1.
        ind1, ind2 = indexes_2, indexes_1
    # get number of samples:
    num_samps_1 = len(ch1.weights)
    num_samps_2 = len(ch2.weights)
    if boost is None:
        sample_boost = num_samps_2
    else:
        sample_boost = min(boost, num_samps_2)
    # create the arrays (these might be big depending on boost level...):
    weights = np.empty((num_samps_1*sample_boost))
    difference_samples = np.empty((num_samps_1*sample_boost, num_params))
    if ch1.loglikes is not None and ch2.loglikes is not None:
        loglikes = np.empty((num_samps_1*sample_boost))
    else:
        loglikes = None
    # compute the samples:
    for ind in range(sample_boost):
        base_ind = int(float(ind)/float(sample_boost)*num_samps_2)
        _indexes = range(base_ind, base_ind+num_samps_1)
        # compute weights (as the product of the weights):
        weights[ind*num_samps_1:(ind+1)*num_samps_1] = \
            ch1.weights*np.take(ch2.weights, _indexes, mode='wrap')
        # compute the likelihood:
        if ch1.loglikes is not None and ch2.loglikes is not None:
            loglikes[ind*num_samps_1:(ind+1)*num_samps_1] = \
                ch1.loglikes+np.take(ch2.loglikes, _indexes, mode='wrap')
        # compute the difference samples:
        difference_samples[ind*num_samps_1:(ind+1)*num_samps_1, :] = \
            ch1.samples[:, ind1] \
            - np.take(ch2.samples[:, ind2], _indexes, axis=0, mode='wrap')
    # get additional informations:
    if samples_1.name_tag is not None and samples_2.name_tag is not None:
        name_tag = samples_1.name_tag+'_diff_'+samples_2.name_tag
    else:
        name_tag = None
    if samples_1.label is not None and samples_2.label is not None:
        label = samples_1.label+' diff '+samples_2.label
    else:
        label = None
    if samples_1.min_weight_ratio is not None and \
       samples_2.min_weight_ratio is not None:
        min_weight_ratio = min(samples_1.min_weight_ratio,
                               samples_2.min_weight_ratio)
    # initialize the weighted samples:
    diff_samples = WeightedSamples(ignore_rows=0,
                                   samples=sign*difference_samples,
                                   weights=weights, loglikes=loglikes,
                                   name_tag=name_tag, label=label,
                                   min_weight_ratio=min_weight_ratio)
    #
    return diff_samples

###############################################################################


def parameter_diff_chain(chain_1, chain_2, boost=1):
    """
    Compute the chain of the parameter differences between the two input
    chains. The parameters of the difference chain are related to the
    parameters of the input chains, :math:`\\theta_1` and :math:`\\theta_2` by:

    .. math:: \\Delta \\theta \\equiv \\theta_1 - \\theta_2

    This function only returns the differences for the parameters that are
    common to both chains.
    This function preserves the chain separation (if any) so that the
    convergence of the difference chain can be tested.
    This function does not assume Gaussianity of the chain.
    This functions does assume that the parameter determinations from the two
    chains (i.e. the underlying data sets) are uncorrelated.
    Do not use this function for chains that are correlated.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        first input chain with :math:`n_1` samples
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        second input chain with :math:`n_2` samples
    :param boost: (optional) boost the number of samples in the
        difference chain. By default the length of the difference chain
        will be the length of the longest chain.
        Given two chains the full difference chain can contain
        :math:`n_1\\times n_2` samples but this is usually prohibitive
        for realistic chains.
        The boost parameters wil increase the number of samples to be
        :math:`{\\rm boost}\\times {\\rm max}(n_1,n_2)`.
        Default boost parameter is one.
        If boost is None the full difference chain is going to be computed
        (and will likely require a lot of memory and time).
    :return: :class:`~getdist.mcsamples.MCSamples` the instance with the
        parameter difference chain.
    """
    # check input:
    if boost is not None:
        if boost < 1:
            raise ValueError('Minimum boost is 1\n Input value is ', boost)
    # test for type, this function assumes that we are working with MCSamples:
    if not isinstance(chain_1, MCSamples):
        raise TypeError('Input chain_1 is not of MCSamples type.')
    if not isinstance(chain_2, MCSamples):
        raise TypeError('Input chain_2 is not of MCSamples type.')
    # get the parameter names:
    param_names_1 = chain_1.getParamNames().list()
    param_names_2 = chain_2.getParamNames().list()
    # get the common names:
    param_names = [_p for _p in param_names_1 if _p in param_names_2]
    num_params = len(param_names)
    if num_params == 0:
        raise ValueError('There are no shared parameters to difference')
    # get the names and labels:
    diff_param_names = ['delta_'+name for name in param_names]
    diff_param_labels = ['\\Delta '+name.label for name in
                         chain_1.getParamNames().parsWithNames(param_names)]
    # get parameter indexes:
    indexes_1 = [chain_1.index[name] for name in param_names]
    indexes_2 = [chain_2.index[name] for name in param_names]
    # get separate chains:
    if not hasattr(chain_1, 'chain_offsets'):
        _chains_1 = [chain_1]
    else:
        _chains_1 = chain_1.getSeparateChains()
    if not hasattr(chain_2, 'chain_offsets'):
        _chains_2 = [chain_2]
    else:
        _chains_2 = chain_2.getSeparateChains()
    # set the boost:
    if chain_1.sampler == 'nested' \
       or chain_2.sampler == 'nested' or boost is None:
        chain_boost = max(len(_chains_1), len(_chains_2))
        sample_boost = None
    else:
        chain_boost = min(boost, max(len(_chains_1), len(_chains_2)))
        sample_boost = boost
    # get the combinations:
    if len(_chains_1) > len(_chains_2):
        temp_ind = np.indices((len(_chains_2), len(_chains_1)))
    else:
        temp_ind = np.indices((len(_chains_1), len(_chains_2)))
    ind1 = np.concatenate([np.diagonal(temp_ind, offset=i, axis1=1, axis2=2)[0]
                           for i in range(chain_boost)])
    ind2 = np.concatenate([np.diagonal(temp_ind, offset=i, axis1=1, axis2=2)[1]
                           for i in range(chain_boost)])
    chains_combinations = [[_chains_1[i], _chains_2[j]]
                           for i, j in zip(ind1, ind2)]
    # compute the parameter difference samples:
    diff_chain_samples = [parameter_diff_weighted_samples(samp1,
                          samp2, boost=sample_boost, indexes_1=indexes_1,
                          indexes_2=indexes_2) for samp1, samp2
                          in chains_combinations]
    # create the samples:
    diff_samples = MCSamples(names=diff_param_names, labels=diff_param_labels)
    diff_samples.chains = diff_chain_samples
    diff_samples.makeSingle()
    # get the ranges:
    _ranges = {}
    for name, _min, _max in zip(diff_param_names,
                                np.amin(diff_samples.samples, axis=0),
                                np.amax(diff_samples.samples, axis=0)):
        _ranges[name] = [_min, _max]
    diff_samples.setRanges(_ranges)
    # initialize other things:
    if chain_1.name_tag is not None and chain_2.name_tag is not None:
        diff_samples.name_tag = chain_1.name_tag+'_diff_'+chain_2.name_tag
    # set distinction between base and derived parameters:
    _temp = diff_samples.getParamNames().list()
    _temp_paramnames = chain_1.getParamNames()
    for _nam in diff_samples.getParamNames().parsWithNames(_temp):
        _temp_name = _nam.name.replace('delta_', '')
        _nam.isDerived = _temp_paramnames.parWithName(_temp_name).isDerived
    # update and compute everything:
    diff_samples.updateBaseStatistics()
    diff_samples.deleteFixedParams()
    #
    return diff_samples

###############################################################################
# KDE bandwidth selection:


def _helper_whiten_samples(samples, weights):
    """
    """
    # compute sample covariance:
    _cov = np.cov(samples.T, aweights=weights)
    # compute its inverse square root:
    _temp = sqrtm(utils.QR_inverse(_cov))
    # whiten the samples:
    white_samples = samples.dot(_temp)
    #
    return white_samples


def Scotts_bandwidth(num_params, num_samples):
    """
    Compute Scott's rule of thumb bandwidth covariance scaling.
    This should be a fast approximation of the 1d MISE estimate.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: Scott's scaling.
    """
    return num_samples**(-2./(num_params+4.)) * np.identity(int(num_params))


def AMISE_bandwidth(num_params, num_samples):
    """
    Compute Silverman's rule of thumb bandwidth covariance scaling.
    This is the default scaling that is used to compute the KDE estimate of
    parameter shifts.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: Silverman's scaling.
    """
    coeff = (num_samples * (num_params + 2.) / 4.)**(-2. / (num_params + 4.))
    return coeff * np.identity(int(num_params))


def MAX_bandwidth(num_params, num_samples):
    """
    """
    d, n = num_params, num_samples
    coeff = (d + 8.)**((d + 6.) / (d + 4.)) / 4.
    coeff = coeff*(1./n/(d + 2.)/scipy.special.gamma(d/2. + 4))**(2./(d + 4.))
    return coeff*np.identity(int(num_params))


@jit(nopython=True, fastmath=True)
def _mise1d_optimizer(alpha, d, n):
    """
    """
    tmp = 2**(-d/2.) - 2/(2 + alpha)**(d/2.) + (2 + 2*alpha)**(-d/2.) \
        + (alpha**(-d/2.) - (1 + alpha)**(-d/2.))/(2**(d/2.)*n)
    return tmp


@jit(nopython=True, fastmath=True)
def _mise1d_optimizer_jac(alpha, d, n):
    """
    """
    tmp = d*(2 + alpha)**(-1 - d/2.) - d*(2 + 2*alpha)**(-1 - d/2.) \
        + (2**(-1 - d/2.)*d*(-alpha**(-1 - d/2.)
                             + (1 + alpha)**(-1 - d/2.)))/n
    return tmp


def MISE_bandwidth_1d(num_params, num_samples, **kwargs):
    """
    """
    # initial calculations:
    alpha0 = kwargs.pop('alpha0', None)
    if alpha0 is None:
        alpha0 = AMISE_bandwidth(num_params, num_samples)[0, 0]
    d, n = num_params, num_samples
    # explicit optimization:
    opt = scipy.optimize.minimize(lambda alpha:
                                  _mise1d_optimizer(np.exp(alpha), d, n),
                                  np.log(alpha0),
                                  jac=lambda alpha:
                                  _mise1d_optimizer_jac(np.exp(alpha), d, n),
                                  **kwargs)
    # check for success:
    if not opt.success:
        print(opt)
    #
    return np.exp(opt.x[0]) * np.identity(num_params)


@jit(nopython=True, fastmath=True)
def _mise_optimizer(H, d, n):
    Id = np.identity(d)
    tmp = 1./np.sqrt(np.linalg.det(2.*H))/n
    tmp = tmp + (1.-1./n)/np.sqrt(np.linalg.det(2.*H + 2.*Id)) \
        - 2./np.sqrt(np.linalg.det(H + 2.*Id)) + np.power(2., -d/2.)
    return tmp


def MISE_bandwidth(num_params, num_samples, feedback=0, **kwargs):
    """
    """
    # initial calculations:
    alpha0 = kwargs.pop('alpha0', None)
    if alpha0 is None:
        alpha0 = MISE_bandwidth_1d(num_params, num_samples)
    alpha0 = utils.PDM_to_vector(alpha0)
    d, n = num_params, num_samples
    # build a constraint:
    bounds = kwargs.pop('bounds', None)
    if bounds is None:
        bounds = np.array([[None, None] for i in range(d*(d+1)//2)])
        bounds[np.tril_indices(d, 0)[0] == np.tril_indices(d, 0)[1]] = [alpha0[0]/100, alpha0[0]*100]
    # explicit optimization:
    opt = scipy.optimize.minimize(lambda x: _mise_optimizer(utils.vector_to_PDM(x), d, n),
                                  x0=alpha0, bounds=bounds, **kwargs)
    # check for success:
    if not opt.success or feedback > 2:
        print('MISE_bandwidth')
        print(opt)
    #
    return utils.vector_to_PDM(opt.x)


@jit(nopython=True, fastmath=True, parallel=True)
def _UCV_optimizer_brute_force(H, weights, white_samples):
    """
    Note this solves for sqrt(H)
    """
    # digest:
    n, d = white_samples.shape
    fac = 2**(-d/2.)
    # compute the weights vectors:
    wtot = np.sum(weights)
    neff = wtot**2 / np.sum(weights**2)
    alpha = wtot / (wtot - weights)
    # compute determinant:
    detH = np.linalg.det(H)
    # whiten samples with inverse H:
    samps = white_samples.dot(np.linalg.inv(H))
    # brute force summation:
    res = 0.
    for i in range(1, n):
        for j in range(i):
            temp_samp = samps[i]-samps[j]
            r2 = np.dot(temp_samp, temp_samp)
            temp = fac*np.exp(-0.25*r2) - 2.*alpha[i]*np.exp(-0.5*r2)
            res += weights[i]*weights[j]*temp
    res = 2. * res / wtot**2
    #
    return (fac/neff + res)/detH


def _UCV_optimizer_nearest(H, weights, white_samples, n_nearest=20):
    """
    Note this solves for sqrt(H) and uses a truncated KD-tree
    """
    # digest:
    n, d = white_samples.shape
    fac = 2**(-d/2.)
    # compute the weights vectors:
    wtot = np.sum(weights)
    neff = wtot**2 / np.sum(weights**2)
    alpha = wtot / (wtot - weights)
    # compute determinant:
    detH = np.linalg.det(H)
    # whiten samples with inverse H:
    samps = white_samples.dot(np.linalg.inv(H))
    # KD-tree computation:
    data_tree = cKDTree(samps, balanced_tree=True)
    # query for nearest neighbour:
    r2, idx = data_tree.query(samps, np.arange(2, n_nearest), n_jobs=-1)
    r2 = np.square(r2)
    temp = weights[:, None]*weights[idx]*(fac*np.exp(-0.25*r2)
                                          - 2.*np.exp(-0.5*r2)*alpha[:, None])
    res = np.sum(temp) / wtot**2
    #
    return (fac/neff + res)/detH


def UCV_bandwidth(weights, white_samples, alpha0=None, feedback=0, mode='full', **kwargs):
    """
    """
    # digest input:
    n, d = white_samples.shape
    n_nearest = kwargs.pop('n_nearest', 20)
    # get number of effective samples:
    wtot = np.sum(weights)
    neff = wtot**2 / np.sum(weights**2)
    # initial guess calculations:
    t0 = time.time()
    if alpha0 is None:
        alpha0 = AMISE_bandwidth(d, neff)
    # select mode:
    if mode == '1d':
        opt = scipy.optimize.minimize(lambda alpha: _UCV_optimizer_nearest(np.sqrt(np.exp(alpha)) * np.identity(d), weights, white_samples, n_nearest),
                                      np.log(alpha0[0, 0]), **kwargs)
        res = np.exp(opt.x[0]) * np.identity(d)
    elif mode == 'diag':
        opt = scipy.optimize.minimize(lambda alpha: _UCV_optimizer_nearest(np.diag(np.sqrt(np.exp(alpha))), weights, white_samples, n_nearest),
                                      x0=np.log(np.diag(alpha0)), **kwargs)
        res = np.diag(np.exp(opt.x))
    elif mode == 'full':
        # build a constraint:
        bounds = kwargs.pop('bounds', None)
        if bounds is None:
            bounds = np.array([[None, None] for i in range(d*(d+1)//2)])
            bounds[np.tril_indices(d, 0)[0] == np.tril_indices(d, 0)[1]] = [alpha0[0, 0]/10, alpha0[0, 0]*10]
        # explicit optimization:
        alpha0 = utils.PDM_to_vector(sqrtm(alpha0))
        opt = scipy.optimize.minimize(lambda alpha: _UCV_optimizer_nearest(utils.vector_to_PDM(alpha), weights, white_samples, n_nearest),
                                      x0=alpha0, bounds=bounds, **kwargs)
        res = utils.vector_to_PDM(opt.x)
        res = np.dot(res, res)
    # check for success and final feedback:
    if not opt.success or feedback > 2:
        print(opt)
    if feedback > 0:
        t1 = time.time()
        print('Time taken for UCV_bandwidth '+mode+' calculation:',
              round(t1-t0, 1), '(s)')
    #
    return res


@jit(nopython=True, fastmath=True)
def _MSE_CV_optimizer(H, weights, white_samples, K0):
    """
    Note this solves for sqrt(H)
    """
    # digest:
    n, d = white_samples.shape
    fac = (2*np.pi)**(-d/2.)
    # compute the weights vectors:
    wtot = np.sum(weights)
    # compute determinant:
    detH = np.linalg.det(H)
    # whiten samples with inverse H:
    X = white_samples.dot(np.linalg.inv(H))
    # compute kernel:
    K = fac/detH*np.exp(-0.5*(X*X).sum(axis=1))
    temp = np.dot(weights, K)
    temp = ((wtot - weights)*K0 - (temp - weights*K))**2
    fac = (weights / (wtot - weights))**2
    #
    return np.dot(fac, temp)


def MSE_CV_bandwidth(weights, white_samples, alpha0=None, feedback=0, mode='full', **kwargs):
    """
    """
    # digest input:
    n, d = white_samples.shape
    # get number of effective samples:
    wtot = np.sum(weights)
    neff = wtot**2 / np.sum(weights**2)
    # initial guess calculations:
    t0 = time.time()
    if alpha0 is None:
        alpha0 = AMISE_bandwidth(d, neff)
    # initial calculation of K0:
    H0 = sqrtm(alpha0)
    fac = (2*np.pi)**(-d/2.)
    detH0 = np.linalg.det(H0)
    X = white_samples.dot(np.linalg.inv(H0))
    K0 = fac/detH0*np.exp(-0.5*(X*X).sum(axis=1))
    # select mode:
    if mode == '1d':
        opt = scipy.optimize.minimize(lambda alpha: _MSE_CV_optimizer(np.sqrt(np.exp(alpha)) * np.identity(d), weights, white_samples, K0),
                                      np.log(100*alpha0[0,0]), **kwargs)
        res = np.exp(opt.x[0]) * np.identity(d)
    elif mode == 'diag':
        opt = scipy.optimize.minimize(lambda alpha: _MSE_CV_optimizer(np.diag(np.sqrt(np.exp(alpha))), weights, white_samples, K0),
                                      x0=np.log(np.diag(100*alpha0)), **kwargs)
        res = np.diag(np.exp(opt.x))
    elif mode == 'full':
        # build a constraint:
        bounds = kwargs.pop('bounds', None)
        if bounds is None:
            bounds = np.array([[None, None] for i in range(d*(d+1)//2)])
            bounds[np.tril_indices(d, 0)[0] == np.tril_indices(d, 0)[1]] = [alpha0[0, 0]/100, alpha0[0, 0]*200]
        # explicit optimization:
        alpha0 = utils.PDM_to_vector(sqrtm(100*alpha0))
        opt = scipy.optimize.minimize(lambda alpha: _MSE_CV_optimizer(utils.vector_to_PDM(alpha), weights, white_samples, K0),
                                      x0=alpha0, bounds=bounds, **kwargs)
        res = utils.vector_to_PDM(opt.x)
        res = np.dot(res, res)
    # check for success:
    if not opt.success or feedback > 2:
        print('MSE_CV_bandwidth')
        print(opt)
    # some feedback:
    if feedback > 0:
        t1 = time.time()
        print('Time taken for MSE_CV_bandwidth calculation:',
              round(t1-t0, 1), '(s)')
    #
    return res

###############################################################################
# Parameter difference integrals:


def _gauss_kde_logpdf(x, samples, weights):
    """
    Utility function to compute the Gaussian log KDE probability at x from
    already whitened samples, possibly with weights.
    Normalization constants are ignored.
    """
    X = x-samples
    return np.log(np.dot(weights, np.exp(-0.5*(X*X).sum(axis=1))))


def _brute_force_parameter_shift(white_samples, weights, zero_prob,
                                 num_samples, feedback):
    """
    Brute force parallelized algorithm for parameter shift.
    """
    # get feedback:
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # select kernel:
    log_kernel = _gauss_kde_logpdf
    # run:
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf = parallel(joblib.delayed(log_kernel)
                                 (samp, white_samples, weights)
                                 for samp in feedback_helper(white_samples))
    # filter for probability calculation:
    _filter = _kde_eval_pdf > zero_prob
    # compute number of filtered elements:
    _num_filtered = np.sum(weights[_filter])
    #
    return _num_filtered


def _neighbor_parameter_shift(white_samples, weights, zero_prob, num_samples,
                              feedback, **kwargs):
    """
    """
    # import specific for this function:
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # get options:
    stable_cycle = kwargs.get('stable_cycle', 4)
    chunk_size = kwargs.get('chunk_size', 40)
    smallest_improvement = kwargs.get('smallest_improvement', 1.e-4)
    # the tree elimination has to work with probabilities to go incremental:
    _zero_prob = np.exp(zero_prob)
    # build tree:
    if feedback > 1:
        print('Building KD-Tree')
    data_tree = cKDTree(white_samples, leafsize=chunk_size,
                        balanced_tree=True)
    # make sure that the weights are floats:
    _weights = weights.astype(np.float)
    # initialize the calculation to zero:
    _num_elements = len(_weights)
    _kde_eval_pdf = np.zeros(_num_elements)
    _filter = np.ones(_num_elements, dtype=bool)
    _last_n = 0
    _stable_cycle = 0
    # loop over the neighbours:
    if feedback > 1:
        print('Neighbours elimination')
    for i in range(_num_elements//chunk_size):
        ind_min = chunk_size*i
        ind_max = chunk_size*i+chunk_size
        _dist, _ind = data_tree.query(white_samples[_filter],
                                      ind_max, n_jobs=-1)
        _kde_eval_pdf[_filter] += np.sum(
            _weights[_ind[:, ind_min:ind_max]]
            * np.exp(-0.5*np.square(_dist[:, ind_min:ind_max])), axis=1)
        _filter[_filter] = _kde_eval_pdf[_filter] < _zero_prob
        _num_filtered = np.sum(_filter)
        if feedback > 2:
            print('neighbor_elimination: chunk', i+1)
            print('    surviving elements', _num_filtered,
                  'of', _num_elements)
        # check if calculation has converged:
        _term_check = float(np.abs(_num_filtered-_last_n)) \
            / float(_num_elements) < smallest_improvement
        if _term_check and _num_filtered < _num_elements:
            _stable_cycle += 1
            if _stable_cycle >= stable_cycle:
                break
        elif not _term_check and _stable_cycle > 0:
            _stable_cycle = 0
        elif _num_filtered == 0:
            break
        else:
            _last_n = _num_filtered
    # clean up memory:
    del(data_tree)
    # brute force the leftovers:
    if feedback > 1:
        print('neighbor_elimination: polishing')
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf[_filter] = parallel(joblib.delayed(_gauss_kde_logpdf)
                                          (samp, white_samples, weights)
                                          for samp in feedback_helper(white_samples[_filter]))
        _filter[_filter] = _kde_eval_pdf[_filter] < np.log(_zero_prob)
    if feedback > 1:
        print('    surviving elements', np.sum(_filter),
              'of', _num_elements)
    # compute number of filtered elements:
    _num_filtered = num_samples - np.sum(weights[_filter])
    #
    return _num_filtered


def kde_parameter_shift_1D_fft(diff_chain, param_names=None,
                               scale=None, nbins=1024, feedback=1,
                               boundary_correction_order=1,
                               mult_bias_correction_order=1,
                               **kwarks):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain in 2 dimensions and by using FFT.
    This function uses GetDist 2D fft and optimal bandwidth estimates to
    perform the MCMC parameter shift integral discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses GetDist optimized bandwidth.
    :param nbins: (optional) number of 2D bins for the fft.
        Default is 1024.
    :param mult_bias_correction_order: (optional) multiplicative bias
        correction passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param boundary_correction_order: (optional) boundary correction
        passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :return: probability value and error estimate.
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # check that we only have two parameters:
    if len(param_names) != 1:
        raise ValueError('Calling 1D algorithm with more than 1 parameters')
    # initialize scale:
    if scale is None or isinstance(scale, str):
        scale = -1
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # compute the density with GetDist:
    t0 = time.time()
    density = diff_chain.get1DDensity(name=ind[0], normalized=True,
                                      num_bins=nbins,
                                      smooth_scale_1D=scale,
                                      boundary_correction_order=boundary_correction_order,
                                      mult_bias_correction_order=mult_bias_correction_order)
    # initialize the spline:
    density._initSpline()
    # get density of zero:
    prob_zero = density.Prob([0.])[0]
    # do the MC integral:
    probs = density.Prob(diff_chain.samples[:, ind[0]])
    # filter:
    _filter = probs > prob_zero
    # if there are samples above zero then use MC:
    if np.sum(_filter) > 0:
        _num_filtered = float(np.sum(diff_chain.weights[_filter]))
        _num_samples = float(np.sum(diff_chain.weights))
        _P = float(_num_filtered)/float(_num_samples)
        _low, _upper = utils.clopper_pearson_binomial_trial(_num_filtered,
                                                            _num_samples,
                                                            alpha=0.32)
    # if there are no samples try to do the integral:
    else:
        norm = simps(density.P, density.x)
        _second_filter = density.P < prob_zero
        density.P[_second_filter] = 0
        _P = simps(density.P, density.x)/norm
        _low, _upper = None, None
    #
    t1 = time.time()
    if feedback > 0:
        print('Time taken for 1D FFT-KDE calculation:', round(t1-t0, 1), '(s)')
    #
    return _P, _low, _upper


def kde_parameter_shift_2D_fft(diff_chain, param_names=None,
                               scale=None, nbins=1024, feedback=1,
                               boundary_correction_order=1,
                               mult_bias_correction_order=1,
                               **kwarks):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain in 2 dimensions and by using FFT.
    This function uses GetDist 2D fft and optimal bandwidth estimates to
    perform the MCMC parameter shift integral discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses GetDist optimized bandwidth.
    :param nbins: (optional) number of 2D bins for the fft.
        Default is 1024.
    :param mult_bias_correction_order: (optional) multiplicative bias
        correction passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param boundary_correction_order: (optional) boundary correction
        passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :return: probability value and error estimate.
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # check that we only have two parameters:
    if len(param_names) != 2:
        raise ValueError('Calling 2D algorithm with more than 2 parameters')
    # initialize scale:
    if scale is None or isinstance(scale, str):
        scale = -1
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # compute the density with GetDist:
    t0 = time.time()
    density = diff_chain.get2DDensity(x=ind[0], y=ind[1], normalized=True,
                                      fine_bins_2D=nbins,
                                      smooth_scale_2D=scale,
                                      boundary_correction_order=boundary_correction_order,
                                      mult_bias_correction_order=mult_bias_correction_order)
    # initialize the spline:
    density._initSpline()
    # get density of zero:
    prob_zero = density.spl([0.], [0.])[0][0]
    # do the MC integral:
    probs = density.spl.ev(diff_chain.samples[:, ind[0]],
                           diff_chain.samples[:, ind[1]])
    # filter:
    _filter = probs > prob_zero
    # if there are samples above zero then use MC:
    if np.sum(_filter) > 0:
        _num_filtered = float(np.sum(diff_chain.weights[_filter]))
        _num_samples = float(np.sum(diff_chain.weights))
        _P = float(_num_filtered)/float(_num_samples)
        _low, _upper = utils.clopper_pearson_binomial_trial(_num_filtered,
                                                            _num_samples,
                                                            alpha=0.32)
    # if there are no samples try to do the integral:
    else:
        norm = simps(simps(density.P, density.y), density.x)
        _second_filter = density.P < prob_zero
        density.P[_second_filter] = 0
        _P = simps(simps(density.P, density.y), density.x)/norm
        _low, _upper = None, None
    #
    t1 = time.time()
    if feedback > 0:
        print('Time taken for FFT-KDE calculation:', round(t1-t0, 1), '(s)')
    #
    return _P, _low, _upper


def kde_parameter_shift(diff_chain, param_names=None,
                        scale=None, method='brute_force',
                        feedback=1, **kwargs):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain.
    This function uses a Kernel Density Estimate (KDE) algorithm discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).
    If the difference chain contains :math:`n_{\\rm samples}` this algorithm
    scales as :math:`O(n_{\\rm samples}^2)` and might require long run times.
    For this reason the algorithm is parallelized with the
    joblib library.
    To compute the KDE density estimate several methods are implemented.

    In the 2D case this defaults to
    :meth:`~tensiometer.mcmc_tension.exact_parameter_shift_2D_fft`
    unless the kwarg use_fft is False.

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses Silverman's
        rule of thumb scaling.
    :param method: (optional) a string containing the indication for the method
        to use in the KDE calculation. This can be very intensive so different
        techniques are provided.

        * method = brute_force is a parallelized brute force method. This
          method scales as :math:`O(n_{\\rm samples}^2)` and can be afforded
          only for small tensions. When suspecting a difference that is
          larger than 95% other methods are better.
        * method = neighbor_elimination is a KD Tree based elimination method.
          For large tensions this scales as
          :math:`O(n_{\\rm samples}\\log(n_{\\rm samples}))`
          and in worse case scenarions, with small tensions, this can scale
          as :math:`O(n_{\\rm samples}^2)` but with significant overheads
          with respect to the brute force method.
          When expecting a statistically significant difference in parameters
          this is the recomended algorithm.

        Suggestion is to go with brute force for small problems, neighbor
        elimination for big problems with signifcant tensions.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :param kwargs: extra options to pass to the KDE algorithm.
        The neighbor_elimination algorithm accepts the following optional
        arguments:

        * stable_cycle: (default 2) number of elimination cycles that show
          no improvement in the result.
        * chunk_size: (default 40) chunk size for elimination cycles.
          For best perfornamces this parameter should be tuned to result
          in the greatest elimination rates.
        * smallest_improvement: (default 1.e-4) minimum percentage improvement
          rate before switching to brute force.
        * use_fft: whether to use fft when possible.
    :return: probability value and error estimate.
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # in the 2D case use FFT:
    use_fft = kwargs.get('use_fft', True)
    if len(ind) == 1 and use_fft:
        res = kde_parameter_shift_1D_fft(diff_chain,
                                         param_names=param_names,
                                         scale=scale,
                                         feedback=feedback, **kwargs)
        _P, _low, _upper = res
        return _P, _low, _upper
    elif len(ind) == 2 and use_fft:
        res = kde_parameter_shift_2D_fft(diff_chain,
                                         param_names=param_names,
                                         scale=scale,
                                         feedback=feedback, **kwargs)
        _P, _low, _upper = res
        return _P, _low, _upper
    # some initial calculations:
    _num_samples = np.sum(diff_chain.weights)
    _num_params = len(ind)
    # number of effective samples:
    _num_samples_eff = np.sum(diff_chain.weights)**2 / \
        np.sum(diff_chain.weights**2)
    # whighten samples:
    _white_samples = _helper_whiten_samples(diff_chain.samples[:, ind],
                                            diff_chain.weights)
    # scale for the kde:
    if (isinstance(scale, str) and scale == 'MISE') or scale is None:
        scale = MISE_bandwidth_1d(_num_params, _num_samples_eff, **kwargs)
    elif isinstance(scale, str) and scale == 'AMISE':
        scale = AMISE_bandwidth(_num_params, _num_samples_eff)
    elif isinstance(scale, str) and scale == 'MAX':
        scale = MAX_bandwidth(_num_params, _num_samples_eff)
    elif isinstance(scale, int) or isinstance(scale, float):
        scale = scale*np.identity(int(_num_params))
    elif isinstance(scale, np.ndarray):
        if not scale.shape == (_num_params, _num_params):
            raise ValueError('Input scaling matrix does not have correct size')
        scale = scale
    # feedback:
    if feedback > 0:
        with np.printoptions(precision=3):
            print(f'N    samples    : {int(_num_samples)}')
            print(f'Neff samples    : {_num_samples_eff:.2f}')
            if np.count_nonzero(scale - np.diag(np.diagonal(scale))) == 0:
                print(f'Smoothing scale :', np.diag(scale))
            else:
                print(f'Smoothing scale :', scale)
    # scale the samples:
    _kernel_cov = sqrtm(np.linalg.inv(scale))
    _white_samples = _white_samples.dot(_kernel_cov)
    # probability of zero:
    _kde_prob_zero = _gauss_kde_logpdf(np.zeros(_num_params),
                                       _white_samples,
                                       diff_chain.weights)
    # compute the KDE:
    t0 = time.time()
    if method == 'brute_force':
        _num_filtered = _brute_force_parameter_shift(_white_samples,
                                                     diff_chain.weights,
                                                     _kde_prob_zero,
                                                     _num_samples,
                                                     feedback)
    elif method == 'neighbor_elimination':
        _num_filtered = _neighbor_parameter_shift(_white_samples,
                                                  diff_chain.weights,
                                                  _kde_prob_zero,
                                                  _num_samples,
                                                  feedback, **kwargs)
    else:
        raise ValueError('Unknown method provided:', method)
    t1 = time.time()
    # clean up:
    gc.collect()
    # feedback:
    if feedback > 0:
        print('KDE method:', method)
        print('Time taken for KDE calculation:', round(t1-t0, 1), '(s)')
    # probability and binomial error estimate:
    _P = float(_num_filtered)/float(_num_samples)
    _low, _upper = utils.clopper_pearson_binomial_trial(float(_num_filtered),
                                                        float(_num_samples),
                                                        alpha=0.32)
    #
    return _P, _low, _upper


def chi2_parameter_shift(diff_chain, param_names=None):
    """
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # some initial calculations:
    _num_samples = np.sum(diff_chain.weights)
    # whighten samples:
    _white_samples = _helper_whiten_samples(diff_chain.samples[:, ind],
                                            diff_chain.weights)
    # compute the chi2:
    X0 = np.average(_white_samples, axis=0, weights=diff_chain.weights)
    _filter = (_white_samples*_white_samples).sum(axis=1) > np.dot(X0, X0)
    _num_filtered = float(np.sum(diff_chain.weights[_filter]))
    _P = float(_num_filtered)/float(_num_samples)
    _low, _upper = utils.clopper_pearson_binomial_trial(_num_filtered,
                                                        _num_samples,
                                                        alpha=0.32)
    #
    return _P, _low, _upper

###############################################################################
# Flow-based gaussianization:

import tensorflow as tf
# tf.enable_v2_behavior()
# tf.config.run_functions_eagerly(True)
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

#from IPython.display import clear_output, set_matplotlib_formats
import scipy.stats
from . import gaussian_tension

from matplotlib import pyplot as plt
from collections.abc import Iterable

from scipy.stats import binom
from scipy.optimize import bisect

class DiffFlowCallback(tf.keras.callbacks.Callback):
    """
    Notations:
        - X designates samples in the original parameter difference space;
        - Y designates samples in the gaussian approximation space, computed from mean(X) and cov(X);
        - Z designates samples in the gaussianized space, connected to Y with a normalizing flow, with Z~N(0,1).
    """
    def __init__(self, diff_chain, param_names=None, Z2Y_bijector='MAF', pregauss_bijector=None, learning_rate=1e-4, feedback=1, **kwargs):
        # Chain
        self._init_diff_chain(diff_chain, param_names=None)

        # Model
        if Z2Y_bijector == 'MAF':
            Z2Y_bijector = _build_dist_maf(self.num_params, **kwargs, feedback=feedback)
        assert isinstance(Z2Y_bijector, tfp.bijectors.Bijector)

        # Transformed distribution
        self._init_transf_dist(Z2Y_bijector, learning_rate=1e-4)
        if feedback>0:
            print("Building flow")
            print("    - trainable parameters:", self.model.count_params())

        # Full bijector
        self.Z2X_bijector = tfb.Chain([self.Y2X_bijector, self.Z2Y_bijector])

        # Full distribution
        self.dist_learned = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(np.zeros(self.num_params, dtype=np.float32), np.ones(self.num_params, dtype=np.float32)), bijector=self.Z2X_bijector) # samples from std gaussian mapped to X

        # Metrics
        keys = ["loss", "shift0_pval", "shift0_nsigma"]
        self.log = {_k:[] for _k in keys}

        self.chi2Y = np.sum(self.Y**2, axis=1)
        self.chi2Y_ks, self.chi2Y_ks_p = scipy.stats.kstest(self.chi2Y, 'chi2', args=(self.num_params,))

        # Options
        self.feedback = feedback

        # Pre-gaussianization
        if pregauss_bijector is not None:
            # The idea is to introduce yet another step of deterministic gaussianization, eg using the prior CDF
            # or double prior (convolved with itself, eg a triangular distribution)
            raise NotImplementedError

    def _init_diff_chain(self, diff_chain, param_names=None):
        # initialize param names:
        if param_names is None:
            param_names = diff_chain.getParamNames().getRunningNames()
        else:
            chain_params = diff_chain.getParamNames().list()
            if not np.all([name in chain_params for name in param_names]):
                raise ValueError('Input parameter is not in the diff chain.\n',
                                'Input parameters ', param_names, '\n'
                                'Possible parameters', chain_params)

        # indexes:
        ind = [diff_chain.index[name] for name in param_names]
        self.num_params = len(ind)

        # Gaussian approximation
        mcsamples_gaussian_approx = gaussian_tension.gaussian_approximation(diff_chain, param_names=param_names)
        self.dist_gaussian_approx = tfd.MultivariateNormalTriL(loc=mcsamples_gaussian_approx.means[0].astype(np.float32), scale_tril=tf.linalg.cholesky(mcsamples_gaussian_approx.covs[0].astype(np.float32)))
        self.Y2X_bijector = self.dist_gaussian_approx.bijector

        # Samples
        self.X = diff_chain.samples[:,ind]
        self.weights = diff_chain.weights
        self.Y = np.array(self.Y2X_bijector.inverse(self.X))
        assert not np.any(np.isnan(self.Y))

        Y_ds = tf.data.Dataset.from_tensor_slices((self.Y.astype(np.float32),
                                                   np.zeros(self.Y.shape[0], dtype=np.float32),
                                                   self.weights * len(self.weights) / np.sum(self.weights),)) #normalize weights to number of samples
        Y_ds = Y_ds.prefetch(tf.data.experimental.AUTOTUNE).cache()
        self.Y_ds = Y_ds.shuffle(self.num_params, reshuffle_each_iteration=True).repeat()

    def _init_transf_dist(self, Z2Y_bijector, learning_rate=1e-4):
        self.Z2Y_bijector = Z2Y_bijector
        self.dist_transformed = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(np.zeros(self.num_params, dtype=np.float32), np.ones(self.num_params, dtype=np.float32)), bijector=Z2Y_bijector)

        # Construct model
        x_ = Input(shape=(self.num_params,), dtype=tf.float32)
        log_prob_ = self.dist_transformed.log_prob(x_)
        self.model = Model(x_, log_prob_)

        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=lambda _, log_prob: -tf.reduce_mean(tf.boolean_mask(log_prob, tf.math.is_finite(log_prob))))

    def train(self, batch_size=256, epochs=100, steps_per_epoch=32, callbacks=[], verbose=1, **kwargs):
        hist = self.model.fit(x=self.Y_ds.batch(batch_size),
                        batch_size=batch_size,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=verbose,
                        callbacks=[tf.keras.callbacks.TerminateOnNaN(), self]+callbacks,
                        **kwargs)

        return hist

    def estimate_shift_significance(self, tol=0.05, max_iter=1000, step=1000000):
        err = np.inf
        counter = max_iter

        _thres = self.dist_learned.log_prob(np.zeros(self.num_params, dtype=np.float32))

        def binomial_ci( k, N, alpha):
            to_minimise = lambda c: binom.cdf(k,N,c)-alpha
            return bisect(to_minimise,0,1)

        res = []
        while err>tol and counter>=0:
            _s = self.dist_learned.sample(step)
            res += list(np.array(self.dist_learned.log_prob(_s) > _thres))

            r = np.array(res)
            k = np.sum(r)
            N = len(r)
            p = float(k)/N

            if k==0:
                continue

            try:
                pu = binomial_ci(k, N, 1.-0.68/2)
                pl = binomial_ci(k, N, 0.68/2)
            except:
                print(k,N,"error")
                continue
            err = np.abs(utils.from_confidence_to_sigma(pu)-utils.from_confidence_to_sigma(pl))
            counter -= 1

        return p, utils.from_confidence_to_sigma(p)


    def compute_shift_proba(self):
        zero = np.array(self.Z2X_bijector.inverse(np.zeros(self.num_params, dtype=np.float32)))
        chi2Z0 = np.sum(zero**2)
        pval = scipy.stats.chi2.cdf(chi2Z0, df=self.num_params)
        nsigma = utils.from_confidence_to_sigma(pval)
        return zero, pval, nsigma

    def plot_loss(self, ax, logs={}):
        self.log["loss"].append(logs.get('loss'))
        if ax is not None:
            # ax.plot(np.ma.masked_array(np.array(self.log["loss"]), utils.is_outlier(np.array(self.log["loss"]))))
            ax.plot(self.log["loss"])
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Loss")

    def plot_shift_proba(self, ax, logs={}):
        zero, pval, nsigma = self.compute_shift_proba()
        self.log["shift0_pval"].append(pval)
        self.log["shift0_nsigma"].append(nsigma)
        logs["zero"] = zero
        logs["shift0_pval"] = pval
        logs["shift0_nsigma"] = nsigma
        if ax is not None:
            ax.plot(self.log["shift0_nsigma"])
            ax.set_title("Zero-shift significance ($\sigma$)")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("$\sigma$")

    def plot_chi2_dist(self, ax, logs={}):
        chi2Z = np.sum(np.array(self.Z2Y_bijector.inverse(self.Y))**2, axis=1)
        _s = np.isfinite(chi2Z)
        chi2Z = chi2Z[_s]
        try:
            chi2Z_ks, chi2Z_ks_p = scipy.stats.kstest(chi2Z, 'chi2', args=(self.num_params,))
        except:
            chi2Z_ks, chi2Z_ks_p = 0., 0.

        xx = np.linspace(0, self.num_params*4, 1000)
        bins = np.linspace(0, self.num_params*4, 100)
        if ax is not None:
            ax.plot(xx, scipy.stats.chi2.pdf(xx, df=self.num_params), label='$\chi^2_{{{}}}$ PDF'.format(self.num_params), c='k', lw=1)
            ax.hist(self.chi2Y, bins=bins, density=True, histtype='step', color='orange', weights=self.weights, label='Pre-gauss ($D_n$={:.3f})'.format(self.chi2Y_ks)); #, $p$={:.3f})'.format(self.chi2Y_ks, self.chi2Y_ks_p));
            ax.hist(chi2Z, bins=bins, density=True, histtype='step', color='dodgerblue', weights=self.weights[_s], label='Post-gauss ($D_n$={:.3f})'.format(chi2Z_ks)); #, $p$={:.3f})'.format(chi2Z_ks, chi2Z_ks_p));
            ax.set_title('$\chi^2_{{{}}}$ PDF'.format(self.num_params))
            ax.set_xlabel('$\chi^2$')
            ax.legend(fontsize=8)

    def on_epoch_end(self, epoch, logs={}):
        if self.feedback:
            if isinstance(self.feedback,int):
                if epoch % self.feedback:
                    return
            clear_output(wait=True)
            fig, axes = plt.subplots(1,3,figsize=(12,3))
        else:
            axes = [None]*3

        self.plot_loss(axes[0], logs=logs)
        self.plot_shift_proba(axes[1], logs=logs)
        self.plot_chi2_dist(axes[2], logs=logs)

        plt.tight_layout()
        plt.show()




def _build_dist_maf(num_params, n_maf=None, hidden_units=None, activation=tf.nn.leaky_relu, permutations=True, feedback=0, **kwargs):
    """[summary]

    :param num_params: [description]
    :type num_params: [type]
    :param n_maf: [description], defaults to None
    :type n_maf: [type], optional
    :param hidden_units: [description], defaults to None
    :type hidden_units: [type], optional
    :param activation: [description], defaults to tf.nn.leaky_relu
    :type activation: [type], optional
    :param learning_rate: [description], defaults to 1e-4
    :type learning_rate: [type], optional
    :return: [description]
    :rtype: [type]
    """
    if n_maf is None:
        n_maf = 2*num_params
    event_shape = (num_params,)

    if hidden_units is None:
        hidden_units = [num_params*2]*2

    if permutations is None:
        _permutations = False
    elif isinstance(permutations, Iterable):
        assert len(permutations)==n_maf
        _permutations = permutations
    elif isinstance(permutations, bool):
        if permutations:
            _permutations = [np.random.permutation(num_params) for _ in range(n_maf)]
        else:
            _permutations = False

    # Build transformed distribution
    bijectors = []
    for i in range(n_maf):
        if _permutations:
            bijectors.append(tfb.Permute(_permutations[i]))
        made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape, hidden_units=hidden_units, activation=activation, **kwargs)
        bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made))

    bijector = tfb.Chain(bijectors)

    if feedback>0:
        print("Building MAF")
        print("    - number of MAFs:", n_maf)
        # print("    - trainable parameters:", model.count_params())
        print("    - activation:", activation)
        print("    - hidden_units:", hidden_units)

    return bijector


def flow_parameter_shift(diff_chain, param_names=None, **kwargs):
    # Callback/model handler
    diff_flow_callback = DiffFlowCallback(diff_chain, param_names=param_names, **kwargs)

    diff_flow_callback.train()
