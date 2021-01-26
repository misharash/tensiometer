"""
Experimental features.

For test purposes:

import os, sys
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

# imports for parallel calculations:
import multiprocessing
import joblib
# number of threads available:
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = multiprocessing.cpu_count()

from getdist import loadMCSamples, MCSamples, WeightedSamples

# add path for correct version of tensiometer:
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import tensiometer.mcmc_tension as tmt
import tensiometer.utilities as utils

chain_1 = loadMCSamples('./tensiometer/test_chains/DES')
chain_2 = loadMCSamples('./tensiometer/test_chains/Planck18TTTEEE')
chain_12 = loadMCSamples('./tensiometer/test_chains/Planck18TTTEEE_DES')
chain_prior = loadMCSamples('./tensiometer/test_chains/prior')

import matplotlib.pyplot as plt

diff_chain = tmt.parameter_diff_chain(chain_1, chain_2, boost=1)
num_params, num_samples = diff_chain.samples.T.shape

param_names = None
scale = None
method = 'brute_force'
feedback=2
n_threads = 1
"""

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
from . import mcmc_tension as tmt

# imports for parallel calculations:
import multiprocessing
import joblib
# number of threads available:
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = multiprocessing.cpu_count()


def _gauss_ballkde_logpdf(x, samples, weights, distance_weights):
    """
    """
    X = x-samples
    return scipy.special.logsumexp(-0.5*(X*X).sum(axis=1)/distance_weights**2,
                                   b=weights)


def _brute_force_ballkde_param_shift(white_samples, weights, distance_weights,
                                     zero_prob, num_samples, feedback):
    """
    Brute force parallelized algorithm for parameter shift.
    """
    # get feedback:
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # account for variable smoothing in the weights:
    _num_params = white_samples.shape[1]
    weights_norm = weights/distance_weights**_num_params
    # run:
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf = parallel(joblib.delayed(_gauss_ballkde_logpdf)
                                 (samp, white_samples, weights_norm, distance_weights)
                                 for samp in feedback_helper(white_samples))
    # filter for probability calculation:
    _filter = _kde_eval_pdf > zero_prob
    # compute number of filtered elements:
    _num_filtered = np.sum(weights[_filter])
    #
    return _num_filtered


def _neighbor_ballkde_parameter_shift(white_samples, weights, distance_weights,
                              zero_prob, num_samples,
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
    # account for variable smoothing in the weights:
    _num_params = white_samples.shape[1]
    weights_norm = weights/distance_weights**_num_params
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
            weights_norm[_ind[:, ind_min:ind_max]]
            * np.exp(-0.5*np.square(_dist[:, ind_min:ind_max]/distance_weights[_ind[:, ind_min:ind_max]])), axis=1)
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
        _kde_eval_pdf[_filter] = parallel(joblib.delayed(_gauss_ballkde_logpdf)
                                         (samp, white_samples, weights_norm, distance_weights)
                                         for samp in feedback_helper(white_samples[_filter]))
        _filter[_filter] = _kde_eval_pdf[_filter] < np.log(_zero_prob)
    if feedback > 1:
        print('    surviving elements', np.sum(_filter),
              'of', _num_elements)
    # compute number of filtered elements:
    _num_filtered = num_samples - np.sum(weights[_filter])
    #
    return _num_filtered


def ball_kde_parameter_shift(diff_chain, param_names=None,
                             near=1, clipping=False, method='brute_force',
                             feedback=1, **kwargs):
    """
    near = 1
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
    _num_params = len(ind)
    # number of effective samples:
    _num_samples_eff = np.sum(diff_chain.weights)**2 / \
        np.sum(diff_chain.weights**2)
    # whighten samples:
    _white_samples = utils.whiten_samples(diff_chain.samples[:, ind],
                                          diff_chain.weights)
    # build tree:
    data_tree = cKDTree(_white_samples, balanced_tree=True)
    _dist, _ind = data_tree.query(_white_samples, near+1, n_jobs=-1)
    distance_weights = _dist[:, near]
    # taper if wanted:
    if clipping:
        distance_weights = np.maximum(distance_weights, tmt.AMISE_bandwidth(_num_params, _num_samples_eff)[0, 0])
    weights_norm = diff_chain.weights/distance_weights**_num_params
    del(data_tree)
    # feedback:
    if feedback > 0:
        with np.printoptions(precision=3):
            print(f'N    samples    : {int(_num_samples)}')
            print(f'Neff samples    : {_num_samples_eff:.2f}')

    # probability of zero:
    _kde_prob_zero = _gauss_ballkde_logpdf(np.zeros(_num_params),
                                           _white_samples,
                                           weights_norm,
                                           distance_weights)
    # compute the KDE:
    t0 = time.time()
    if method == 'brute_force':
        _num_filtered = _brute_force_ballkde_param_shift(_white_samples,
                                                         diff_chain.weights,
                                                         distance_weights,
                                                         _kde_prob_zero,
                                                         _num_samples,
                                                         feedback)
    elif method == 'neighbor_elimination':
        _num_filtered = _neighbor_ballkde_parameter_shift(_white_samples,
                                                  diff_chain.weights,
                                                  distance_weights,
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

##############################################################################
##############################################################################
##############################################################################


def _gauss_ellkde_logpdf(x, samples, weights, distance_weights):
    """
    """
    X = x-samples
    X = np.einsum('...j,...jk,...k', X, distance_weights, X)
    return scipy.special.logsumexp(-0.5*X, b=weights)


def _brute_force_ellkde_param_shift(white_samples, weights, weights_norm, distance_weights,
                                    zero_prob, num_samples, feedback):
    """
    Brute force parallelized algorithm for parameter shift.
    """
    # get feedback:
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # run:
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf = parallel(joblib.delayed(_gauss_ellkde_logpdf)
                                 (samp, white_samples, weights_norm, distance_weights)
                                 for samp in feedback_helper(white_samples))
    # filter for probability calculation:
    _filter = _kde_eval_pdf > zero_prob
    # compute number of filtered elements:
    _num_filtered = np.sum(weights[_filter])
    #
    return _num_filtered


def _neighbor_ellkde_parameter_shift(white_samples, weights, weights_norm,
                              distance_weights,
                              zero_prob, num_samples,
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
        X = white_samples[_ind[:, ind_min:ind_max]] - white_samples[_ind[:,0], np.newaxis, :]
        d2 = np.einsum('...j,...jk,...k', X, distance_weights[_ind[:, ind_min:ind_max]], X)
        _kde_eval_pdf[_filter] += np.sum(
            weights_norm[_ind[:, ind_min:ind_max]] * np.exp(-0.5*d2), axis=1)
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
        _kde_eval_pdf[_filter] = parallel(joblib.delayed(_gauss_ellkde_logpdf)
                                         (samp, white_samples, weights_norm, distance_weights)
                                         for samp in feedback_helper(white_samples[_filter]))
        _filter[_filter] = _kde_eval_pdf[_filter] < np.log(_zero_prob)
    if feedback > 1:
        print('    surviving elements', np.sum(_filter),
              'of', _num_elements)
    # compute number of filtered elements:
    _num_filtered = num_samples - np.sum(weights[_filter])
    #
    return _num_filtered


@jit(nopython=True)
def _helper(_ind, _white_samples, _num_params):
    mats = []
    dets = []
    for idx in _ind:
        temp_samp = _white_samples[idx]
        temp_samp = temp_samp[1:, :] - temp_samp[0, :]
        mat = np.zeros((_num_params, _num_params))
        for v in temp_samp:
            mat += np.outer(v, v)
        mats.append(np.linalg.inv(mat))
        dets.append(np.linalg.det(mat))
    return dets, mats

@jit(nopython=True)
def _helper2(mats, clipping_band):
    out_mats = []
    out_dets = []
    for mat in mats:
        eig, eigv = np.linalg.eigh(mat)
        eig = np.minimum(eig, 1./clipping_band**2.)
        out_mats.append(np.dot(np.dot(eigv, np.diag(eig)), eigv.T))
        out_dets.append(np.prod(1./eig))
    return out_dets, out_mats

def ell_kde_parameter_shift(diff_chain, param_names=None, clipping=False,
                            method='brute_force',
                            feedback=1, **kwargs):
    """
    near = 1
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
    _num_params = len(ind)
    # number of effective samples:
    _num_samples_eff = np.sum(diff_chain.weights)**2 / \
        np.sum(diff_chain.weights**2)
    # whighten samples:
    _white_samples = utils.whiten_samples(diff_chain.samples[:, ind],
                                          diff_chain.weights)
    # build tree:
    data_tree = cKDTree(_white_samples, balanced_tree=True)
    _dist, _ind = data_tree.query(_white_samples, _num_params+1, n_jobs=-1)
    del(data_tree)
    # compute the covariances:
    dets, mats = _helper(_ind, _white_samples, _num_params)
    # clipping:
    if clipping:
        clipping_band = tmt.AMISE_bandwidth(_num_params, _num_samples_eff)[0, 0]
        dets, mats = _helper2(mats, clipping_band)
    weights_norm = diff_chain.weights/np.sqrt(dets)
    norm_mats = np.array(mats)
    # feedback:
    if feedback > 0:
        with np.printoptions(precision=3):
            print(f'N    samples    : {int(_num_samples)}')
            print(f'Neff samples    : {_num_samples_eff:.2f}')
    # probability of zero:
    _kde_prob_zero = _gauss_ellkde_logpdf(np.zeros(_num_params),
                                           _white_samples,
                                           weights_norm,
                                           norm_mats)
    # compute the KDE:
    t0 = time.time()
    if method == 'brute_force':
        _num_filtered = _brute_force_ellkde_param_shift(_white_samples,
                                                         diff_chain.weights,
                                                         weights_norm,
                                                         norm_mats,
                                                         _kde_prob_zero,
                                                         _num_samples,
                                                         feedback)

    elif method == 'neighbor_elimination':
        _num_filtered = _neighbor_ellkde_parameter_shift(_white_samples,
                                                  diff_chain.weights,
                                                  weights_norm,
                                                  norm_mats,
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




pass
