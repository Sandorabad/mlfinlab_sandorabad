"""
# =============================================================================
# COMBINATORIAL_OPT.PY
# =============================================================================
# Code snippets from the first half of Chapter 21 of "Advances in Financial
# Machine Learning" by Marcos López de Prado. These functions deal with
# representing portfolio optimization as an integer optimization problem.
# =============================================================================
"""

# Imports.
import time
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement, product


# ==================================================
# SNIPPET 21.1. PARTITIONS OF k OBJECTS INTO n SLOTS
def pigeon_hole(k, n):
    """
    Organize k objects into n slots.
    :param k: (int) Number of discrete objects to organize.
    :param n: (int) Number of bins.
    :return:(generator object) Generator for getting the next allocation set.
    """
    for j in combinations_with_replacement(range(n), k):
        r = [0]*n
        for i in j:
            r[i] += 1
        yield r


# =====================================================================
# SNIPPET 21.2. SET omega OF ALL VECTORS ASSOCIATED WITH ALL PARTITIONS
def get_all_weights(k, n):
    """
    Given the allocation permutations from the 'pigeon_hole()' function,
    determine the set of all fractional allocation vectors (i.e. sum of
    absolute weights equals 1) by finding all combinations of
    negative and positive allocation weightings.

    :param k: (int) Units of capital to allocate.
    :param n: (int) Number of assets.
    :return: (numpy.array) Set of all fractional allocations across 'n' assets.
    """
    # 1) Generate partitions using 'pigeon_hole()'.
    parts = pigeon_hole(k, n)
    w = None

    # 2) Go through the partitions.
    for part_ in parts:
        w_ = np.array(part_) / float(k)
        for prod_ in product([-1, 1], repeat=n):
            w_signed_ = (w_*prod_).reshape(-1, 1)
            if w is None:
                w = w_signed_.copy()
            else:
                w = np.append(w, w_signed_, axis=1)
    return w


# =========================================
# SNIPPET 21.3. EVALUATING ALL TRAJECTORIES
def evaluate_t_costs(w, params):
    """
    Compute t-costs of a particular trajectory.

    :param w: (numpy.array) Matrix of all combinations of weights.
    :param params: (list) A list of dictionaries that contain values
     for ('c', 'mean', 'cov'), where 'c' is the relative transaction costs,
     and 'mean' and 'cov' are the forecasted mean and variance across assets.
    :return: (numpy.array) Matrix of the transaction costs across assets.
    """
    t_cost = np.zeros(w.shape[1])
    w_ = np.zeros(shape=w.shape[0])
    for i in range(t_cost.shape[0]):
        c_ = params[i]['c']
        t_cost[i] = (c_ * abs(w[:, i] - w_)**.5).sum()
        w_ = w[:, i].copy()
    return t_cost


def evaluate_sr(params, w, t_cost):
    """
    Evaluates the Sharpe Ratio (SR) over multiple horizons.

    :param params: (list) A list of dictionaries that contain values
     for ('c', 'mean', 'cov'), where 'c' is the relative transaction costs,
     and 'mean' and 'cov' are the forecasted mean and variance across assets.
    :param w: (numpy.array) Matrix of all combinations of weights.
    :param t_cost: (numpy.array) Matrix of the transaction costs across assets.
    :return: (numpy.array) A matrix of the Sharpe Ratios over multiple
     horizons.
    """
    mean = 0
    cov = 0
    for h in range(w.shape[1]):
        params_ = params[h]
        mean += np.dot(w[:, h].T, params_['mean'])[0] - t_cost[h]
        cov += np.dot(w[:, h].T, np.dot(params_['cov'], w[:, h]))
    sr = mean/cov**.5
    return sr


def dynamic_optimal_portfolio(params, k=None):
    """
    Dynamically calculate the optimal portfolio given the units of
    capital to allocate 'k', and the parameters describing the
    relative transaction costs, means, and variances, 'params'.

    :param params: (list) A list of dictionaries that contain values
     for ('c', 'mean', 'cov'), where 'c' is the relative transaction costs,
     and 'mean' and 'cov' are the forecasted mean and variance across assets.
    :param k: (int) Units of capital to allocate. Default is 'None' which
     assumes there is as many units of capital as there are assets.
    :return: (numpy.array) The optimal portfolio weights.
    """
    # 1) Generate partitions.
    if k is None:
        k = params[0]['mean'].shape[0]
    n = params[0]['mean'].shape[0]
    w_all = get_all_weights(k, n)
    sr = None

    # 2) Generate trajectories as cartesian product.
    for prod_ in product(w_all.T, repeat=len(params)):
        w_ = np.array(prod_).T  # Concatenate product into a trajectory.
        t_cost_ = evaluate_t_costs(w_, params)
        sr_ = evaluate_sr(params, w_, t_cost_)  # Evaluate trajectory.
        if (sr is None) or (sr < sr_):
            sr = sr_
            w = w_.copy()
    return w

