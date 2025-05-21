#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:47:31 2025

@author: samy
"""

import numpy as np
from scipy.stats import zscore
from numba import njit, prange

# =============================================================================
# =============================================================================
# # Optimization functions
# =============================================================================
# =============================================================================


@njit(parallel=True)
def fast_corrcoef_numba_parallel(X):
    """
    Parallel Numba version of Pearson correlation matrix.
    X shape: (T, N) → T timepoints, N features
    Returns: (N, N) correlation matrix
    """
    T, N = X.shape
    means = np.empty(N)
    stds = np.empty(N)
    corr = np.empty((N, N))

    # Compute means (parallel)
    for i in prange(N):
        s = 0.0
        for t in range(T):
            s += X[t, i]
        means[i] = s / T

    # Compute stds (parallel)
    for i in prange(N):
        s = 0.0
        for t in range(T):
            diff = X[t, i] - means[i]
            s += diff * diff
        stds[i] = (s / (T - 1)) ** 0.5

    # Compute correlation matrix (parallel upper triangle)
    for i in prange(N):
        for j in range(i, N):
            s = 0.0
            for t in range(T):
                s += (X[t, i] - means[i]) * (X[t, j] - means[j])
            cov = s / (T - 1)
            c = cov / (stds[i] * stds[j])
            corr[i, j] = c
            corr[j, i] = c

    return corr


@njit(fastmath=True)
def fast_corrcoef_numba(X):
    """
    Numba-accelerated Pearson correlation matrix for 2D array (observations x features).
    Equivalent to fast_corrcoef(X), assumes columns are variables (features).
    """
    T, N = X.shape
    out = np.empty((N, N))
    
    # Compute means and stds manually
    means = np.empty(N)
    stds = np.empty(N)
    for i in range(N):
        s = 0.0
        for t in range(T):
            s += X[t, i]
        means[i] = s / T

    for i in range(N):
        s = 0.0
        for t in range(T):
            diff = X[t, i] - means[i]
            s += diff * diff
        stds[i] = (s / (T - 1)) ** 0.5

    # Compute correlation matrix
    for i in range(N):
        for j in range(i, N):
            s = 0.0
            for t in range(T):
                s += (X[t, i] - means[i]) * (X[t, j] - means[j])
            cov = s / (T - 1)
            corr = cov / (stds[i] * stds[j])
            out[i, j] = corr
            out[j, i] = corr

    return out

# @njit(fastmath=True)
def fast_corrcoef(ts):
    """
    Numba-accelerated Pearson correlation matrix using z-score and dot product.
    ts: np.ndarray (timepoints, features)
    """
    n_samples, n_features = ts.shape
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0, ddof=1)
    z = (ts - mean) / std
    return np.dot(z.T, z) / (n_samples - 1)
