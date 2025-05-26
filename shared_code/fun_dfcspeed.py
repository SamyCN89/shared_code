#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:45:43 2024

@author: samy
""" 


# =============================================================================
# This code is for the functions in Ines folder
# =============================================================================

# =============================================================================
#  Mostly of the functions are the python version of the dfc speed toolbox
# Dynamic Functional Connectivity as a complex random walk: Definitions and the dFCwalk toolbox
# Lucas Arbabyazd, Diego Lombardo, Olivier Blin, Mira Didic, Demian Battaglia, Viktor Jirsa
# doi:  10.1016/j.mex.2020.101168 
# https://github.com/FunDyn/dFCwalk
# =============================================================================

from pathlib import Path
import numpy as np
import brainconn as bct

from tqdm import tqdm
import numexpr as ne
from joblib import Parallel, delayed, parallel_backend
from collections import Counter
import logging

from .fun_optimization import fast_corrcoef, fast_corrcoef_numba
from .fun_loaddata import *

#%%
# =============================================================================
# # fc and fcd functions
# =============================================================================

def ts2fc(timeseries, format_data = '2D', method='pearson'):
    """
    Calculate functional connectivity from time series data.
    
    Parameters:
    timeseries (array): Time series data of shape (timepoints, nodes).
    format_data (str): Output format, '2D' for full matrix or '1D' for lower-triangular vector.
    
    Returns:
    fc (array): Functional connectivity matrix ('2D') or vector ('1D').
    
    Adapted from Lucas Arbabyazd et al 2020. Methods X, doi: 10.1016/j.neuroimage.2020.117156
    """
    # Calculate correlation coefficient matrix
    if method=='pearson':
        fc = fast_corrcoef(timeseries)
        # fc = fast_corrcoef2(timeseries)
        # fc = fast_corrcoef_numba(timeseries)

        # fc = np.corrcoef(timeseries.T)
    elif method=='plv':
        fc = compute_plv_matrix_vectorized(timeseries.T)

    # Optionally zero out the diagonal for '2D' format
    if format_data=='2D':
        np.fill_diagonal(fc,0)#fill the diagonal with 0
        return fc
    elif format_data=='1D':
        # Return the lower-triangular part excluding the diagonal
        return fc[np.tril_indices_from(fc, k=-1)]

# Function to compute phase locking value (PLV)
def compute_plv_matrix_vectorized(data):
    """
    Compute Phase Locking Value (PLV) matrix for a multi-channel signal.
    
    Parameters:
    data : numpy array
        A 2D array of shape (channels, timepoints) where each row is a signal for a channel.
        
    Returns:
    plv_matrix : numpy array
        A 2D array of shape (channels, channels) representing PLV between each pair of channels.
    """
    num_channels = data.shape[0]
    
    # Compute the phase for each channel
    phase_data = np.angle(np.exp(1j * np.angle(data)))
    
    # Compute pairwise phase differences for all channels at once using broadcasting
    # The result is an array of shape (channels, channels, timepoints)
    phase_diff = phase_data[:, np.newaxis, :] - phase_data[np.newaxis, :, :]
    
    # Compute the complex exponential of the phase differences for all pairs
    # Shape remains (channels, channels, timepoints)
    complex_phase_diff = ne.evaluate("exp(1j * phase_diff)")
    
    # Compute the mean across timepoints (axis=-1) for all pairs of channels
    # and take the absolute value to get the PLV matrix
    plv_matrix = np.abs(np.mean(complex_phase_diff, axis=-1))
    
    # Ensure the diagonal is exactly 1 (because PLV between a channel and itself is 1)
    np.fill_diagonal(plv_matrix, 1.0)
    
    return plv_matrix
#%%
#===============================================================================
# Dynamic functional connectivity stream Functions
#===============================================================================
def ts2dfc_stream(ts, window_size, lag=None, format_data='2D', method='pearson'):
    """
    Compute dynamic functional connectivity (DFC) stream using a sliding window approach.

    Parameters:
        ts (np.ndarray): Time series data (timepoints x regions).
        window_size (int): Size of the sliding window.
        lag (int): Step size between windows (default = window_size).
        format_data (str): '2D' for vectorized FC, '3D' for FC matrices.
        method (str): Correlation method (currently only 'pearson').

    Returns:
        np.ndarray: DFC stream, either in 2D (n_pairs x frames) or 3D (n_regions x n_regions x frames).
    """
    t_total, n = ts.shape
    lag = lag or window_size
    frames = (t_total - window_size) // lag + 1
    n_pairs = n * (n - 1) // 2

    #Preallocate DFC stream
    if format_data == '2D':
        dfc_stream = np.empty((n_pairs, frames))
        tril_idx = np.tril_indices(n, k=-1)  # Precompute once
    elif format_data == '3D':
        dfc_stream = np.empty((n, n, frames))

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + window_size
        window = ts[wstart:wstop, :]
        fc = fast_corrcoef(window)

        if format_data == '2D':
            dfc_stream[:, k] = fc[tril_idx]
        else:
            dfc_stream[:, :, k] = fc

    return dfc_stream


def dfc_stream2fcd(dfc_stream):
    """
    Calculate the dynamic functional connectivity (dFC) matrix from a dfc_stream.
    
    Parameters:
    dfc_stream (numpy.ndarray): Input dynamic functional connectivity stream, can be 2D or 3D.
    
    Returns:
    numpy.ndarray: The dFC matrix computed as the correlation of the dfc_stream.
    """
    if dfc_stream.ndim < 2 or dfc_stream.ndim > 3:
        raise ValueError("Provide a valid size dfc_stream (2D or 3D)!")
    # Convert 3D dfc_stream to 2D if necessary
  
    if dfc_stream.ndim == 3:
        dfc_stream_2D = matrix2vec(dfc_stream)
    else:
        dfc_stream_2D = dfc_stream

    # Compute dFC
    dfc_stream_2D = dfc_stream_2D.T
    dfc = np.corrcoef(dfc_stream_2D)
    
    return dfc
#%%
logger = logging.getLogger(__name__)

def compute_dfc_stream(ts_data, window_size=7, lag=1, format_data='3D', save_path=None, n_jobs=-1):
    """
    Calculate dynamic functional connectivity (DFC) streams for time-series data.

    Parameters:
        ts_data (np.ndarray): 3D array (n_animals, n_regions, n_timepoints).
        window_size (int): Sliding window size.
        lag (int): Step size for the sliding window.
        format_data (str): '2D' for vectorized, '3D' for matrices.
        save_path (str): Directory to save results.
        n_jobs (int): Number of parallel jobs (-1 for all cores).

    Returns:
        np.ndarray: 4D array of DFC streams (n_animals, time_windows, roi, roi)
    """
    logger = logging.getLogger(__name__)

    n_animals, _, nodes = ts_data.shape
    full_save_path = make_save_path(save_path, "dfc", window_size, lag, n_animals, nodes)
    # get_save_path(save_path, window_size, lag, n_animals, nodes)

    print(f"full_save_path: {full_save_path}")
    # Load from cache if possible
    if full_save_path is not None and full_save_path.exists():
        return load_npz_cache(full_save_path, key="dfc_stream", label='dfc-stream')
        # dfc_stream = load_cached_dfc(full_save_path)

    # Compute DFC streams in parallel
    logger.info(f"Computing dFC stream in parallel (window_size={window_size}, lag={lag})...")
    with parallel_backend("loky", n_jobs=n_jobs):
        dfc_stream = np.stack(Parallel()(
            delayed(ts2dfc_stream)(ts_data[i], window_size, lag, format_data) for i in range(n_animals)
        ))
    dfc_stream = dfc_stream.astype(np.float32)  # Convert to float32 for memory efficiency
    # Save results if needed
    save_npz_stream(full_save_path, prefix='dfc_stream', dfc_stream=dfc_stream)
    return dfc_stream

def handler_tnet_analysis(ts_data, prefix='dfc', window_size=7, lag=1, format_data='3D', save_path=None, n_jobs=-1):
    """
    Calculate temporal network analysis (dfc_stream, meta-connectivity) for time-series data.

    Parameters:
        ts_data (np.ndarray): 3D array (n_animals, n_regions, n_timepoints).
        window_size (int): Sliding window size.
        lag (int): Step size for the sliding window.
        format_data (str): '2D' for vectorized, '3D' for matrices.
        save_path (str): Directory to save results.
        n_jobs (int): Number of parallel jobs (-1 for all cores).

    Returns:
        np.ndarray: 4D array of DFC streams (n_animals, time_windows, roi, roi)
    """
    logger = logging.getLogger(__name__)

    n_animals, _, nodes = ts_data.shape
    # Define the full save path based on parameters
    full_save_path = make_save_path(save_path, prefix, window_size, lag, n_animals, nodes)
    # get_save_path(save_path, window_size, lag, n_animals, nodes)
    # Load from cache if possible
    print(f"full_save_path: {full_save_path}")
    if full_save_path is not None and full_save_path.exists():
        if prefix == 'dfc':
            return load_npz_cache(full_save_path, key="dfc_stream", label='dfc-stream')
        else:
            return load_npz_cache(full_save_path, key=prefix, label='meta-connectivity')
            # dfc_stream = load_cached_dfc(full_save_path)

    # Compute tnet streams in parallel
    logger.info(f"Computing {prefix} (window_size={window_size}, lag={lag})...")

    dfc_stream = np.array([ts2dfc_stream(
        ts_data[i], window_size, lag, format_data) 
        for i in range(n_animals)])
    # with parallel_backend("loky", n_jobs=n_jobs):
    #     dfc_stream = np.stack(Parallel()(
    #         delayed(ts2dfc_stream)(ts_data[i], window_size, lag, format_data) for i in range(n_animals)
    #     ))
    dfc_stream = dfc_stream.astype(np.float32)  # Convert to float32 for memory efficiency
    # Save results if needed
    save_npz_stream(full_save_path, prefix='dfc_stream', dfc_stream=dfc_stream)
    return dfc_stream

#%%
# =============================================================================
# Speed functions from dFC data
# =============================================================================

def dfc_speed(dfc_stream, vstep=1):
    """
    Calculate speeds of variation in dfc over a specified step size.
    
    Parameters:
    dfc_stream (numpy.ndarray): Input dynamic functional connectivity stream (2D or 3D).
    vstep (int): Step size for computing speed of variation (default=1).
    
    Returns:
    speed_median (float): Median of computed distribution of speeds.
    Speeds (numpy.ndarray): Time series of computed speeds.
    """
    # Check the dimensionality of dfc_stream and process accordingly
    if dfc_stream.ndim == 3:
        # Assuming a reshapedfc_stream function exists to convert 3D dfc_stream to 2D
        fc_stream = dfc_stream.reshape(dfc_stream.shape[0]*dfc_stream.shape[1], dfc_stream.shape[2])
    elif dfc_stream.ndim == 2:
        fc_stream = dfc_stream
    else:
        raise ValueError("Provide a valid 2D or 3D dFC stream!")
    
    nslices = fc_stream.shape[1]
    speeds = np.empty(nslices - vstep)
    # speeds = []

    # Compute speeds using correlation distance
    # for sp in range(nslices - vstep):
    for sp in range(nslices - vstep):
        fc1 = fc_stream[:, sp]
        fc2 = fc_stream[:, sp + vstep]
        # Directly compute the Pearson correlation coefficient (faster than fast_corrcoef)
        covariance = np.cov(fc1, fc2)
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        speeds[sp] = 1 - correlation

    return np.median(speeds), speeds


#%%
# def dfc_speed_series(ts, window_parameter, lag=1, tau=3, get_speed_dist=False):
def dfc_speed_oversampled_series(ts, window_parameter, lag=1, tau=3, min_tau_zero=False, get_speed_dist=False):
    """
    Computes the median speed of dynamic functional connectivity (DFC) variation over a range of window sizes. 
    This function facilitates the analysis of DFC speed variations across different scales of temporal resolution.

    The computation is performed for each window size within the specified range and optionally for different
    values of temporal shift (tau). The function supports returning a distribution of speeds across all window sizes
    and tau values if required.

    Parameters
    ----------
    ts : numpy.ndarray
        Time series data from which the DFC stream is to be calculated.
    window_parameter : tuple of (int, int, int)
        A tuple specifying the minimum window size, maximum window size, and the step size for iterating through window sizes.
    lag : int, optional
        The lag parameter for DFC stream calculation, by default 1.
    tau : int, optional
        The maximum temporal shift to consider for over-sampled speed calculations, by default 3.
        Speeds will be calculated for shifts in the range [-tau, tau], inclusive.
    get_speed_dist : bool, optional
        If True, returns the flattened distribution of speeds across all considered window sizes and tau values,
        by default False.

    Returns
    -------
    numpy.ndarray
        An array containing the median speed of DFC variation for each window size considered.
    list of numpy.ndarray, optional
        A list of numpy arrays containing the distributions of speeds for each window size and tau value,
        returned only if `get_speed_dist` is True.
        
    Samy Castro 2024
    """
    
    if min_tau_zero==True:
        min_tau=0
    else:
        min_tau=-tau
    
    time_windows_min, time_windows_max, time_window_step = window_parameter
    time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)
    tau_array       = np.append(np.arange(min_tau, tau), tau ) 
    
    speed_windows_tau = np.zeros((len(time_windows_range), len(tau_array)))
    speed_dist    = []
    
    for idx_tt, tt in tqdm(enumerate(time_windows_range)):
    
        windows_size    = tt
    
        aux_dfc_stream   = ts2dfc_stream(ts, windows_size, lag, format_data='2D')
        height_stripe      = aux_dfc_stream.shape[1]-windows_size-tau
    
        speed_oversampl    = np.array([dfc_speed(aux_dfc_stream, vstep=windows_size + sp)[1][:height_stripe] for sp in tau_array])
        speed_windows_tau[idx_tt] = np.median(speed_oversampl,axis=1)

        if get_speed_dist==True:        # speed_dist = np.mean(speed_oversampl,axis=1)
            speed_dist.append(speed_oversampl.flatten())
        
    if get_speed_dist==True:        # speed_dist = np.mean(speed_oversampl,axis=1)
        return speed_windows_tau, speed_dist
    else:
        return speed_windows_tau



def parallel_dfc_speed_oversampled_series(ts, window_parameter, lag=1, tau=3, 
                                          min_tau_zero=False, get_speed_dist=False, 
                                          method='pearson', n_jobs=-1):
    """
    Compute DFC speed over a range of window sizes and tau values using parallel processing.

    Parameters:
        ts (np.ndarray): Time series data (timepoints x regions).
        window_parameter (tuple): (min_win, max_win, step).
        lag (int): Lag between windows (default: 1).
        tau (int): Max temporal shift for oversampling.
        min_tau_zero (bool): If True, tau starts at 0; else from -tau.
        get_speed_dist (bool): If True, also return full speed distributions.
        method (str): Correlation method.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).

    Returns:
        np.ndarray: Median speeds for each window size and tau.
        list[np.ndarray]: Flattened speed distributions (if get_speed_dist=True).
    Samy Castro 2024
    """
    
    # min_tau = 0 if min_tau_zero else -tau
    min_tau = 0 if min_tau_zero else -tau
    
    win_min, win_max, win_step = window_parameter
    time_windows_range = np.arange(win_min, win_max + 1, win_step)
    tau_array = np.append(np.arange(min_tau, tau), tau)

    def compute_speed_for_window_size(tt):
        aux_dfc_stream = ts2dfc_stream(ts, tt, lag, format_data='2D', method=method)
        height_stripe = aux_dfc_stream.shape[1] - tt - tau
        speed_oversampl = [dfc_speed(aux_dfc_stream, vstep=tt + sp)[1][:height_stripe] 
                           for sp in tau_array]
        return np.median(speed_oversampl, axis=1), speed_oversampl if get_speed_dist else None

    results = Parallel(n_jobs=n_jobs)(
                delayed(compute_speed_for_window_size)(tt) 
                for tt in tqdm(time_windows_range)
    )
    speed_medians, speed_dists = zip(*results) if get_speed_dist else (zip(*results), None)

    if get_speed_dist:
        # Flatten the speed_dist list of lists to a single list
        speed_dists = [item for sublist in speed_dists for item in sublist]
        return np.array(speed_medians), speed_dists
    else:
        return np.array(speed_medians)

#%%
# =============================================================================
# Window pooling functions of speed data
# =============================================================================

def pool_vel_windows(vel, lentau, limits, strategy="pad"):
    """
    Pool speed windows over short, mid, long ranges with optional padding or filtering.
    
    Parameters:
    - vel: list of arrays (length = n_animals)
    - lentau: int, number of tau values per window
    - limits: tuple (short_limit, mid_limit), in window units
    - strategy: 'pad' or 'drop' (for unequal pooled lengths)
    
    Returns:
    - pooled_dict: dict with keys 'short', 'mid', 'long' and 2D np.arrays
    """
    short_mid, mid_long = limits
    pooled = {'short': [], 'mid': [], 'long': []}
    max_lengths = {'short': 0, 'mid': 0, 'long': 0}
    
    for v in vel:
        segments = {
            'short': np.hstack(v[0 : short_mid * lentau]),
            'mid':   np.hstack(v[short_mid * lentau : mid_long * lentau]),
            'long':  np.hstack(v[mid_long * lentau :])
        }
        for k in segments:
            pooled[k].append(segments[k])
            max_lengths[k] = max(max_lengths[k], len(segments[k]))
    
    pooled_final = {}
    for k, values in pooled.items():
        if strategy == "drop":

            # keep only rows with common length
            lengths = [len(x) for x in values]
            most_common_len = Counter(lengths).most_common(1)[0][0]
            pooled_final[k] = np.array([x for x in values if len(x) == most_common_len])

        elif strategy == "pad":
            # pad with NaN to match max length
            padded = np.full((len(values), max_lengths[k]), np.nan)
            for i, x in enumerate(values):
                padded[i, :len(x)] = x
            pooled_final[k] = padded
        else:
            raise ValueError("strategy must be 'pad' or 'drop'")
    
    return pooled_final

def get_population_wpooling(wp_list, index_group):
    print('Group:', np.sum(index_group))
    index_mask = index_group.to_numpy() if hasattr(index_group, "to_numpy") else np.asarray(index_group)

    wp_wt = np.array([
        np.hstack([arr for i, arr in enumerate(wp_list[xx]) if index_mask[i]])
        for xx in range(3)
    ], dtype=object)

    # wp_wt = np.asarray([np.hstack(wp_list[xx][index_group]) for xx in range(3)], dtype=object)
    return wp_wt

#Remove at some point 
wpool_impaired = get_population_wpooling


#%%

