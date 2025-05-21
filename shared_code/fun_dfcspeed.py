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

import numpy as np
import brainconn as bct
from tqdm import tqdm
import numexpr as ne
from joblib import Parallel, delayed
from collections import Counter

from .fun_optimization import fast_corrcoef, fast_corrcoef_numba
# =============================================================================
# # fc and fcd functions
# =============================================================================

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

#%%
def ts2dfc_stream(ts, window_size, lag=None, format_data='2D', method='pearson'):
    """
    Calculate dynamic functional connectivity stream (dfc_stream) from time series.

    Parameters:
    - ts: np.ndarray (timepoints x regions)
    - window_size: int
    - lag: int (defaults to window_size)
    - format_data: '2D' for vectorized, '3D' for matrices

    Returns:
    - dfc_stream: np.ndarray
    """
    t_total, n = ts.shape
    lag = lag or window_size
    frames = (t_total - window_size) // lag + 1
    n_pairs = n * (n - 1) // 2

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

def ts2dfc_stream_old(ts, windows_size, lag=None, format_data='2D', method='pearson'):
    """
    Calculate dynamic functional connectivity stream (dfc_stream) from time series data.

    Parameters:
    ts (array): Time series data of shape (t, n), where t is timepoints, n is regions.
    windows_size (int): Window size to slide over the ts.
    lag (int): Shift value for the window. Defaults to W if not specified.
    format (str): Output format. '2D' for a (l, F) shape, '3D' for a (n, n, F) shape.

    Returns:
    dFCstream (array): Dynamic functional connectivity stream.
    """

    t_total, n = np.shape(ts)
    #Not overlap
    lag = lag or windows_size
    # if lag is None:
    #     lag = windows_size
    # Calculate the number of frames/windows
    frames = (t_total - windows_size)//lag + 1
    n_pairs               = n * (n-1)//2 #number of pairwise correlations
    
    if format_data=='2D':
        dfc_stream = np.empty((n_pairs, frames))
    elif format_data=='3D':
        dfc_stream = np.empty((n, n, frames))
        

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + windows_size
        if format_data =='2D':
            dfc_stream[:, k]    = ts2fc(ts[wstart:wstop, :], '1D', method=method)  # Assuming TS2FC returns a vector
        elif format_data == '3D':
            dfc_stream[:, :, k] = ts2fc(ts[wstart:wstop, :], '2D',method=method)  # Assuming TS2FC returns a matrix
    #         dfc_stream[:, :, k] = fc
    return dfc_stream


#%%
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
# =============================================================================
# Speed functions
# =============================================================================

def dfc_speed(dfc_stream, vstep=1):
    """
    Calculate speeds of variation in dynamic functional connectivity over a specified step size.
    
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
    for sp in tqdm(range(nslices - vstep)):
        # if (sp + vstep)>0:
        # if (sp + vstep)>=0 or (nslices - vstep)>sp:
        fc1 = fc_stream[:, sp]
        fc2 = fc_stream[:, sp + vstep]
        # Directly compute the Pearson correlation coefficient
        covariance = np.cov(fc1, fc2)
        correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        speed = 1 - correlation
        speeds[sp] = speed

    # Calculate median speed
    speed_median    = np.median(speeds)

    return speed_median, speeds

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
    
        dfc_streamaux   = ts2dfc_stream(ts, windows_size, lag, format_data='2D')
        height_stripe      = dfc_streamaux.shape[1]-windows_size-tau
    
        speed_oversampl    = np.array([dfc_speed(dfc_streamaux, vstep=windows_size + sp)[1][:height_stripe] for sp in tau_array])
        speed_windows_tau[idx_tt] = np.median(speed_oversampl,axis=1)

        if get_speed_dist==True:        # speed_dist = np.mean(speed_oversampl,axis=1)
            speed_dist.append(speed_oversampl.flatten())
        
    if get_speed_dist==True:        # speed_dist = np.mean(speed_oversampl,axis=1)
        return speed_windows_tau, speed_dist
    else:
        return speed_windows_tau



def parallel_dfc_speed_oversampled_series(ts, window_parameter, lag=1, tau=3, 
                                          min_tau_zero=False, get_speed_dist=False, method='pearson', 
                                          n_jobs=-1):
    """
    Computes the median speed of dynamic functional connectivity (DFC) variation over a range of window sizes,
    using parallel processing for improved performance. This function facilitates the analysis of DFC speed
    variations across different scales of temporal resolution.

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
    min_tau_zero : bool, optional
        If True, the minimum tau value is set to 0, otherwise it is set to -tau, by default False.
    method : str, optional
        The method used for calculating functional connectivity, by default 'pearson'.
    n_jobs : int, optional
        The number of jobs to run in parallel, by default -1 (use all available cores).
        This parameter is passed to the joblib.Parallel function.
        If set to -1, all available cores will be used for parallel processing.
        If set to 1, no parallel processing will be used.
        If set to any other positive integer, that number of jobs will be used for parallel processing.
        This allows for flexibility in managing computational resources and optimizing performance based on the system's capabilities.
    Returns
    -------
    numpy.ndarray
        An array containing the median speed of DFC variation for each window size considered.
    list of numpy.ndarray, optional
        A list of numpy arrays containing the distributions of speeds for each window size and tau value,
        returned only if `get_speed_dist` is True.

    Notes
    -----
    This function relies on parallel processing to improve computation time, making it suitable for datasets
    where analyzing DFC variation across multiple scales and temporal shifts is computationally intensive.
    Samy Castro 2024
    """
    
    # min_tau = 0 if min_tau_zero else -tau
    min_tau = 0 if min_tau_zero else -tau
    
    win_min, win_max, win_step = window_parameter
    time_windows_range = np.arange(win_min, win_max + 1, win_step)
    tau_array = np.append(np.arange(min_tau, tau), tau)


    def compute_speed_for_window_size(tt):
        dfc_streamaux = ts2dfc_stream(ts, tt, lag, format_data='2D', method=method)
        height_stripe = dfc_streamaux.shape[1] - tt - tau
        speed_oversampl = [dfc_speed(dfc_streamaux, vstep=tt + sp)[1][:height_stripe] for sp in tau_array]
        return np.median(speed_oversampl, axis=1), speed_oversampl if get_speed_dist else None

    results = Parallel(n_jobs=n_jobs)(delayed(compute_speed_for_window_size)(tt) for tt in tqdm(time_windows_range))
    speed_medians, speed_dists = zip(*results) if get_speed_dist else (zip(*results), None)

    if get_speed_dist:
        # Flatten the speed_dist list of lists to a single list
        speed_dists = [item for sublist in speed_dists for item in sublist]
        return np.array(speed_medians), speed_dists
    else:
        return np.array(speed_medians)


# =============================================================================
# Window pooling functions
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


