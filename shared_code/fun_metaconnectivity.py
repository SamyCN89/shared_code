#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 00:16:53 2025

@author: samy
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import pandas as pd
from pathlib import Path
import copy
import pickle
from tqdm import tqdm

from itertools import combinations_with_replacement
from joblib import Parallel, delayed, parallel_backend

from .fun_dfcspeed import ts2dfc_stream
from .fun_loaddata import *
from .fun_optimization import fast_corrcoef#, fast_corrcoef_numba, fast_corrcoef_numba_parallel

import logging
# import time
# from functions_analysis import *
# from scipy.io import loadmat, savemat
# from scipy.special import erfc
# from scipy.stats import pearsonr, spearmanr

# from scipy.spatial.distance import squareform


#%%Metaconnectivity 

logger = logging.getLogger(__name__)

def animal_mc(ts, window_size, lag):
    """Compute MC for a single animal."""
    dfc = ts2dfc_stream(ts, window_size, lag, format_data='2D')
    mc = fast_corrcoef(dfc.T)
    return mc

def compute_metaconnectivity(ts_data, window_size=7, lag=1, save_path=None, n_jobs=-1):
    """Compute meta-connectivity matrices from time-series data using a sliding window approach.
    This function supports parallel computation and caching of results to optimize performance.
    Parameters:
    - ts_data: 3D numpy array of shape (n_animals, n_regions, n_timepoints)
    - window_size: Size of the sliding window (default: 7)
    - lag: Lag to apply to the time series (default: 1)
    - save_path: Path to save the computed meta-connectivity (default: None)
    - n_jobs: Number of parallel jobs to run (default: -1, use all available cores)
    Returns:
    - mc: 3D numpy array of shape (n_animals, n_regions, n_regions) representing the meta-connectivity matrices
    """
    n_animals, _, nodes = ts_data.shape
    full_save_path = make_save_path(save_path, "mc", window_size, lag, n_animals, nodes)
    # Load from cache if available
    if full_save_path is not None and full_save_path.exists():
        return load_npz_cache(full_save_path, key="mc", label='meta-connectivity')

    # Compute meta-connectivity in parallel
    with parallel_backend("loky", n_jobs=n_jobs):
        results = Parallel()(
            delayed(animal_mc)(ts_data[i].astype(np.float32), window_size, lag)
            for i in range(n_animals)
        )
    # Stack results into a 3D array
    mc = np.stack(results)
    # Save results if a save path is provided
    save_npz_stream(full_save_path, prefix='mc', mc=mc)
    if save_path:
        logger.info(f"Saving meta-connectivity to: {full_save_path}")
        np.savez_compressed(full_save_path, mc=mc)
    return mc

#%%
# Deprecated - Old version of compute_metaconnectivity function
def compute_metaconnectivity_old(ts_data, window_size=7, lag=1, return_dfc=False, save_path=None, n_jobs=-1):
    """
    This function calculates meta-connectivity matrices from time-series data using 
    a sliding window approach. It supports parallel computation and caching of results 
    to optimize performance.

    -----------
    ts_data : np.ndarray
        A 3D array of shape (n_animals, n_regions, n_timepoints) representing the 
        time-series data for multiple animals and brain regions.
    window_size : int, optional
        The size of the sliding window used for dynamic functional connectivity (DFC) 
        computation. Default is 7.
    lag : int, optional
        The lag parameter for time-series analysis. Default is 1.
    return_dfc : bool, optional
        If True, the function also returns the DFC stream. Default is False.
    save_path : str or None, optional
        The directory path where the computed meta-connectivity and DFC stream will 
        be saved. If None, results are not saved. Default is None.
    n_jobs : int, optional
        The number of parallel jobs to use for computation. Use -1 to utilize all 
        available CPU cores. Default is -1.

    --------
    mc : np.ndarray
        A 3D array of meta-connectivity matrices for each animal.
    dfc_stream : np.ndarray, optional
        A 4D array of DFC streams for each animal, returned only if `return_dfc` is True.

    Notes:
    ------
    - If a `save_path` is provided and a cached result exists, the function will load 
      the cached data instead of recomputing it.
    - The function uses joblib for parallel computation, with the "loky" backend.
    - The meta-connectivity matrices are computed by correlating the DFC streams.

    Examples:
    ---------
    # Example usage:
    mc = compute_metaconnectivity(ts_data, window_size=10, lag=2, save_path="./cache")
    mc, dfc_stream = compute_metaconnectivity(ts_data, return_dfc=True, n_jobs=4)
    """

    n_animals, tr_points, nodes  = ts_data.shape
    dfc_stream  = None
    mc          = None

    # File path setup
    save_path = Path(save_path) if save_path else None
    full_save_path = (
        save_path / f"mc_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
        if save_path else None
    )
    if full_save_path:
        full_save_path.parent.mkdir(parents=True, exist_ok=True)
        # full_save_path = os.path.join(save_path, f'mc_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz')
        # os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    # Load from cache
    if full_save_path and full_save_path.exists():
        print(f"Loading meta-connectivity from: {full_save_path}")
        data = np.load(full_save_path, allow_pickle=True)
        mc = data['mc']
        dfc_stream = data['dfc_stream'] if return_dfc and 'dfc_stream' in data else None

    else:
        print(f"Computing meta-connectivity in parallel (window_size={window_size}, lag={lag})...")

        # Parallel DFC stream computation per animal
        with parallel_backend("loky", n_jobs=n_jobs):
            dfc_stream_list = Parallel()(
                delayed(ts2dfc_stream)(ts_data[i], window_size, lag, format_data='2D')
                # for i in tqdm(range(n_animals), desc="DFC Streams")
                for i in range(n_animals)
            )
        dfc_stream = np.stack(dfc_stream_list)

        # Parallel MC matrices per animal
        with parallel_backend("loky", n_jobs=n_jobs):
            mc_list = Parallel()(
                delayed(fast_corrcoef)(dfc.T)
                # for dfc in tqdm(dfc_stream, desc="Meta-connectivity")
                for dfc in dfc_stream
                )
        mc = np.stack(mc_list)

        # Save results if path is provided
        if full_save_path:
            print(f"Saving meta-connectivity to: {full_save_path}")
            if return_dfc:
                np.savez_compressed(full_save_path, mc=mc, dfc_stream=dfc_stream if return_dfc else None)
            else:
                np.savez_compressed(full_save_path, mc=mc)
    # print(f"Max RAM usage during run: {max(all_mem_use):.2f} MB")
    return (mc, dfc_stream) if return_dfc else mc


#%%
# =============================================================================
# Allegiance computing functions - Using und sign Louvain method
# Maybe add Leiden ? 
# =============================================================================

# Louvain method for community detection function
def _run_louvain(mc_data, gamma):
    """
    Run Louvain community detection on the N,N data.
    """
    Ci, Q = bct.modularity.modularity_louvain_und_sign(mc_data, gamma=gamma)
    return np.asanyarray(Ci, dtype=np.int32), Q

def _build_agreement_matrix(communities):
    """
    Vectorized computation of agreement matrix from community labels.
    This function computes the agreement matrix for a list of community labels.
    Each community label is a 1D array of integers representing the community
    assignment of each node. The agreement matrix is a 2D array where the entry
    (i, j) represents the number of communities that nodes i and j belong to.
    The function uses broadcasting to efficiently compute the agreement matrix
    without explicit loops.
    Parameters
    ----------
    communities : list of 1D arrays
        List of community labels for each run. Each array should have the same length.
    Returns
    -------
    agreement : 2D array
        The agreement matrix, where entry (i, j) represents the number of communities
        that nodes i and j belong to.
    Notes
    -----
    - The function assumes that all community labels are integers starting from 0.
    - The function uses broadcasting to compute the agreement matrix efficiently.
    - The resulting agreement matrix is symmetric and has the same shape as the input
      community labels.
    """
    n_runs, n_nodes = communities.shape
    agreement = np.zeros((n_nodes, n_nodes), dtype=np.uint16)

    for Ci in communities:
        # agreement += (Ci[:, None] == Ci[None, :])
        agreement += (Ci[:, None] == Ci)

    return agreement.astype(np.float32)

#%%

def contingency_matrix_fun(n_runs, mc_data, gamma_range=10, gmin=0.8, gmax=1.3, cache_path=None, ref_name='', n_jobs=-1):
    """
    Compute or load a contingency matrix from community detection runs using joblib and vectorized agreement matrix.
    """
    # Initialize parameters 
    n_nodes = mc_data.shape[0]
    gamma_mod = np.linspace(gmin, gmax, gamma_range)
    
    # Setup cache directory 
    if cache_path:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_ref_name = ref_name.replace(" ", "_")
        full_cache_path = cache_dir / f'contingency_matrix_ref={safe_ref_name}_regions={n_nodes}_nruns={n_runs}_gamma_repetitions={gamma_range}.pkl'
        if full_cache_path.exists():
            with full_cache_path.open('rb') as f:
                print(f"[cache] Loading contingency matrix from {full_cache_path}")
                return pickle.load(f)
    else:
        full_cache_path = None

    # Prepare job list for all gamma/runs
    job_list = [(gamma, run_id) 
                for gamma in gamma_mod 
                for run_id in range(n_runs)]

    # Run all in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(_run_louvain)(mc_data, gamma) for gamma, _ in tqdm(job_list, desc="Running Louvain jobs")
    )

    # Reshape into [gamma_index][runs]
    results_by_gamma = [[] for _ in range(gamma_range)]
    for (gamma, _), result in zip(job_list, all_results):
        gamma_idx = np.argmin(np.abs(gamma_mod - gamma))  # match gamma to index
        results_by_gamma[gamma_idx].append(result)

    # Initialize containers
    contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float32)
    gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float32)

    # Process per gamma
    for idx, gamma in enumerate(tqdm(gamma_mod, desc="Processing gammas")):
        results = results_by_gamma[idx]
        communities, modularities = zip(*results)
        communities = np.array(communities, dtype=np.int32)
        gamma_qmod_val[idx] = modularities

        # Build agreement matrix
        agreement = _build_agreement_matrix(communities)
        gamma_agreement_mat[idx] = agreement
        contingency_matrix += agreement

    contingency_matrix /= (n_runs * gamma_range)

    # Save to cache
    if full_cache_path is not None:
        with full_cache_path.open('wb') as f:
            pickle.dump((contingency_matrix, gamma_qmod_val, gamma_agreement_mat), f)
            print(f"[cache] Saved to {full_cache_path}")

    return contingency_matrix, gamma_qmod_val, gamma_agreement_mat

#%%
def allegiance_matrix_analysis(mc_data, n_runs=100, gamma_pt=10, cache_path=None, ref_name='', n_jobs=-1):

    """
    Wrapper to compute allegiance communities and sorting indices.

    Parameters
    ----------
    mc_data : ndarray
        Mean meta-connectivity matrix.
    n_runs : int
        Number of repetitions per gamma.
    gamma_pt : int
        Number of gamma values.
    cache_path : str or None
        Path to cache contingency matrix result.

    Returns
    -------
    allegancy_communities : ndarray
        Final Louvain community labels.
    argsort_allegancy_communities : ndarray
        Sorting indices by community.
    """
    # print('here',cache_path)
    # contingency_matrix, gamma_mean, gamma_std = contingency_matrix_fun_old(
    contingency_matrix, _, _ = contingency_matrix_fun(
        n_runs=n_runs, 
        mc_data=mc_data, 
        gamma_range=gamma_pt, 
        cache_path=cache_path, 
        ref_name=ref_name,
        n_jobs=n_jobs
    )

    allegancy_communities, allegancy_modularity_q = bct.modularity.modularity_louvain_und_sign(contingency_matrix, gamma=1.2)
    argsort_allegancy_communities = np.argsort(allegancy_communities)

    return allegancy_communities, argsort_allegancy_communities, allegancy_modularity_q, contingency_matrix

def fun_allegiance_communities(mc_data, n_runs=1000, gamma_pt=100, ref_name=None, save_path=None, n_jobs=-1):
    """ 
    Compute allegiance communities from a single or multiple mc matrices.

    
    Parameters:
        mc_data: 2D or 3D ndarray
        n_runs: int
        gamma_pt: float
        ref_name: str
        save_path: Path
        n_jobs: int
    Returns:
        communities, sort_idx, contingency_matrix
    """

    # Load from the cache if the file already exists 

    full_save_path = None 
    if save_path and ref_name:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        safe_ref_name = ref_name.replace(" ", "_")
        full_save_path = save_path / f"allegiance_{safe_ref_name}.joblib"

        # Load from cache if it exists
        if full_save_path.exists():
            print(f"[cache] Loading allegiance communities from {full_save_path}")
            return joblib.load(full_save_path)
            # Uncomment the following lines if you want to use pickle instead of joblib
            # with full_save_path.open('rb') as f:
            #     return pickle.load(f)
   
    # Compute the contingency matrix
    contingency_matrix, _, _ = contingency_matrix_fun(
        n_runs=n_runs, 
        mc_data=mc_data, 
        gamma_range=gamma_pt, 
        cache_path=save_path, 
        ref_name=ref_name, 
        n_jobs=n_jobs
    )
    # Compute the allegiance communities using the contingency matrix
    communities, sort_idx, _, contingency = allegiance_matrix_analysis(
        contingency_matrix,
        n_runs=n_runs,
        gamma_pt=gamma_pt,
        cache_path=save_path,
        ref_name=ref_name+'_recursive',
        n_jobs=n_jobs
    ) 

    # Sort the communities 
    # communities = communities[sort_idx] 
    # Save the results if save_path and ref_name are provided
    if full_save_path:
        print(f"[cache] Saving allegiance communities to {full_save_path}")
        # Save using joblib
        joblib.dump((communities, sort_idx, contingency), full_save_path)
    return communities, sort_idx, contingency  # Return the communities, sorting indices, and contingency matrix


def fun_allegiance_communities2(mc_data, n_runs=1000, gamma_pt=100, ref_name=None, save_path=None, n_jobs=-1):
    """
    Compute allegiance communities from a single or multiple meta-connectivity matrices.

    Parameters:
        mc_data: ndarray (2D or 3D)
            Meta-connectivity matrix or matrices.
        n_runs: int
            Number of Louvain runs per gamma value.
        gamma_pt: int
            Number of gamma values to test.
        ref_name: str
            Reference name for saving results.
        save_path: Path
            Directory to save results.
        n_jobs: int
            Number of parallel jobs.

    Returns:
        communities: ndarray
            Community labels for nodes.
        sort_idx: ndarray
            Sorting indices for nodes based on communities.
        contingency_matrix: ndarray
            Contingency matrix from Louvain runs.
    """
    def process_single(mc_matrix):
        communities, sort_idx, _, contingency = allegiance_matrix_analysis(
            mc_matrix, n_runs=n_runs, gamma_pt=gamma_pt, cache_path=save_path, ref_name=ref_name, n_jobs=n_jobs
        )
        return communities, sort_idx, contingency

    if mc_data.ndim == 3:
        # Process multiple MC matrices
        communities_list, sort_idx_list, contingency_list = zip(
            *(process_single(mc_data[i]) for i in range(mc_data.shape[0]))
        )
        communities = np.mean(communities_list, axis=0)
        sort_idx = np.argsort(communities)
        contingency_matrix = np.mean(contingency_list, axis=0)
    elif mc_data.ndim == 2:
        # Process a single MC matrix
        communities, sort_idx, contingency_matrix = process_single(mc_data)
    else:
        raise ValueError("Input mc_data must be 2D or 3D.")

    communities = communities[sort_idx]
    # Save results if save_path and ref_name are provided
    if save_path and ref_name:
        np.savez_compressed(
            Path(save_path) / f"allegiance_{ref_name}.npz",
            communities=communities,
            sort_idx=sort_idx,
            contingency=contingency_matrix
        )

    return communities, sort_idx, contingency_matrix
#%%
def allegiance_wrapper_(mc_data, n_runs=1000, gamma_pt=100, ref_name=None, save_path=None, n_jobs=-1):
    """
    Compute allegiance communities from one or more MC matrices.
    """
    mc_data = mc_data[None, ...] if mc_data.ndim == 2 else mc_data  # Standardize to 3D

    def process_single(mc_matrix):
        communities, sort_idx, _, contingency = allegiance_matrix_analysis(
            mc_matrix, n_runs=n_runs, gamma_pt=gamma_pt, cache_path=save_path,
            ref_name=ref_name, n_jobs=n_jobs)
        return communities, sort_idx, contingency
    # Compute allegiance communities for each matrix
    allegiances = [process_single(mc) for mc in mc_data]  # Updated to call process_single

    communities = np.mean([al[0] for al in allegiances], axis=0)
    sort_idx = np.argsort(communities)
    contingency = np.mean([al[2] for al in allegiances], axis=0)
    communities = communities[sort_idx]
    # Save results if save_path and ref_name are provided


    return communities, sort_idx, contingency

#%%
# Parallell cluster function of Allegiance communities


def load_merged_allegiance(paths, window_size=9, lag=1):
    """Load merged allegiance data. Use after running the merge_allegiance function.

    Args:
        paths (dict): Dictionary containing paths for the dataset.
        window_size (int, optional): Window size used for the analysis. Defaults to 9.
        lag (int, optional): Lag used for the analysis. Defaults to 1.

    Returns:
        tuple: A tuple containing the dfc_communities, sort_allegiances, and contingency_matrices.
        
    """
    ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)['ts']
    n_animals = len(ts)
    n_regions = ts[0].shape[1]
    filename = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
    path = paths['allegiance'] / f'merged_allegiance_{filename}.npz'
    data = np.load(path)
    return data['dfc_communities'], data['sort_allegiances'], data['contingency_matrices']

#%%
# =============================================================================
# Modularity functions
# =============================================================================
def intramodule_indices_mask(allegancy_communities):
    
    n_2 = len(allegancy_communities)

    # Dictionary mapping module → list of node indices in that module
    intramodules_idx = {
        mod: np.where(mod == allegancy_communities)[0] 
        for mod in np.unique(allegancy_communities)}
    
    # Build an array of (mod, i, j) for every intra-module pair (i, j)
    # Uses combinations_with_replacement to include self-connections (i == j)
    # pairs = [(mod, list(combinations_with_replacement(intramodules_idx[1], 2))) for mod in np.unique(allegancy_communities)]
    intramodule_indices = np.array([
        (mod, i, j)
        for mod in np.unique(allegancy_communities)
        for i, j in combinations_with_replacement(intramodules_idx[mod], 2)
        ]).T
    
    # intramodule_indices[:,intramodule_indices[0]==1]
    
    mc_modules_mask = np.zeros((n_2, n_2))
    for ind, mod in enumerate(range(1, np.max(np.unique(allegancy_communities))+1)):
        idx = np.abs(intramodules_idx[mod])
        mc_modules_mask[np.ix_(idx, idx)] = ind + 1
    
    return intramodules_idx, intramodule_indices, mc_modules_mask

#%%
# =============================================================================
# MC FC - connectivity indices 
# =============================================================================
# The following functions are used to compute the indices of the functional connectivity (FC) and meta-connectivity (MC) matrices.
def get_fc_mc_indices(regions, allegiance_sort=None):
    """
    Get the indices of the functional connectivity (FC) and meta-connectivity (MC) matrices
    for a given number of regions.
    The function returns the indices for both FC and MC matrices.

    Parameters:
    ----------     
    regions : int
        The number of regions in the functional connectivity matrix.
    allegiance_sort : array-like, optional
        An array of indices to sort the functional connectivity matrix.

    Returns:
    -------
    fc_idx : (N,2) ndarray
        The indices of the functional connectivity matrix.
    mc_idx : (M,2) ndarray
        The indices of the meta-connectivity matrix.
    """
    # Get the indices of the lower triangular part of the functional connectivity matrix
    # and the meta-connectivity matrix
    # The lower triangular part is used to avoid redundancy in the connectivity matrices.
    # The k=-1 argument excludes the diagonal.
    # The fc_idx and mc_idx arrays are used to index the functional connectivity and
    # meta-connectivity matrices, respectively.    
    fc_idx = np.array(np.tril_indices(regions, k=-1)).T
    #if sort allegiance provided, order the indices accordingly
    if allegiance_sort is not None:
        fc_idx = fc_idx[allegiance_sort]
    mc_idx = np.array(np.tril_indices(fc_idx.shape[0], k=-1)).T
    return fc_idx, mc_idx


def get_mc_region_identities(fc_idx, mc_idx):
    """
    Get the region identities for the functional connectivity (FC) and meta-connectivity (MC) matrices.
    The function reshapes the indices to group them into sets of 4 (representing connections
    between 4 regions) and transposes the result to make each column represent a meta-connection.
    Parameters:
    ----------
    fc_idx : (N,2) ndarray
        The indices of the functional connectivity matrix.
    mc_idx : (M,2) ndarray
        The indices of the meta-connectivity matrix.
    Returns:
    -------
    mc_reg_idx : (4, M) ndarray
        The reshaped indices for the meta-connectivity matrix.
    fc_reg_idx : (2, M) ndarray
        The reshaped indices for the functional connectivity matrix.
    """
 
    fc_reg_idx = fc_idx[mc_idx]  # shape: (n_mc, 2, 2)
    mc_reg_idx = fc_reg_idx.reshape(-1, 4).T  # shape: (4, n_mc)    
    return mc_reg_idx, fc_reg_idx

#%%
# =============================================================================
# Identify Trimers functions
# =============================================================================
# Compute trimer-based mask and index for meta-connectivity matrix

def compute_trimers_identity(regions, allegiance_sort=None):
    """
    Compute the indices of trimers in the meta-connectivity matrix.
    A trimer is defined as a set of three unique nodes among the four defining a meta-connection.
    The function returns the indices of the trimers, their region identities, and the apex node.

    Parameters
    ----------
    regions : int
        The number of regions in the functional connectivity matrix.

    Returns
    -------
    trimer_idx : (2, M) ndarray
        The indices of the trimers in the meta-connectivity matrix.
    trimer_reg_id : (4, M) ndarray
        The region identities of the trimers.
    trimer_apex : (M,) ndarray
        The apex node of each trimer.
    """
    # Get FC and MC indices
    fc_idx, mc_idx = get_fc_mc_indices(regions)
    
    # Get the indices of the regions in the functional connectivity matrix
    if allegiance_sort is not None:
        # Sort the indices based on the allegiance sort order
        fc_idx = fc_idx[allegiance_sort]
    
    mc_reg_idx, _ = get_mc_region_identities(fc_idx, mc_idx)

    # Identify trimers: rows with exactly 3 unique nodes
    unique_counts = np.apply_along_axis(lambda x: len(set(x)), axis=0, arr=mc_reg_idx)
    trimer_mask = unique_counts == 3

    # Extract trimer indices and region identities
    trimer_idx = mc_idx[trimer_mask].T
    trimer_reg_id = mc_reg_idx[:, trimer_mask]

    # Find apex node: the node that appears twice
    trimer_apex = np.full(trimer_reg_id.shape[1], np.nan)
    for i in range(trimer_reg_id.shape[1]):
        vals, counts = np.unique(trimer_reg_id[:, i], return_counts=True)
        repeated = vals[counts > 1]
        if repeated.size > 0:
            trimer_apex[i] = repeated[0]
    return trimer_idx, trimer_reg_id, trimer_apex



def build_trimer_mask(trimer_idx, trimer_apex, n_fc_edges):
    mask = np.zeros((n_fc_edges, n_fc_edges))
    np.fill_diagonal(mask, np.nan)
    for i in range(trimer_idx.shape[1]):
        a, b = trimer_idx[0, i], trimer_idx[1, i]
        apex_val = trimer_apex[i] + 1  # optional +1 offset
        mask[a, b] = mask[b, a] = apex_val
    return mask


#%%
def compute_mc_nplets_mask_and_index(regions, allegiance_sort=None):
    """
    Computes a mask and index array identifying trimers in the meta-connectivity matrix.

    Parameters
    ----------
    regions : int
        Number of regions in the functional connectivity matrix.
    mc_idx : ndarray
        Meta-connectivity indices (N, 2).
    allegiance_sort : ndarray or None
        Optional sort order for mc_idx/fc_idx (typically from allegiance sorting).

    Returns
    -------
    mc_nplets_mask : ndarray
        Symmetric matrix with apex node identifiers at trimer positions.
    mc_nplets_index : ndarray
        Array of apex node identifiers for each MC edge, or np.nan if not a trimer.
    """
    #Get indices of the functional connectivity (FC) and meta-connectivity (MC) matrices
    fc_idx, mc_idx = get_fc_mc_indices(regions)

    #Apply allegiance sorting if provided
    if allegiance_sort is not None:
        fc_idx = fc_idx[allegiance_sort]

    #Get the region identities for the functional connectivity (FC) and meta-connectivity (MC) matrices
    mc_reg_idx, _ = get_mc_region_identities(fc_idx, mc_idx)

    #Identify trimers: rows with exactly 3 unique nodes
    unique_counts = np.apply_along_axis(lambda x: len(set(x)), axis=0, arr=mc_reg_idx)
    trimer_mask = unique_counts == 3

    trimer_idx = mc_idx[trimer_mask].T
    trimer_reg_id = mc_reg_idx[:, trimer_mask]

    #Find root node: the node that appears twice
    trimer_apex = np.full(trimer_reg_id.shape[1], np.nan)
    for i in range(trimer_reg_id.shape[1]):
        vals, counts = np.unique(trimer_reg_id[:, i], return_counts=True)
        repeated = vals[counts > 1]
        if repeated.size > 0:
            trimer_apex[i] = repeated[0]

    #Build the mask and index array
    n_fc_edges = int(regions * (regions - 1) / 2)
    mask = np.zeros((n_fc_edges, n_fc_edges))
    np.fill_diagonal(mask, np.nan)
    for i in range(trimer_idx.shape[1]):
        a, b = trimer_idx[0, i], trimer_idx[1, i]
        apex_val = trimer_apex[i] + 1  # optional +1 offset
        mask[a, b] = mask[b, a] = apex_val

    # Get the indices of the lower triangular part of the mask
    mc_nplets_index = mask[mc_idx[:,0], mc_idx[:,1]]
    
    return mask, mc_nplets_index

#%%
# def compute_trimers_identity_old(regions):
#     """
#     Compute the indices of trimers in the meta-connectivity matrix.
#     A trimer is defined as a set of three unique nodes among the four defining a meta-connection.
#     The function returns the indices of the trimers, their region identities, and the apex node.

#     Parameters
#     ----------
#     regions : int
#         The number of regions in the functional connectivity matrix.

#     Returns
#     -------
#     trimer_idx : (2, M) ndarray
#         The indices of the trimers in the meta-connectivity matrix.
#     trimer_reg_id : (4, M) ndarray
#         The region identities of the trimers.
#     trimer_apex : (M,) ndarray
#         The apex node of each trimer.
#     """
#     # Get FC and MC indices
#     fc_idx, mc_idx = get_fc_mc_indices(regions)
#     mc_reg_idx, _ = get_mc_region_identities(fc_idx, mc_idx)

#     # Identify trimers: rows with exactly 3 unique nodes
#     unique_counts = np.apply_along_axis(lambda row: len(np.unique(row)), axis=0, arr=mc_reg_idx)
#     trimer_mask = unique_counts == 3

#     # Extract trimer indices and region identities
#     trimer_idx = mc_idx[trimer_mask].T
#     trimer_reg_id = mc_reg_idx[:, trimer_mask]

#     # Find apex nodes (nodes appearing twice in a trimer)
#     def find_apex(row):
#         unique_nodes, counts = np.unique(row, return_counts=True)
#         apex_candidates = unique_nodes[counts > 1]
#         return apex_candidates[0] if len(apex_candidates) > 0 else np.nan

#     trimer_apex = np.apply_along_axis(find_apex, axis=0, arr=trimer_reg_id)

#     return trimer_idx, trimer_reg_id, trimer_apex

def trimers_by_apex(trimer_values, trimer_reg_apex):
    """
    Splits trimer MC values by apex region and group.
    
    Parameters
    ----------
    mc_values : ndarray, shape (n_animals, n_trimers)
        Trimer values per subject.
    trimer_reg_apex : ndarray, shape (n_trimers,)
        Apex region for each trimer.
    index1, index2 : boolean arrays
        Group masks (e.g., Good vs Impaired)
    
    Returns
    -------
    regval_index1 : list of arrays
        Each entry: trimer values for group 1 animals, per apex.
    regval_index2 : list of arrays
        Each entry: trimer values for group 2 animals, per apex.
    apex_ids : ndarray
        Unique apex region IDs, in order.
    """
    unique_apexes = np.unique(trimer_reg_apex)
    
    regval = [
        trimer_values[:, trimer_reg_apex == apex]
        for apex in unique_apexes
        ]  # List of shape (n_animals, n_trimers_per_apex)

    return regval

# trimers_per_region = np.array(trimers_by_apex(trimers_mc_values, trimer_reg_apex))

#%%Genuine trimers
def trimers_leaves_fc(arr):
    flat = arr.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    non_repeated = unique[counts == 1]
    repeated = unique[counts == 2]
    return non_repeated
def trimers_root_fc(arr):
    flat = arr.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    # non_repeated = unique[counts == 1]
    repeated = unique[counts == 2]
    return repeated