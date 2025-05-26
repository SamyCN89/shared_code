#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:56:50 2024

@author: samy
"""

# =============================================================================
#  Functions to load data of Laura Harsan and Ines a
# Samy Castro March 2024
# =============================================================================

from pathlib import Path
import numpy as np
import os
from scipy.io import loadmat
# from .fun_dfcspeed import compute_for_window_size_new

#%%
def make_save_path(save_path, prefix, window_size, lag, n_animals, nodes):
    """
    Generate a consistent save path for cached files.

    Args:
        save_path (str or Path): Directory to save the file.
        prefix (str): File type prefix, e.g., 'mc' or 'dfc'.
        window_size (int): Window size parameter.
        lag (int): Lag parameter.
        n_animals (int): Number of animals.
        nodes (int): Number of regions/nodes.

    Returns:
        Path or None: Full file path for saving, or None if save_path not given.
    """
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path / f"{prefix}_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
    return None

def load_npz_cache(full_save_path, key, logger=None, label=None):
    """
    Load a value from an npz cache file by key.

    Args:
        full_save_path (Path or str): Path to the .npz file.
        key (str): Key to extract from the npz file (e.g., 'mc', 'dfc_stream').
        logger (logging.Logger, optional): Logger for info messages.
        label (str, optional): Label to use in printed/logged messages.

    Returns:
        The value from the npz file for the specified key, or None if not found.
    """
    if full_save_path and Path(full_save_path).exists():
        msg = f"Loading {label or key} from: {full_save_path}"
        if logger:
            logger.info(msg)
        print(msg)
        try:
            data = np.load(full_save_path, allow_pickle=True)
            if key in data:
                return data[key]
            else:
                print(f"Key '{key}' not found in cache file: {full_save_path}")
        except Exception as e:
            print(f"Failed to load cached {label or key} (reason: {e}). Recomputing...")
    return None
        

def save_npz_stream(save_path, prefix, **data):
    """
    Generate a save path and save data as a compressed npz file.

    Args:
        save_path (str or Path): Directory to save the file.
        prefix (str): Prefix for the file name (e.g., 'dfc', 'mc').
        **data: Data to save (e.g., dfc_stream=dfc_stream).

    Returns:
        Path or None: The path where data was saved, or None if save_path not given.
    """
    print('here')
    if save_path:
        # full_save_path = save_path / f"{prefix}_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
        print(f"Saving {prefix} stream to: {save_path}")
        np.savez_compressed(save_path, **data)
        return save_path
    return None

#%%
# Check if the prefix files exist and their sizes: using the check_and_rerun_missing_files function and get_missing_files function
def get_missing_files(paths, prefix, time_window_range, lag, n_animals, roi, size_threshold=1_000_000):
    """
    Check if the prefix files exist for all specified window sizes after computing.
    If a file is empty/corrupt, it will be added to the *missing_files* list.
    Args:
        paths (dict): Dictionary containing paths for saving files.
        prefix (str): Prefix for the file names. Now implemented as 'dfc' and 'mc'.
        time_window_range (list): List of time window sizes to check.
        lag (int): Lag parameter used in the computation.
        n_animals (int): Number of animals in the dataset.
        roi (list): List of regions of interest.
        size_threshold (int): Minimum file size threshold to consider a file valid.
    Returns:
        missing_files (list): List of time window sizes for which files are missing or invalid.     
    """
    missing_files = []
    for ws in time_window_range:
        # 1. Check the existence of the file for each window size
        full_save_path = make_save_path(paths, prefix, ws, lag, n_animals, roi)
        if not full_save_path.exists():
            missing_files.append(ws)
        # 2. Check if the file is empty or corrupt (less than 1 MB)
        else:
            if full_save_path.stat().st_size < size_threshold:  # This will raise an error if the file is not valid
                # Remove the file if it's empty or corrupt
                print(f"File {full_save_path} exists but is empty or corrupt. Removing it.")
                full_save_path.unlink(missing_ok=True)
                missing_files.append(ws)
    return missing_files

def check_and_rerun_missing_files(paths, prefix, time_window_range, lag, n_animals, roi):
    """
    Check for missing prefix files and compute them if necessary.
    Args:
        paths (dict): Dictionary containing paths for different data types.
        prefix (str): Prefix of the files to check. 'dfc' for DFC stream files, 'mc' for meta-connectivity files.
        time_window_range (np.ndarray): Array of time window sizes to check.
        lag (int): Lag parameter for DFC computation.
        n_animals (int): Number of animals in the dataset.
        roi (str): Region of interest for DFC computation.
    Returns:
        missing_files (list): List of time window sizes for which files are missing or invalid.
    """
    missing_files = get_missing_files(paths, prefix, time_window_range, lag, n_animals, roi)
    if not missing_files:
        print(f"All {prefix} files already exist.")
    else:
        print(f"Missing {prefix} files for window sizes:", missing_files)
        Parallel(n_jobs=min(PROCESSORS, len(missing_files)))(
            delayed(compute_for_window_size_new)(ws, prefix) for ws in missing_files
        )
    return missing_files


#%%
def filename_sort_mat(folder_path):
    """Read and sort MATLAB file names in a given folder path."""
    files_name      = np.sort(os.listdir(folder_path))
    return files_name


def extract_hash_numbers(filenames, prefix='lot3_'):
    """Extract hash numbers from filenames based on a given prefix."""
    hash_numbers    = [int(name.split(prefix)[-1][:4]) for name in filenames if prefix in name]
    return hash_numbers

def load_matdata(folder_data, specific_folder, files_name):
    ts_list = []
    hash_dir        = Path(folder_data) / specific_folder

    for idx,file_name in enumerate(files_name):
        file_path       = hash_dir / file_name
        
        try:
            data = loadmat(file_path)['tc']
            ts_list.append(data)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
    
    
    # Check if the first dimension is consistent
    first_dim_size = ts_list[0].shape[0]
    if all(data.shape[0] == first_dim_size for data in ts_list):
        # Convert the list to a NumPy array
        ts_array = np.array(ts_list)
        return ts_array
    else:
        print("Error: Inconsistent shapes along the first dimension.")
