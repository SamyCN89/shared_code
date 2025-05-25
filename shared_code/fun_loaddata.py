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
    if save_path:
        # full_save_path = save_path / f"{prefix}_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
        print(f"Saving {prefix} stream to: {save_path}")
        np.savez_compressed(save_path, **data)
        return save_path
    return None


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
