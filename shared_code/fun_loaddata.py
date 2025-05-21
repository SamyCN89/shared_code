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
