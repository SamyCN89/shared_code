#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 00:18:49 2025

@author: samy
"""
#%%
import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# =============================================================================
# Set Figure Params
# =============================================================================

def set_figure_params(savefig=False):
    plt.rcParams.update({
        'axes.labelsize': 15,
        'axes.titlesize': 13,
        'axes.spines.right': False,
        'axes.spines.top': False,
    })
    if savefig==True:
        return savefig


# =============================================================================
# Get Paths folder
# =============================================================================


def get_root_path():
    root = os.environ.get("PROJECT_DATA_ROOT")
    if not root:
        raise EnvironmentError("Environment variable PROJECT_DATA_ROOT is not set.")
    return Path(root)


def get_root_path_old(external_disk=True, external_path=None, internal_path=None):
    if external_path is None:
        external_path = '/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/'
    if internal_path is None:
        internal_path = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/'

    root_path = Path(external_path if external_disk else internal_path)

    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    return root_path

def get_paths(
    external_disk=True, 
    external_path=None, 
    internal_path=None,
    timecourse_folder="Timecourses_updated_03052024"
):
    """
    Generate a dictionary of paths for various data and result directories.

    Parameters:
        external_disk (bool): Whether to use the external disk path.
        external_path (str or None): Path to the external disk. Defaults to a predefined path.
        internal_path (str or None): Path to the internal disk. Defaults to a predefined path.
        timecourse_folder (str): Name of the folder containing timecourse data.

    Returns:
        dict: A dictionary containing paths with the following keys:
            - 'root': Root directory path.
            - 'results': Path to the results directory.
            - 'timeseries': Path to the timecourse data directory.
            - 'cog_data': Path to the cognitive data file.
            - 'mc': Path to the metaconnectivity results directory.
            - 'sorted': Path to the sorted data directory.
            - 'mc_mod': Path to the metaconnectivity modularity results directory.
            - 'allegiance': Path to the allegiance results directory.
            - 'figures': Path to the figures directory.
            - 'fmodularity': Path to the modularity figures directory.
    """
    root = get_root_path()
    return {
        'root': root,
        'results': root / 'results',
        'timeseries': root / f'results/{timecourse_folder}',
        'cog_data': root / f'results/{timecourse_folder}/ROIs.xlsx',
        'mc': root / 'results/mc/',
        'sorted': root / 'results/sorted_data/',
        'mc_mod': root / 'results/mc_mod/',
        'allegiance': root / 'results/allegiance/',
        'trimers': root / 'results/trimers/',
        'figures': root / 'fig',
        'fmodularity': root / 'fig/modularity',
        
    }

# =============================================================================
# Load data functions
# =============================================================================
# def load_cogdata_sorted(paths):
#     cog_data_filtered = pd.read_csv(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
#     data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz')
#     with open(paths['results'] / "grouping_data_oip.pkl", "rb") as f:
#         mask_groups, label_variables = pickle.load(f)
#     return data_ts, cog_data_filtered, mask_groups, label_variables

def load_cognitive_data(path_to_csv: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_csv)

def load_timeseries_data(path_to_npz: Path) -> dict:
    data = np.load(path_to_npz)
    return {
        'ts': data['ts'],
        'n_animals': int(data['n_animals']),
        'total_tp': data['total_tp'],
        'regions': data['regions'],
        'anat_labels': data['anat_labels'],
        'is_2month_old': data['is_2month_old']
    }

def load_timeseries(ts_file: Path) -> np.ndarray:
    """
    Load unstacked time series from a .npz file.

    Parameters
    ----------
    ts_file : Path
        Path to the .npz file containing 'ts'.

    Returns
    -------
    np.ndarray
        Array of time series data.
    """
    data = np.load(ts_file, allow_pickle=True)
    return data['ts']

def validate_alignment(ts_data: np.ndarray, cog_data: pd.DataFrame):
    """
    Ensure time series and cognitive data are aligned.

    Raises
    ------
    AssertionError
        If the lengths do not match.
    """
    assert len(ts_data) == len(cog_data), "Mismatch between time series and cognitive data entries."

#%% functions to load grouping data 

def load_grouping_data(path_to_pkl: Path):
    with open(path_to_pkl, "rb") as f:
        mask_groups, label_variables = pickle.load(f)
    return mask_groups, label_variables

# =============================================================================
# Preprocessing data
# =============================================================================

def filename_sort_mat(folder_path):
    """Read and sort MATLAB file names in a given folder path."""
    files_name = sorted(f for f in os.listdir(folder_path) if f.endswith('.mat'))

    return files_name


# def extract_hash_numbers(filenames, prefix='lot3_'):
#     hash_numbers = []
#     for name in filenames:
#         if prefix in name:
#             try:
#                 hash_num = int(name.split(prefix)[-1][:4])
#                 hash_numbers.append(hash_num)
#             except ValueError:
#                 print(f"Warning: Could not extract hash from {name}")
#     return hash_numbers

def load_matdata(folder_data, specific_folder, files_name):
    ts_list = []
    hash_dir        = os.path.join(folder_data, specific_folder)

    for idx,file_name in enumerate(files_name):
        file_path       = os.path.join(hash_dir, file_name)
        
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

def split_groups_by_age(group_masks, age_mask, group_labels=None, age_labels=('2m', '4m')):
    """
    Splits groups into 2 age-based subgroups each.

    Parameters:
    - group_masks: list of boolean arrays (e.g., [wt_mask, dki_mask])
    - age_mask: boolean array (e.g., is_2month_old)
    - group_labels: optional list of group names
    - age_labels: tuple for age split names ('2m', '4m')

    Returns:
    - masks: list of boolean masks
    - labels: list of strings matching each mask
    """
    group_masks = [np.tile(np.asarray(mask),2) for mask in group_masks]
    age_mask = np.asarray(age_mask)
    n_groups = len(group_masks)

    if group_labels is None:
        group_labels = [f"Group{i}" for i in range(n_groups)]

    masks = []
    labels = []

    for g_mask, g_label in zip(group_masks, group_labels):
        for is_2m, age_label in zip([True, False], age_labels):
            cond_mask = np.logical_and(g_mask, age_mask == is_2m)
            masks.append(cond_mask)
            labels.append(f"{g_label} {age_label}")

    return masks, labels


def classify_phenotypes(df, metric_prefix='OiP', threshold=0.2):
    """
    Classify cognitive phenotypes for a given metric, appending metric name to phenotype labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing the cognitive data.
        metric_prefix (str): Prefix for the metric (e.g., 'OiP', 'RO24H').
        threshold (float): Threshold to determine high vs. low performance.

    Returns:
        pd.DataFrame: DataFrame with a new column 'Phenotype_<metric>' with labels like 'good_OiP'.
    """
    col_2m = f"{metric_prefix}_2M"
    col_4m = f"{metric_prefix}_4M"

    good     = (df[col_2m] > threshold) & (df[col_4m] > threshold)
    learners = (df[col_2m] < threshold) & (df[col_4m] > threshold)
    impaired = (df[col_2m] > threshold) & (df[col_4m] < threshold)
    bad      = (df[col_2m] < threshold) & (df[col_4m] < threshold)

    labels = np.select(
        [good, learners, impaired, bad],
        [f'good', f'learners',
         f'impaired', f'bad'],
        default=f'undefined'
    )

    phenotype_column = f'Phenotype_{metric_prefix}'
    df = df.copy()
    df[phenotype_column] = pd.Categorical(
        labels,
        categories=[
            f'good', f'learners',
            f'impaired', f'bad'
        ],
        ordered=False
    )

    return df

def make_masks(group_dict, is_2month_old):
    masks = []
    labels = []
    for group, label in group_dict:
        mask, lab = split_groups_by_age(group, is_2month_old, label)
        masks.append(mask)
        labels.append(lab)
    return tuple(masks), tuple(labels)

def make_combination_masks(df, primary_col, by_col, primary_levels, by_levels, is_2month_old):
    labels = [f"{p}_{b}" for p in primary_levels for b in by_levels]
    conditions = [
        (df[primary_col] == p) & (df[by_col] == b)
        for p in primary_levels for b in by_levels
    ]
    return split_groups_by_age(tuple(conditions), is_2month_old, tuple(labels))


def matrix2vec(matrix3d):
    """
    Convert a 3D matrix into a 2D matrix by vectorizing each 2D matrix along the third dimension.
    
    Parameters:
    matrix3d (numpy.ndarray): 3D numpy array.
    
    Returns:
    numpy.ndarray: 2D numpy array where each column is the vectorized form of the 2D matrices from the 3D input.
    """
    #F: Frame, n: node
    F, n, _ = matrix3d.T.shape  # Assuming matrix3d shape is [F, n, n]
    return matrix3d.reshape((n*n,F))
