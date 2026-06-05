"""

This scripts perform Kriging, i.e. Gaussian Process regression, to correct the AGB predictions of the model.
The script takes as input the AGB predictions of the model, the GEDI footprints, and the DEM data, and outputs
the corrected AGB predictions. The script performs k-fold cross validation, and saves the corrected predictions
to a file.

"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
import numpy as np
import rasterio as rs
import pandas as pd
from rasterio.crs import CRS
import geopandas as gpd
import torch
import gpytorch
from os.path import join
from time import time
from os.path import join, isdir, isfile
from os import makedirs
import gc
import pickle
from itertools import combinations
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import os
from copy import deepcopy
from os import getcwd

from kriging.core import *  # shared GP models + helpers (see kriging/core.py)
from sumatra.common import *  # shared Sumatra-GEDI helpers (see sumatra/common.py)


torch.backends.cuda.matmul.allow_tf32 = False
os.environ['WANDB_INIT_TIMEOUT'] = '300'

#######################################################################################################################
# Helper functions 
















NODATAVALS = {'S2' : 0, 'CH': 255, 'ALOS': 0, 'LC': 255, 'DEM': -9999, 'LC': 255}







def geographical_train_test_split(GEDI, model_name, order, width, height, val_size = 0.2, test_size = 0.3, stripe_size = 100, max_train_footprints = 25000, random_state = 42) :
    """
    This function takes the Sentinel-2 tile, and splits the data into a training, validation, and test set;
    such that they are geographically separated. The function either uses a checkerboard pattern, or a stripe
    pattern to split the data.

    Args:
    - GEDI: geopandas dataframe, GEDI data.
    - order: list of strings, order of the sets, e.g. ['train', 'val', 'test'].
    - val_size: float, size of the validation set.
    - test_size: float, size of the test set.
    - stripe_size: int, size of the stripes to use for the geographical split. If 1, a random split is performed.
    - max_train_footprints: int, maximum number of training footprints to use.
    - random_state: int, random state to use for the random split.

    Returns:
    - GEDI_train: geopandas dataframe, training set.
    - GEDI_val: geopandas dataframe, validation set.
    - GEDI_test: geopandas dataframe, test set.
    """

    if stripe_size == 1 :
        if test_size != 0 : 
            GEDI_train, GEDI_test = train_test_split(GEDI, test_size = test_size, random_state = random_state)
            GEDI_train, GEDI_val = train_test_split(GEDI_train, test_size = val_size, random_state = random_state)
        else: 
            GEDI_train, GEDI_val = train_test_split(GEDI, test_size = val_size, random_state = random_state)
            GEDI_test = None
        print(f'    Number of footprints: {GEDI_train.shape[0]} (train) | {GEDI_val.shape[0]} (validation) | {GEDI_test.shape[0] if GEDI_test is not None else 0} (test)')

        # Plot the random split
        plt.figure(figsize = (10,10))
        colors = {'train': '#5B9BD5', 'val': '#F79646', 'test': '#70AD47'}
        for split in ['train', 'val', 'test'] :
            if split == 'test' and GEDI_test is None : continue
            GEDI_split = GEDI_train if split == 'train' else (GEDI_val if split == 'val' else GEDI_test)
            plt.scatter(GEDI_split['col_idx'].values, GEDI_split['row_idx'].values, c = colors[split], label = split, s = 1)

        plt.legend()
        plt.title(f"Random Split (Stripe Size: {stripe_size})")
        plt.xlabel("Column Index")
        plt.gca().invert_yaxis()
        plt.ylabel("Row Index")
        plt.savefig(f'predictions/splits/checkerboard_{stripe_size}.png')
        plt.close()

        return GEDI_train, GEDI_val, GEDI_test

    # Number of rows and columns in the S2 tile
    footprints_rows, footprints_cols = GEDI['row_idx'].values, GEDI['col_idx'].values
    min_row, max_row = np.min(footprints_rows), np.max(footprints_rows)
    min_col, max_col = np.min(footprints_cols), np.max(footprints_cols)

    # Create a checkerboard pattern
    horizontal_stripes = np.arange(min_row, max_row + 1, stripe_size).tolist()
    vertical_stripes = np.arange(min_col, max_col + 1, stripe_size).tolist()
    cols, rows = np.meshgrid(vertical_stripes, horizontal_stripes)

    # Calculate the number of footprints in each square
    num_footprints = np.zeros((len(horizontal_stripes), len(vertical_stripes)))
    assert num_footprints.shape == cols.shape, f"num_footprints shape {num_footprints.shape} does not match hs shape {cols.shape}"
    for i in range(len(horizontal_stripes)) :
        for j in range(len(vertical_stripes)) :
            row, col = rows[i,j], cols[i,j]
            num_footprints[i,j] = len(GEDI[(GEDI['row_idx'] >= row) & (GEDI['row_idx'] < row + stripe_size) & (GEDI['col_idx'] >= col) & (GEDI['col_idx'] < col + stripe_size)])
    
    # Get the indices of the squares with footprints
    indices = [(i,j) for i in range(len(horizontal_stripes)) for j in range(len(vertical_stripes)) if num_footprints[i,j] > 0]
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    # Expected number of footprints in each set
    total_footprints = len(GEDI)
    assert total_footprints == np.sum(num_footprints), f"total footprints {total_footprints} does not match sum of num_footprints {np.sum(num_footprints)}"
    goal_num_test = int(test_size * total_footprints)
    goal_num_val = int(val_size * (total_footprints - goal_num_test))
    goal_num_train = min(total_footprints - goal_num_test - goal_num_val, int((1 - val_size) * max_train_footprints))
    goals = {'train': goal_num_train, 'val': goal_num_val, 'test': goal_num_test}
    print(f"    Expected number of footprints: {goal_num_train} (train) | {goal_num_val} (validation) | {goal_num_test} (test)")
    
    # Randomly assign each square to a set
    metrics = {split: {'goal': goals[split], 'num': 0, 'stripes': []} for split in order}
    for split in order:
        while metrics[split]['num'] < metrics[split]['goal'] :
            try: i,j = indices.pop(0)
            except IndexError: break
            metrics[split]['num'] += num_footprints[i,j]
            metrics[split]['stripes'].append((rows[i,j], cols[i,j]))
            
    # If any of the splits are empty, skip this CV
    if metrics['train']['num'] == 0 or metrics['val']['num'] == 0 or (metrics['test']['num'] == 0 and test_size > 0) :
        print(f"    Skipping this CV, because one of the splits is empty: {metrics['train']['num']} (train) | {metrics['val']['num']} (validation) | {metrics['test']['num']} (test)")
        return None, None, None

    # From the stripes, list the indices
    GEDI['split'] = 'none'
    for split in ['train', 'val', 'test'] :
        for patch in metrics[split]['stripes'] :
            row, col = patch
            rows, cols = list(range(row, row + stripe_size)), list(range(col, col + stripe_size))
            GEDI.loc[(GEDI['row_idx'].isin(rows) & GEDI['col_idx'].isin(cols)), ['split']] = split
    
    # Actually split the data
    GEDI_train, GEDI_val, GEDI_test = GEDI[GEDI['split'] == 'train'], GEDI[GEDI['split'] == 'val'], GEDI[GEDI['split'] == 'test']

    # Plot the checkerboard pattern
    # Create a 2d array to visualize the split
    split_map = np.zeros((height, width), dtype = np.uint8)
    for split in ['train', 'val', 'test'] :
        for patch in metrics[split]['stripes'] :
            row, col = patch
            rows, cols = list(range(row, min(row + stripe_size, height))), list(range(col, min(col + stripe_size, width)))
            if split == 'train' : split_map[np.ix_(rows, cols)] = 1
            elif split == 'val' : split_map[np.ix_(rows, cols)] = 2
            elif split == 'test' : split_map[np.ix_(rows, cols)] = 3
    
    # Create a colormap for the split map
    cmap = ListedColormap(['white', '#5B9BD5', '#F79646', '#70AD47'])
    # Plot the split map
    plt.figure(figsize = (10,10))
    plt.imshow(split_map, cmap = cmap, interpolation = 'nearest')
    plt.title('Geographical split of the GEDI data')
    plt.xticks()
    plt.yticks()
    plt.colorbar(ticks = [0, 1, 2, 3], label = 'Split', orientation = 'horizontal')
    plt.clim(-0.5, 3.5)
    plt.savefig(f'predictions/splits/checkerboard_{stripe_size}_{model_name}.png')

    # Return the data
    print(f'    Number of footprints: {GEDI_train.shape[0]} (train) | {GEDI_val.shape[0]} (validation) | {GEDI_test.shape[0]} (test)')
    return GEDI_train, GEDI_val, GEDI_test




def get_fold(all_GEDI, model_name, run, satisfied, s_id, val_holdout, test_holdout, stripe_size, width, height, max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std, test_indices = None, order = ['train', 'test', 'train', 'val'], norm_coords = False, norm_aux = None, max_split_diff = 15) :
    """
    This function performs the geographical train-test split of the GEDI data, and returns the training, validation, and test sets.
    It also checks that the RMSEs of the training, validation, and test sets are not too different, and reshuffles the data if they are.

    Args:
    - all_GEDI: geopandas dataframe, all the GEDI data.
    - run: wandb run object, to log the RMSEs.
    - satisfied: boolean, whether the RMSEs are satisfied or not.
    - s_id: int, the current split ID.
    - val_holdout: float, size of the validation set.
    - test_holdout: float, size of the test set.
    - stripe_size: int, size of the stripes to use for the geographical split.
    - max_train_footprints: int, maximum number of training footprints to use.
    - order: list of strings, order of the sets, e.g. ['train', 'val', 'test'].

    Returns:
    - satisfied: boolean, whether the RMSEs are satisfied or not.
    - s_id: int, the next split ID.
    - GEDI: geopandas dataframe, training set.
    - GEDI_val: geopandas dataframe, validation set.
    - GEDI_hold_out: geopandas dataframe, test set.
    - max_pairwise_diff: float, maximum pairwise difference between the RMSEs of the training, validation, and test sets.
    - others: tuple, containing the residuals, indices, AGB values, predictions, standard errors, DEM data, and standard deviation data.
    """

    # Do the splitting
    GEDI, GEDI_val, GEDI_hold_out = geographical_train_test_split(all_GEDI, model_name = model_name, order = order, width = width, height = height, val_size = val_holdout, \
                                                                test_size = test_holdout, stripe_size = stripe_size, \
                                                                max_train_footprints = max_train_footprints, random_state = 42 + s_id * 10)
    if GEDI is None: return False, s_id + 1, None, None, None, None, None

    # If test indices are provided, filter the test set to have only those indices
    if test_indices is not None: GEDI_hold_out = all_GEDI[all_GEDI['idx'].isin(test_indices)]

    # Get the row and column indices
    Y, X = GEDI['row_idx'].values, GEDI['col_idx'].values
    Y_val, X_val = GEDI_val['row_idx'].values, GEDI_val['col_idx'].values
    Y_test, X_test = GEDI_hold_out['row_idx'].values, GEDI_hold_out['col_idx'].values

    # Get the AGB values
    gedi_agb, gedi_agb_val, gedi_agb_test = GEDI['agbd'].values, GEDI_val['agbd'].values, GEDI_hold_out['agbd'].values

    # Get the predictions at those points
    predictions = pred_agb[Y, X]
    predictions_val = pred_agb[Y_val, X_val]
    if test_holdout > 0 : predictions_test = pred_agb[Y_test, X_test]
    else: predictions_test = None

    # Calculate the residuals
    residuals = predictions - gedi_agb
    residuals_val = predictions_val - gedi_agb_val
    if test_holdout > 0 : residuals_test = predictions_test - gedi_agb_test
    else: residuals_test = None

    # Calculate the RMSE
    rmse_train = np.sqrt(np.mean(np.pow(residuals, 2)))
    rmse_val = np.sqrt(np.mean(np.pow(residuals_val, 2)))
    if test_holdout > 0 : rmse_test = np.sqrt(np.mean(np.pow(residuals_test, 2)))
    else: rmse_test = -1
    print(f'    RMSEs: {rmse_train:.2f} (train) | {rmse_val:.2f} (validation) | {rmse_test:.2f} (test)')

    if not satisfied :
        # If the RMSEs are too different (> max_split_diff %) across splits, skip this fold
        if test_holdout > 0 : pairwise_differences = [2 * (val1 - val2) / (val1 + val2) * 100 for val1, val2 in list(combinations([rmse_train, rmse_val, rmse_test], 2))]
        else: pairwise_differences = [2 * (val1 - val2) / (val1 + val2) * 100 for val1, val2 in list(combinations([rmse_train, rmse_val], 2))]
        # If we've already tried many splits, adapt the tolerance for selection
        if s_id > 50 : tolerance = max_split_diff + 20
        elif s_id > 5 : tolerance = max_split_diff + 10
        else: tolerance = max_split_diff
        if np.any(np.abs(pairwise_differences) > tolerance) :
            satisfied = False
            print('    The RMSEs are too different across splits. Skipping this fold.')
        else: satisfied = True

    # Log the RMSEs if the fold is satisfied
    if satisfied:
        pairwise_differences = [0]
        
        # Get the GEDI standard errors
        agbd_se, agbd_se_val, agbd_se_test = None, None, None

        # Get the auxiliary data
        norm_values = {}
        if aux == 'STD' : 
            std = pred_std[Y, X]
            std_val = pred_std[Y_val, X_val]
            if norm_aux :
                if norm_aux == 'min_max' :
                    std_min, std_max = np.min(std), np.max(std)
                    norm_values[aux] = {'min': std_min, 'max': std_max}
                    std = (std - std_min) / (std_max - std_min)
                    std_val = (std_val - std_min) / (std_max - std_min)
            dem, gedi_dem, gedi_dem_val = None, None, None
        elif aux == 'none' :
            dem, gedi_dem, gedi_dem_val, std, std_val = None, None, None, None, None
        
        if extra_ft is not None : # shape (W,H,C)
            eft = extra_ft[Y, X, :]
            eft_val = extra_ft[Y_val, X_val, :]
            if norm_aux :
                if norm_aux == 'min_max' :
                    eft_min, eft_max = np.min(eft, axis=0), np.max(eft, axis=0)
                    norm_values['EFT'] = {'min': eft_min, 'max': eft_max}
                    eft = (eft - eft_min) / (eft_max - eft_min)
                    eft_val = (eft_val - eft_min) / (eft_max - eft_min)
        else: eft, eft_val = None, None

        if norm_coords : # min_max scaling of the coordinates
            print('    Normalizing the coordinates with min-max scaling')
            norm_values['coords_X'] = {'min': 0, 'max' : width}
            norm_values['coords_Y'] = {'min': 0, 'max' : height}
            og_X_train, og_Y_train = X.copy(), Y.copy()
            og_X_val, og_Y_val = X_val.copy(), Y_val.copy()
            X, Y = X / width, Y / height
            X_val, Y_val = X_val / width, Y_val / height
            # we don't do it on the test set on purpose

    else: 
        agbd_se, agbd_se_val, agbd_se_test = None, None, None
        dem, gedi_dem, gedi_dem_val = None, None, None
        std, std_val = None, None
        norm_values = {}
        og_X_train, og_Y_train = None, None
        og_X_val, og_Y_val = None, None
        eft, eft_val = None, None

    return satisfied, s_id + 1, GEDI, GEDI_val, GEDI_hold_out, max(pairwise_differences), \
        (residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test,
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test,
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val)
































def get_train_val_test_split(GEDI, model_name, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, \
                            upsampling_shape, width, height, run, test_holdout, val_holdout, max_train_footprints, max_split_diff, stripe_size, \
                            max_tries, test_indices = None, save_dir = './splits', SAVE = True, SPLITS_EXIST = False, perform_ood = False, \
                            fname = None, ood_extra = None) :

    # If it has already been computed, and ood is not necessary, load them
    if SPLITS_EXIST and (not perform_ood) :

        print("Loading existing splits.")
        with open(join(save_dir, fname), 'rb') as f: split_indices = pickle.load(f)
        GEDI_train = GEDI[GEDI['idx'].isin(split_indices['train'])]
        GEDI_val = GEDI[GEDI['idx'].isin(split_indices['val'])]
        GEDI_hold_out = GEDI[GEDI['idx'].isin(split_indices['test'])]
        residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val, \
            = get_data(GEDI_train, GEDI_val, GEDI_hold_out, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run)
        GEDI = GEDI_train
        print(f"Train/val/test sizes: {GEDI.shape[0]} | {GEDI_val.shape[0]} | {GEDI_hold_out.shape[0]}")
    
    # If the base split exists but we need to do ood
    elif SPLITS_EXIST and perform_ood :

        print("Loading existing test split and removing OOD samples to get new train and val splits.")
        with open(join(save_dir, fname), 'rb') as f: test_indices = pickle.load(f)['test']
        _GEDI_holdout = GEDI[GEDI['idx'].isin(test_indices)]
        GEDI = GEDI[~GEDI['idx'].isin(test_indices)]
        test_holdout = 0.0                
        
        # Load the statistics of the residuals and filter out the outliers            
        with open(join(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/correction', 'stats_residuals_vs_true_agb_0-25.pkl'), 'rb') as f: residuals_statistics = pickle.load(f)
        residuals_statistics = residuals_statistics['global']
        og_indices, og_gedi_agb, predictions = ood_extra
        residuals = predictions - og_gedi_agb
        true_agb_bins = np.arange(0, 51, 25) # bins for which we drop the OOD values: 0-25, 25-50 t/ha
        # true_agb_bins = np.arange(75, 126, 25) # bins for which we drop the OOD values: 75-100, 100-125 t/ha
        _lbs, _ubs = true_agb_bins[:-1], true_agb_bins[1:]
        indices_to_remove = []
        for lb, ub in zip(_lbs, _ubs):
            mask = (og_gedi_agb >= lb) & (og_gedi_agb < ub)
            bin_stats = residuals_statistics.get(f"{lb}-{ub}", None)
            if bin_stats is not None:
                mean, std = bin_stats['mean'], bin_stats['std']
                # Filter out outliers beyond mean ± std
                mask &= ((residuals < (mean - std)) | (residuals > (mean + std)))
            else: mask[:] = False
            indices_to_remove.extend(og_indices[mask])
        print(f"Removing {len(indices_to_remove)} out-of-distribution samples.")
        GEDI = GEDI[~GEDI['idx'].isin(indices_to_remove)]
        num_footprints = GEDI.shape[0]

        # Split the data into train/val sets #####################################################
        max_total_footprints = int(max_train_footprints / (1 - test_holdout))
        if num_footprints > max_total_footprints :
            print(f"Subsampling {num_footprints} footprints to {max_total_footprints} footprints.")
            GEDI = GEDI.sample(max_total_footprints, random_state = 42)
            num_footprints = max_train_footprints
        all_GEDI = GEDI.copy()

        # Repeat until we have a satisfying split
        satisfied, s_id, split_differences = False, 0, []
        while not satisfied and s_id < max_tries :
            satisfied, s_id, GEDI, GEDI_val, GEDI_hold_out, max_diff, rest = get_fold(all_GEDI, model_name, run, False, s_id, val_holdout, test_holdout, stripe_size, width, height, max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std, test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
            if max_diff is not None: split_differences.append(max_diff)

        # If could not find a satisfying split after max_tries attempts, skip this fold
        if not satisfied :
            if len(split_differences) == 0 : raise Exception('No valid splits were found, exiting.')
            print(f'Could not find a satisfying split after {max_tries} attempts. Picking least worst.')
            s_id = np.argmin(split_differences)
            _, _, GEDI, GEDI_val, GEDI_hold_out, _, rest = get_fold(all_GEDI, model_name, run, True, s_id, val_holdout, test_holdout, stripe_size, width, height, max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std, test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
        if GEDI is None or rest is None: raise Exception('No valid split found, exiting.')

        # Load back the hold-out set
        residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val, \
            = get_data(GEDI, GEDI_val, _GEDI_holdout, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run)
        GEDI_hold_out = _GEDI_holdout
        
        # Save the indices
        if SAVE:
            split_indices = {'train' : GEDI['idx'].values, 'val' : GEDI_val['idx'].values, 'test' : GEDI_hold_out['idx'].values}
            
            if 'ood' not in fname :
                base, middle, end = fname.split('-')
                fname = f"{base}-{middle}_ood-{end}"
            
            with open(join(save_dir, fname), 'wb') as f: pickle.dump(split_indices, f)

    # Otherwise, do the split
    else:

        # Split the data into train/val/test sets #####################################################

        # Randomly subsample the data if there are too many footprints
        num_footprints = GEDI.shape[0]
        max_total_footprints = int(max_train_footprints / (1 - test_holdout))
        if num_footprints > max_total_footprints :
            print(f"Subsampling {num_footprints} footprints to {max_total_footprints} footprints.")
            GEDI = GEDI.sample(max_total_footprints, random_state = 42)
            num_footprints = max_train_footprints
        all_GEDI = GEDI.copy()

        # Repeat until we have a satisfying split
        satisfied, s_id, split_differences = False, 0, []
        while not satisfied and s_id < max_tries :
            satisfied, s_id, GEDI, GEDI_val, GEDI_hold_out, max_diff, rest = get_fold(all_GEDI, model_name, run, False, s_id, val_holdout, test_holdout, stripe_size, width, height, max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std, test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
            if max_diff is not None: split_differences.append(max_diff)

        # If could not find a satisfying split after max_tries attempts, skip this fold
        if not satisfied :
            if len(split_differences) == 0 : raise Exception('No valid splits were found, exiting.')
            print(f'Could not find a satisfying split after {max_tries} attempts. Picking least worst.')
            s_id = np.argmin(split_differences)
            _, _, GEDI, GEDI_val, GEDI_hold_out, _, rest = get_fold(all_GEDI, model_name, run, True, s_id, val_holdout, test_holdout, stripe_size, width, height, max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std, test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
        if GEDI is None or rest is None: raise Exception('No valid split found, exiting.')

        # Parse the results from the split
        residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val = rest

        # Save the indices
        if SAVE:
            split_indices = {'train' : GEDI['idx'].values, 'val' : GEDI_val['idx'].values, 'test' : GEDI_hold_out['idx'].values}
            with open(join(save_dir, fname), 'wb') as f: pickle.dump(split_indices, f)

    return GEDI, GEDI_val, GEDI_hold_out, residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
           gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
           agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val




#######################################################################################################################
# Code execution

if __name__ == '__main__':


    # Initialize everything #######################################################################

    program_start = time()

    # Parse the arguments
    s2_tile, year, arch, ens_models, test_holdout, val_holdout, stripe_size, _path_predictions, path_gedi, path_geometries, path_dem, path_kriging, aux, extra_features, norm_aux, norm_coords, norm_res, coords, pred_vals, matern_nu, num_iterations, pos_loss, lr, max_train_footprints, _x_lengthscale, y_lengthscale, fix_x_y, z_aux_lengthscale, z_pred_lengthscale, eft_lengthscale, outputscale, gaussian_noise, learned_noise, _model_name, patience, min_delta, COMPUTE_VAR, SAVE, SAVE_preds, max_split_diff, max_tries, _seed, composites, ood, agb = parser()
    pred_model_name='_'.join(ens_models)

    # Set the random seeds for reproducibility
    random.seed(_seed), np.random.seed(_seed), torch.manual_seed(_seed), torch.cuda.manual_seed_all(_seed)

    tiles = ['47MRV', '48MTE', '48MUE', '47MRU', '48MTD', '48MUD', '47MRT', '48MTC', '48MUC']

    for s2_tile in tiles :

        print('------------------------------------------------------------------------------')
        print(f'Processing tile {s2_tile}...\n')

        path_kriging = getcwd()
        reference = 'gedi'
        run = None
        path_predictions = join(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/predictions/nico_film/17997535-1_17997535-2_17997535-3', f"{s2_tile}_2021_composite.tif")
        model_name = f'{_model_name}-{s2_tile}'

        # Define paths
        ckpt_path = join(path_kriging, 'checkpoints', model_name)
        if not isdir(ckpt_path) : makedirs(ckpt_path, exist_ok=True)

        tif_path = join(path_kriging, 'predictions', arch, str(year), '_'.join(ens_models))
        x_lengthscale = _x_lengthscale
        #if isfile(join(tif_path, f"{s2_tile}_{stripe_size}_{max_train_footprints}{f'_{_x_lengthscale}' if _x_lengthscale == 'dynamic' else ''}{'_ood' if ood else ''}.tif")) : 
        #    print(f"Tile {s2_tile} already processed, skipping.")
        #    continue

        # Load and pre-process the data ###############################################################
        time_start = time()
        print('Loading data...')

        # Load the AGB predictions for the specified tile
        pred_agb, pred_std, pred_mask, meta, transform, upsampling_shape, nodataval, target_crs = load_dense_preds(s2_path = path_predictions, model_name = model_name, aux = aux) # shape (nrows, ncolumns) = (height, width)

        # If needed, compute extra features
        if extra_features[0] != 'none' : extra_ft = get_extra_features(pred_agb, features = extra_features)
        else: extra_ft = None

        # Load the GEDI footprints within the geometry
        GEDI = gpd.read_file(path_gedi, engine = 'pyogrio')
        GEDI = GEDI.rename(columns={"index": "idx"})
        if GEDI.empty: raise ValueError('No GEDI footprints in the geometry.')
        
        # Reproject GEDI footprints to the local CRS
        GEDI = GEDI.to_crs(target_crs)
        print(f'Number of footprints: {GEDI.shape[0]}')

        # Get the row and column indices of the GEDI footprints
        width, height = pred_agb.shape[1], pred_agb.shape[0]
        print('height, width:', height, width)
        with rs.open(path_predictions) as src:
            def get_idx(geom, src) :
                lon, lat = geom.x, geom.y
                row_index, col_index = src.index(lon, lat)
                return row_index, col_index
            GEDI[['row_idx', 'col_idx']] = GEDI.apply(lambda row: get_idx(row['geometry'], src), axis = 1).apply(pd.Series)

        # Filter the values that are outside of the width/height of the prediction
        print('height, width:', height, width)
        GEDI = GEDI[(GEDI['row_idx'] < height) & (GEDI['row_idx'] >= 0) & (GEDI['col_idx'] < width) & (GEDI['col_idx'] >= 0)]

        # Remove the values where either the prediction or the STD are not defined
        valid_mask = (pred_mask == 0)
        if np.count_nonzero(~valid_mask) > 0 :
            GEDI = GEDI[valid_mask[GEDI['row_idx'], GEDI['col_idx']]]
            assert GEDI.shape[0] > 0, "No GEDI footprints left after filtering."

        # Dynamically calculate the lengthscales based on the furthest nearest neighbor
        if x_lengthscale == "dynamic" :
            print("Calculating dynamic lengthscales based on the maximum distance to footprint.")
            x_lengthscale, y_lengthscale = get_furthest_neighbor(GEDI, height, width, second_max = True, unit = 10, norm_coords = norm_coords)
            print(f'New lengthscales: x={x_lengthscale}, y={y_lengthscale}')

        # If there are multiple footprints in the same pixel, take the median
        print(f'\nNumber of footprints before groupby: {GEDI.shape[0]}')
        GEDI = (GEDI.groupby(['row_idx', 'col_idx'], as_index=False).agg({'agbd': 'median', 'idx': list, **{col: 'first' for col in GEDI.select_dtypes(include='number').columns if col not in ['row_idx', 'col_idx', 'agbd', 'idx']}}))
        print(f'Number of footprints after groupby: {GEDI.shape[0]}\n')

        # Calculate the overall performance metrics, pre-kriging ######################################
        og_Y, og_X = GEDI['row_idx'].values.astype(int), GEDI['col_idx'].values.astype(int)
        og_indices = GEDI['idx'].values
        og_gedi_agb = GEDI['agbd'].values
        predictions = pred_agb[og_Y, og_X]

        # 1. Overall RMSE
        residuals = predictions - og_gedi_agb
        rmse = np.sqrt(np.nanmean(np.pow(residuals, 2)))
        print(f'\nOverall RMSE: {rmse}\n')
        # 2. Binned RMSE
        bins = np.arange(0, 501, 50)
        lb, ub = bins[:-1], bins[1:]
        for l, u in zip(lb, ub) :
            mask = (og_gedi_agb >= l) & (og_gedi_agb < u)
            binned_residuals = residuals[mask]
            num = np.sum(mask)
            if num > 0 : binned_rmse = np.sqrt(np.nanmean(np.pow(binned_residuals, 2)))
            else: binned_rmse = np.nan
        
        # Number of footprints in the tile
        num_footprints = GEDI.shape[0]

        # Get the train/val/test sets #################################################################
        
        save_dir = join(path_kriging, 'predictions', 'splits')
        if not isdir(save_dir) : makedirs(save_dir, exist_ok=True)
        _id = model_name.split('-')[-1]
        fname = f"splits-{_id}_{test_holdout:.1f}_{val_holdout:.1f}_{max_train_footprints}_{max_split_diff:.1f}_{stripe_size}{'_ood' if ood else ''}-{reference}.pkl" 
        SPLITS_EXIST = isfile(join(save_dir, fname))
        ood = ood and (GEDI.shape[0] > 500)

        if ood :
            if not SPLITS_EXIST :
                # Check if the base split file exists
                base_fname = f"splits-{_id}_{test_holdout:.1f}_{val_holdout:.1f}_{max_train_footprints}_{max_split_diff:.1f}_{stripe_size}-{reference}.pkl"
                if isfile(join(save_dir, base_fname)) : 
                    SPLITS_EXIST = True
                    perform_ood = True
                    fname = base_fname 
                else:
                    raise Exception('OOD splits requested but base split file does not exist. Please run without OOD first to generate the base splits.')
            else: 
                perform_ood = False
        else: 
            perform_ood = False

        GEDI, GEDI_val, GEDI_hold_out, residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, \
        og_X_train, og_Y_train, og_X_val, og_Y_val = get_train_val_test_split(GEDI, model_name, pred_agb, pred_std, extra_ft, \
            aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run, test_holdout, \
            val_holdout, max_train_footprints, max_split_diff, stripe_size, max_tries, test_indices = None, save_dir = save_dir, \
            SAVE = SAVE, SPLITS_EXIST = SPLITS_EXIST, perform_ood = perform_ood, fname = fname, ood_extra = (og_indices, og_gedi_agb, predictions))

        # Pre-process the residuals ####################################################################

        # If regressing the AGB values instead of the residuals, set the residuals to the AGB values
        if agb : residuals, residuals_val, residuals_test = gedi_agb, gedi_agb_val, gedi_agb_test

        # Normalize the residuals if necessary
        if norm_res : 
            res_mu, res_std = np.mean(residuals), np.std(residuals)
            norm_values['res'] = {'mean': res_mu, 'std': res_std}
            print('Residuals normalizations values: mean = %.4f, std = %.4f' % (res_mu, res_std))
            if res_std == 0 : res_std = 1 # Avoid division by zero
            residuals = (residuals - res_mu) / res_std
            residuals_val = (residuals_val - res_mu) / res_std
            residuals_test = (residuals_test - res_mu) / res_std
        
        # Get the training and validation data
        train_x, train_y, val_x, val_y, norm_values = get_train_val_data(X, Y, X_val, Y_val, predictions, predictions_val, std, std_val, eft, eft_val, aux, norm_aux, norm_coords, coords, pred_vals, norm_values, residuals, residuals_val)

        # Define the likelihood
        # homoskedastic noise model (i.e. all inputs have the same observational noise)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()


        # Define the model
        if SAVE_preds: torch.save({'train_x': train_x, 'train_y': train_y, 'likelihood_state': likelihood.state_dict(), 'ndims': train_x.shape[1], 'norm_values': norm_values}, join(ckpt_path, f'{model_name}.pt'))
        

        model = ExactGPModel(train_x, train_y, likelihood, ndims = train_x.shape[1], matern_nu = matern_nu)
        model.likelihood.noise = gaussian_noise
        ###########################################################################################

        # Initialize its parameters
        lengthscales = []
        if coords: lengthscales.extend([x_lengthscale, y_lengthscale])
        if aux != 'none' : lengthscales.append(z_aux_lengthscale)
        if pred_vals : lengthscales.append(z_pred_lengthscale)
        if extra_features[0] != 'none' :
            eft_lengthscales = [deepcopy(eft_lengthscale) for _ in range(len(extra_features))]
            lengthscales.extend(eft_lengthscales)
        model.covar_module.base_kernel.lengthscale = [[lengthscales]]
        model.covar_module.outputscale = outputscale

        # Train the model
        time_start = time()
        print('    Training the model...')

        model.cuda().to(torch.float32)
        likelihood.cuda()

        # Training loop
        model, best_model_state, optimizer, mll = train_with_retries(model, likelihood, train_x, train_y, val_x, val_y, run, num_iterations, patience, min_delta, pos_loss, lr, aux, coords, pred_vals, fix_x_y)
        train_x = train_x.cpu().detach()
        train_y = train_y.cpu().detach()
        val_x = val_x.cpu().detach()
        val_y = val_y.cpu().detach()
        del(train_x, train_y, val_x, val_y)
        torch.cuda.empty_cache()
        gc.collect()

        # Restore best model for evaluation
        if best_model_state:
            model.load_state_dict(best_model_state)
            if SAVE_preds: torch.save(model.state_dict(), join(ckpt_path, f'{model_name}.ckpt'))
        model.eval()
        likelihood.eval()

        print(f'    Done! In {time() - time_start} seconds.')


        # Apply Kriging to the entire tile ############################################################################

        # Now, get the predictions on the entire tile
        start_time = time()
        print('    Predicting...')
        tile_x = get_tile_data(pred_agb, dem, pred_std, extra_ft, norm_values, coords, norm_coords, aux, norm_aux, pred_vals, extra_features)
        pred_y = np.zeros_like(tile_x[:, 0], dtype='float')
        if COMPUTE_VAR : pred_y_var = np.zeros_like(tile_x[:, 0], dtype='float')
        model = model.to(torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            N = 500 # 1_000
            for i in range(0, pred_y.size, N):
                tile_x_crop = tile_x[i : min(i + N, pred_y.size)]
                local_pred = np.zeros(tile_x_crop.shape[0], dtype='float')
                if COMPUTE_VAR : local_var = np.zeros(tile_x_crop.shape[0], dtype='float')
                _valid_mask = ~torch.any(torch.isnan(tile_x_crop), dim=1)
                if not torch.any(_valid_mask) : continue
                tile_x_crop = tile_x_crop[_valid_mask].cuda()
                observed_pred = likelihood(model(tile_x_crop))
                local_pred[_valid_mask.cpu().numpy()] = observed_pred.mean.cpu().numpy()
                pred_y[i : min(i + N, pred_y.size)] = local_pred
                if COMPUTE_VAR : 
                    local_var[_valid_mask.cpu().numpy()] = observed_pred.variance.cpu().numpy()
                    pred_y_var[i : min(i + N, pred_y.size)] = local_var

        tile_x_crop = tile_x_crop.cpu().detach()
        tile_x = tile_x.cpu().detach()
        del(tile_x_crop, tile_x)
        torch.cuda.empty_cache()
        gc.collect()
        print(f'    Done! In {time() - start_time} seconds.')

        # Calculate the residuals on the entire tile
        residuals = pred_y.reshape(pred_agb.shape)
        if norm_res : residuals = residuals * res_std + res_mu
        del(pred_y)
        if COMPUTE_VAR : 
            residuals_var = pred_y_var.reshape(pred_agb.shape)
            if norm_res : residuals_var *= np.pow(res_std, 2)
            residuals_var = np.clip(residuals_var, 0, None) # Ensure non-negativity
            del(pred_y_var)

        # Apply the residuals to the predictions
        if agb : corrected = np.clip(residuals, 0, None)
        else: corrected = np.clip(pred_agb - residuals, 0, None)
        if not SAVE_preds : del(residuals)

        # Where the input data was undefined, mask the output
        print('corrected shape:', corrected.shape)
        print('valid_mask shape:', valid_mask.shape)
        corrected[~valid_mask] = np.nan
        if COMPUTE_VAR : 
            residuals_var[~valid_mask] = np.nan
        del valid_mask

        # Now calculate the new RMSE on the test set
        predictions_test_corrected = corrected[Y_test, X_test]
        residuals_test_corrected = predictions_test_corrected - gedi_agb_test
        rmse_corrected = np.sqrt(np.nanmean(np.pow(residuals_test_corrected, 2)))
        print(f'    RMSE test corrected: {rmse_corrected}')
        print()
        del(predictions_test_corrected, residuals_test_corrected)

        # Cleanup
        model, likelihood = model.to('cpu'), likelihood.to('cpu')
        del(model, optimizer, likelihood, mll)
        torch.cuda.empty_cache()
        gc.collect()

        if SAVE :
            # Path to the output directory
            tif_path = join(path_kriging, 'predictions', arch, str(year), '_'.join(ens_models))
            if not isdir(tif_path) : makedirs(tif_path, exist_ok=True)
            # Save the pre-kriging predictions and GTs at the footprints' locations
            with open(join(tif_path, f"results-{model_name}-{reference}_{stripe_size}{'_ood' if ood else ''}.pkl"), 'wb') as f:
                pickle.dump({'pre': pred_agb[og_Y, og_X], 'ref': og_gedi_agb, 'idx': og_indices, 'post': corrected[og_Y, og_X]}, f)
        del(pred_std)
        if not agb: del(pred_agb)

        ###################################################################################################################
        print('Calculating overall performance...')

        # Calculate the overall performance
        # 1. Overall RMSE
        res = corrected[og_Y, og_X] - og_gedi_agb
        rmse = np.sqrt(np.nanmean(np.pow(res, 2)))
        print(f'\nOverall RMSE (post-kriging): {rmse}\n')
        # 2. Binned RMSE
        bins = np.arange(0, 501, 50)
        lb, ub = bins[:-1], bins[1:]
        for l, u in zip(lb, ub) :
            mask = (og_gedi_agb >= l) & (og_gedi_agb < u)
            binned_residuals = res[mask]
            if np.sum(mask) > 0 :
                binned_rmse = np.sqrt(np.nanmean(np.pow(binned_residuals, 2)))
            else: binned_rmse = np.nan
        
        # Calculate the test set performance
        print('Calculating test set performance...')
        # 1. Overall RMSE
        res_test = corrected[Y_test, X_test] - gedi_agb_test
        rmse_test = np.sqrt(np.nanmean(np.pow(res_test, 2)))
        print(f'\nTest set RMSE (post-kriging): {rmse_test}\n')

        ###################################################################################################################
        # Save the dense predictions
        if SAVE_preds :
            start_time = time()
            print('Saving the corrected prediction...')
            count = 3 if COMPUTE_VAR else 2
            meta.update(count = count, dtype = 'float32', nodata = np.nan)
            if reference == 'field' : meta.update({"height": height, "width": width, "transform": transform})
            if 'cci' in model_name : meta.update({"height": height, "width": width})
            with rs.open(join(tif_path, f"{s2_tile}_{stripe_size}_{max_train_footprints}{f'_{_x_lengthscale}' if _x_lengthscale == 'dynamic' else ''}{'_ood' if ood else ''}.tif"), 'w', **meta) as dst:
                # Write the corrected AGB predictions
                dst.write(corrected, 1)
                dst.set_band_description(1, 'AGB')
                # Write the residuals
                if agb : residuals = pred_agb - residuals
                dst.write(residuals, 2)
                dst.set_band_description(2, 'Residuals')
                # Write the STD
                if COMPUTE_VAR :
                    dst.write(np.sqrt(residuals_var), 3)
                    dst.set_band_description(3, 'STD')
            print(f'Done! In {time() - start_time} seconds.')

        # Measure total time taken
        total_time = time() - program_start
        print(f'Total time taken: {total_time} seconds.')
