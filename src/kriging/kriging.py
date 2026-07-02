"""

This scripts perform Kriging, i.e. Gaussian Process regression, to correct the AGB predictions of the model.
The script takes as input the AGB predictions of the model, the GEDI footprints, and the DEM data, and outputs
the corrected AGB predictions. The script performs k-fold cross validation, and saves the corrected predictions
to a file.

"""

#######################################################################################################################
# Imports

import numpy as np
import rasterio as rs
import pandas as pd
from rasterio.crs import CRS
import geopandas as gpd
import torch
import gpytorch
from os.path import join
from scipy.ndimage import uniform_filter, sobel, laplace, maximum_filter, minimum_filter
from time import time
import argparse
from os.path import join, isdir, isfile
from os import makedirs
from wandb_utils import init_run, wandb_enabled
import gc
import pickle
from itertools import combinations
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from linear_operator.utils.warnings import NumericalWarning
from sklearn.model_selection import train_test_split
import os
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
import warnings
from copy import deepcopy
import shutil

from kriging.core import *  # shared GP models + helpers (see kriging/core.py)

torch.backends.cuda.matmul.allow_tf32 = False
os.environ['WANDB_INIT_TIMEOUT'] = '300'
os.environ["WANDB__SERVICE_WAIT"] = "300"

#######################################################################################################################
# Helper functions 









def load_dense_preds(s2_path, aux = None) :
    """
    This function loads the dense predictions of the model for the Sentinel-2 tile.

    Args:
    - s2_path: string, path to the best model's prediction for the S2 tile.
    - aux: string, auxiliary variable to use. If 'STD', the function will load the STD of the ensemble.

    Returns:
    - pred_agb: 2d array, dense predictions of the model.
    - pred_std: 2d array, standard deviation of the predictions (if aux is 'STD').
    - pred_mask: 2d array, mask of the tile (if available).
    - meta: dict, metadata of the Sentinel-2 tile.
    - _transform: affine.Affine, transform of the Sentinel-2 tile.
    - upsampling_shape: tuple of ints, shape of the Sentinel-2 tile.
    - nodataval: int, value to use for the nodata pixels.
    """

    # Get the AGB prediction and associated STD
    with rs.open(s2_path) as src:

        pred_agb = src.read(1)
        pred_std = src.read(2) if aux == 'STD' else None

        # Get the metadata of the file (to later save the Kriging results)
        meta = src.meta
        _transform = src.transform
        upsampling_shape = pred_agb.shape
        nodataval = src.nodata
    
    # Get the mask
    if nodataval is not None:
        if aux == 'STD' : pred_mask = (equals(pred_agb, nodataval) | equals(pred_std, nodataval)).astype(np.uint8)
        else: pred_mask = (equals(pred_agb, nodataval)).astype(np.uint8)
    else: pred_mask = np.zeros(upsampling_shape, dtype = np.uint8)

    return pred_agb, pred_std, pred_mask, meta, _transform, upsampling_shape, nodataval






NODATAVALS = {'S2' : 0, 'CH': 255, 'ALOS': 0, 'LC': 255, 'DEM': -9999, 'LC': 255}







def geographical_train_test_split(GEDI, order, val_size = 0.2, test_size = 0.3, stripe_size = 100, max_train_footprints = 25000, random_state = 42) :
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

    """ Plot the checkerboard pattern
    # Create a 2d array to visualize the split
    split_map = np.zeros((len(horizontal_stripes) * stripe_size, len(vertical_stripes) * stripe_size), dtype = np.uint8)
    for split in ['train', 'val', 'test'] :
        for patch in metrics[split]['stripes'] :
            row, col = patch
            rows, cols = list(range(row, row + stripe_size)), list(range(col, col + stripe_size))
            if split == 'train' : split_map[np.ix_(rows, cols)] = 1
            elif split == 'val' : split_map[np.ix_(rows, cols)] = 2
            elif split == 'test' : split_map[np.ix_(rows, cols)] = 3
    # Create a colormap for the split map
    cmap = ListedColormap(['white', '#5B9BD5', '#F79646', '#70AD47'])
    # Plot the split map
    plt.figure(figsize = (10,10))
    plt.imshow(split_map, cmap = cmap, interpolation = 'nearest')
    plt.title('Geographical split of the GEDI data')
    plt.xlabel('Column index')
    plt.ylabel('Row index')
    plt.xticks(ticks = np.arange(0, split_map.shape[1], stripe_size), labels = np.arange(min_col, max_col + 1, stripe_size))
    plt.yticks(ticks = np.arange(0, split_map.shape[0], stripe_size), labels = np.arange(min_row, max_row + 1, stripe_size))
    plt.colorbar(ticks = [0, 1, 2, 3], label = 'Split', orientation = 'horizontal')
    plt.clim(-0.5, 3.5)
    plt.savefig('checkerboard.png')
    """

    # Return the data
    print(f'    Number of footprints: {GEDI_train.shape[0]} (train) | {GEDI_val.shape[0]} (validation) | {GEDI_test.shape[0]} (test)')
    return GEDI_train, GEDI_val, GEDI_test


def get_data(GEDI, GEDI_val, GEDI_hold_out, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run) :
    """
    This function extracts necessary data from the splits.

    Args:
    - GEDI: geopandas dataframe, training set.
    - GEDI_val: geopandas dataframe, validation set.
    - GEDI_hold_out: geopandas dataframe, test set.

    Returns: the residuals, indices, AGB values, predictions, standard errors, DEM data, and standard deviation data.
    """

    # Get the row and column indices
    Y, X = GEDI['row_idx'].values, GEDI['col_idx'].values
    Y_val, X_val = GEDI_val['row_idx'].values, GEDI_val['col_idx'].values
    Y_test, X_test = GEDI_hold_out['row_idx'].values, GEDI_hold_out['col_idx'].values

    # Get the AGB values
    gedi_agb, gedi_agb_val, gedi_agb_test = GEDI['agbd'].values, GEDI_val['agbd'].values, GEDI_hold_out['agbd'].values

    # Get the predictions at those points
    predictions = pred_agb[Y, X]
    predictions_val = pred_agb[Y_val, X_val]
    predictions_test = pred_agb[Y_test, X_test]

    # Calculate the residuals
    residuals =  predictions - gedi_agb
    residuals_val = predictions_val - gedi_agb_val
    residuals_test = predictions_test - gedi_agb_test

    # Calculate the RMSE
    rmse_train = np.sqrt(np.mean(np.pow(residuals, 2)))
    rmse_val = np.sqrt(np.mean(np.pow(residuals_val, 2)))
    rmse_test = np.sqrt(np.mean(np.pow(residuals_test, 2)))
    print(f'    RMSEs: {rmse_train:.2f} (train) | {rmse_val:.2f} (validation) | {rmse_test:.2f} (test)')

    run.log({f'rmse_train_pre': rmse_train, 
            f'rmse_val_pre': rmse_val, 
            f'rmse_test_pre': rmse_test}, step = None)
    
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
                norm_values['STD'] = {'min': std_min, 'max': std_max}
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

    return residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, \
        norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val


def get_fold(all_GEDI, run, satisfied, s_id, val_holdout, test_holdout, stripe_size, max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_agb, pred_std, aux, extra_ft, width, height, test_indices = None, order = ['train', 'test', 'train', 'val'], norm_coords = False, norm_aux = None, max_split_diff = 15) :
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
    GEDI, GEDI_val, GEDI_hold_out = geographical_train_test_split(all_GEDI, order = order, val_size = val_holdout, \
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
        run.log({f'rmse_train_pre': rmse_train, 
                f'rmse_val_pre': rmse_val, 
                f'rmse_test_pre': rmse_test}, step = None)
        
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






class TrainingFailure(Exception):
    def __init__(self, message, partial_state=None):
        super().__init__(message)
        self.partial_state = partial_state

def train_loop(model, likelihood, train_x, train_y, val_x, val_y, run, num_iterations, patience, min_delta, pos_loss, lr, aux, coords, pred_vals, fix_x_y = False):
    """
    This function performs the training loop for the GP model, with early stopping based on validation loss.

    Args:
    - model: ExactGPModel, the GP model to train.
    - likelihood: gpytorch.likelihoods.Likelihood, the likelihood function.
    - train_x: torch.Tensor, training inputs.
    - train_y: torch.Tensor, training targets.
    - val_x: torch.Tensor, validation inputs.
    - val_y: torch.Tensor, validation targets.
    - run: wandb run object, to log the training progress.
    - num_iterations: int, number of training iterations.
    - patience: int, number of iterations to wait for improvement before stopping.
    - min_delta: float, minimum change in validation loss to consider as an improvement.
    - pos_loss: bool, whether to stop training if the validation loss is negative.
    - lr: float, learning rate for the optimizer.
    - _likelihood: str, type of likelihood to use ('gaussian' or 'fixed').
    - aux: str, auxiliary variable to use ('none', 'DEM', or 'STD').
    - coords: bool, whether to use spatial coordinates as input features.

    Returns:
    - model: ExactGPModel, the trained GP model.
    - best_model_state: dict, the state of the best model found during training.
    - optimizer: torch.optim.Optimizer, the optimizer used for training.
    - mll: gpytorch.mlls.ExactMarginalLogLikelihood, the marginal log likelihood used for training.
    """

    try:

        model.train(), likelihood.train()
        if coords and fix_x_y : model.covar_module.base_kernel.raw_lengthscale.register_hook(zero_first_two) # don't optimize the lengthscales of the coordinates
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        best_loss, counter, best_model_state = np.inf, 0, None
        for i in range(num_iterations + 1):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)

            # Optimize the model
            loss.backward()
            optimizer.step()

            # Early stopping
            if i % 10 == 0:
                model.eval(), likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    val_loss = -mll(model(val_x), val_y)
                    if pos_loss and val_loss.item() < 0 : # Stop training if the validation loss is negative
                        print(f"    Early stopping at iteration {i} (negative loss)")
                        break
                    if val_loss.item() < best_loss - min_delta:
                        best_loss = val_loss.item()
                        counter = 0
                        best_model_state = model.state_dict()
                    else: counter += 10
                    if counter >= patience: # Stop training if patience is exceeded
                        print(f"    Early stopping at iteration {i} (patience exceeded)")
                        break

                print('    Iter %d/%d - Train Loss: %.3f   Val Loss: %.3f   lengthscale: %s   outputscale: %.3f   noise: %.3f' % (
                    i, num_iterations, loss.item(), val_loss.item(),
                    model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
                    model.covar_module.outputscale.item(),
                    model.likelihood.noise.item()
                ))

                run.log({f'train_loss': loss.item(),
                        f'val_loss': val_loss.item(),
                        f'x_lengthscale': model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0][0] if coords else None,
                        f'y_lengthscale': model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0][1] if coords else None,
                        f'z_aux_lengthscale': model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0][2 if coords else 0] if aux != 'none' else None,
                        f'z_pred_lengthscale': model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0][3 if coords else 1] if aux != 'none' else None,
                        f'outputscale': model.covar_module.outputscale.item(),
                        f'noise': model.likelihood.noise.item()})



                model.train(), likelihood.train()
        
        return model, best_model_state, optimizer, mll

    except Exception as e:
        partial = {"model": model, "best_model_state": best_model_state, "optimizer": optimizer, "mll": mll}
        raise TrainingFailure(str(e), partial_state=partial)


def train_with_retries(model, likelihood, train_x, train_y, val_x, val_y, run, num_iterations, patience, min_delta, pos_loss, lr, aux, coords, pred_vals, fix_x_y) :
    """
    This function attempts to train the GP model with the given learning rate, and retries with lower learning rates if training fails.

    Args:
    - model: ExactGPModel, the GP model to train.
    - likelihood: gpytorch.likelihoods.Likelihood, the likelihood function.
    - (train_x, train_y, val_x, val_y): torch.Tensors, training and validation data.
    - run: wandb run object, to log the training progress.
    - num_iterations: int, number of training iterations.
    - patience: int, number of iterations to wait for improvement before stopping.
    - min_delta: float, minimum change in validation loss to consider as an improvement.
    - pos_loss: bool, whether to stop training if the validation loss is negative.
    - lr: float, initial learning rate for the optimizer.
    - _likelihood: str, type of likelihood to use ('gaussian' or 'fixed').
    - aux: str, auxiliary variable to use ('none', 'DEM', or 'STD').
    - coords: bool, whether to use spatial coordinates as input features.
    - pred_vals: np.array, predicted values from the previous model.
    - fix_x_y: bool, whether to fix the lengthscales of the spatial coordinates.

    Returns:
    - model: ExactGPModel, the trained GP model.
    - best_model_state: dict, the state of the best model found during training.
    - optimizer: torch.optim.Optimizer, the optimizer used for training.
    - mll: gpytorch.mlls.ExactMarginalLogLikelihood, the marginal log likelihood used for training.
    """
    
    learning_rates = [lr, lr/10, lr/100, lr/1000] # List of learning rates to try
    og_model = deepcopy(model)
    og_likelihood = deepcopy(likelihood)

    print('Checking trainable parameters:')
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    for _lr in learning_rates:
        if _lr != lr : print(f"    Changing learning rate to {_lr}.")
        try:
            with warnings.catch_warnings() :
                warnings.filterwarnings("error", category=NumericalWarning) # treat the NumericalWarning as an error
                model, best_model_state, optimizer, mll = train_loop(og_model, og_likelihood, train_x, train_y, val_x, val_y, run, num_iterations, patience, min_delta, pos_loss, _lr, aux, coords, pred_vals, fix_x_y)
            if _lr != lr : run.config.update({'lr': _lr}, allow_val_change=True) # Log the new learning rate used
            return model, best_model_state, optimizer, mll
        except TrainingFailure as e:
            print(f"Training failed at lr={_lr} with error: {e}")
            if e.partial_state is not None: least_worse = e.partial_state
    if least_worse is not None:
        print("Training failed for all learning rates. Taking the last valid model state.")
        model = least_worse["model"]
        best_model_state = least_worse["best_model_state"]
        optimizer = least_worse["optimizer"]
        mll = least_worse["mll"]
        return model, best_model_state, optimizer, mll
    else: raise RuntimeError("Training failed for all learning rates.") 














def get_extra_features(pred_agb, features) :
    """
    This function computes block-wise features (median or mean) of the predicted AGB array.

    Args:
    - pred_agb: numpy array, the predicted AGB values.
    - features: list of str, the features to compute.

    Returns:
    - expanded_feature: numpy array, the block-wise feature expanded back to the original size.
    """

    print("Computing extra features...")
    assert np.count_nonzero(np.isnan(pred_agb)) == 0, "pred_agb contains NaN values."

    _features = deepcopy(features)

    extra_features = []
    for feat in _features :
        if 'mean' in feat : # mean
            patch_size = int(feat.split('_')[-1]) # extract context window
            mean = uniform_filter(pred_agb, size = patch_size, mode = 'reflect')
            extra_features.append(mean)
            if f'std_{patch_size}' in _features : # standard deviation
                mean_sq = uniform_filter(pred_agb ** 2, size = patch_size, mode = 'reflect')
                std = np.sqrt(np.maximum(mean_sq - (mean ** 2), 0))
                extra_features.append(std)
                _features.remove(f'std_{patch_size}') # don't recompute it
                if f'cv_{patch_size}' in _features : # coefficient of variation
                    cv = std / (mean + 1e-6)
                    extra_features.append(cv)
                    _features.remove(f'cv_{patch_size}') # don't recompute it
        elif feat == 'sobel' : # sobel
            sx = sobel(pred_agb, axis=0, mode='reflect')
            sy = sobel(pred_agb, axis=1, mode='reflect')
            extra_features.append(np.hypot(sx, sy)) # Gradient magnitude
        elif feat == 'laplace' : # laplace
            extra_features.append(laplace(pred_agb, mode='reflect'))
        elif 'lr' in feat : # local range
            patch_size = int(feat.split('_')[-1]) # extract context window
            local_max = maximum_filter(pred_agb, size = patch_size, mode = 'reflect')
            local_min = minimum_filter(pred_agb, size = patch_size, mode = 'reflect')
            local_range = local_max - local_min
            extra_features.append(local_range)
        else: 
            print(f"Unknown feature type: {feat}")
            continue
        
    print('done!')

    return np.stack(extra_features, axis = -1)












def check_config_match(ref_model, _id, run) :
    if not wandb_enabled() :  # config-consistency check queries W&B; skipped when running without it
        return
    import wandb
    """
    This function asserts that the configuration of the reference model matches the current model's configuration,
    up to some arguments.

    Args:
    - ref_model: str, name of the reference model (format JOBID)
    - _id: str, ID of the tile.
    - run: current run's wandb run object.
    """

    # Define the arguments to check for identical values
    args_to_check = ["seed", "ens_models", "composites", "test_holdout", "val_holdout", "mem_max_footprints", \
                     "max_split_diff", "max_tries", "stripe_size", "subsample", "ood", "regional", "aux", "extra_features", \
                     "pred_vals", "norm_aux", "norm_res", "coords", "norm_coords", \
                     "num_iterations", "pos_loss", "agb", "x_lengthscale", \
                     "y_lengthscale", "fix_x_y", "z_aux_lengthscale", "z_pred_lengthscale", "outputscale", "gaussian_noise", \
                     "learned_noise", "matern_nu"]
    
    default_values = {"mem_max_footprints": 25000}

    # Get the current run's config
    curr_config = run.config

    # Get the reference run
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/kriging", {"display_name": f'{ref_model}-{_id}'})
    run = runs[len(runs) - 1]
    ref_config = run.config

    # Check for mismatches
    for arg in args_to_check :
        if curr_config[arg] != ref_config.get(arg, default_values.get(arg, None)) :
            raise ValueError(f"Configuration mismatch for argument '{arg}': reference model has value '{ref_config[arg]}', current model has value '{curr_config[arg]}'.")


def parser(argv=None):
    """ 
    Returns an `ArgumentParser()` object containing the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2_tile', type = str, required = True, help = 'S2 tile.')
    parser.add_argument('--year', type = int, required = True, help = 'Year of the prediction.')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--ens_models', type = str, nargs = '+', required = True, help = 'Models for ensemble STD.')
    parser.add_argument('--test_holdout', type = float, required = True, help = 'Percentage of footprints to hold out for test.')
    parser.add_argument('--val_holdout', type = float, required = True, help = 'Percentage of footprints to hold out for validation.')
    parser.add_argument('--stripe_size', type = int, required = True, help = 'Size (in pixels) of a stripe.')
    parser.add_argument('--path_predictions', type = str, required = True, help = 'Directory with the predictions.')
    parser.add_argument('--path_gedi', type = str, required = True, help = 'Directory with the GEDI footprints.')
    parser.add_argument('--path_geometries', type = str, required = True, help = 'Directory with the S2 tiles geometries.')
    parser.add_argument('--path_dem', type = str, required = True, help = 'Directory with the DEM.')
    parser.add_argument('--path_kriging', type = str, required = True, help = 'Directory for Kriging.')
    parser.add_argument('--aux', type = str, required = True, help = 'Which auxiliary variable to use.')
    parser.add_argument('--extra_features', type = str, nargs = '+', required = True, help = 'Additional features to compute, if any.')
    parser.add_argument('--norm_aux', type = str, required = True, help = 'Whether to normalize the auxiliary data.')
    parser.add_argument('--norm_coords', type = str2bool, required = True, help = 'Whether to normalize the coordinates.')
    parser.add_argument('--norm_res', type = str2bool, required = True, help = 'Whether to normalize the residuals.')
    parser.add_argument('--coords', type = str2bool, required = True, help = 'Whether to include the x/y coordinates.')
    parser.add_argument('--pred_vals', type = str2bool, required = True, help = 'Whether to include the predicted values.')
    parser.add_argument('--matern_nu', type = float, required = True, help = 'What nu parameter to use for the Matern kernel.')
    parser.add_argument('--num_iterations', type = int, required = True, help = 'Number of iterations.')
    parser.add_argument('--pos_loss', type = str2bool, required = True, help = 'Whether to stop if the loss becomes negative.')
    parser.add_argument('--lr', type = float, required = True, help = 'Learning rate.')
    parser.add_argument('--max_train_footprints', type = int, required = True, help = 'Max # of footprints to train on.')
    parser.add_argument('--mem_max_footprints', type = int, required = True, help = 'Max # of footprints to use in memory.')
    parser.add_argument('--ref_model', type = str, required = True, help = 'Reference model to use when using fewer footprints for training.')
    parser.add_argument('--x_lengthscale', type = float_or_str, required = True, help = 'Lengthscale in the x direction.')
    parser.add_argument('--y_lengthscale', type = float_or_str, required = True, help = 'Lengthscale in the y direction.')
    parser.add_argument('--fix_x_y', type = str2bool, required = True, help = 'Whether to fix the x/y lengthscales to the provided values.')
    parser.add_argument('--z_aux_lengthscale', type = float, required = True, help = 'Lengthscale in the z direction, for aux.')
    parser.add_argument('--z_pred_lengthscale', type = float, required = True, help = 'Lengthscale in the z direction, for preds.')
    parser.add_argument('--eft_lengthscale', type = float, required = True, help = 'Lengthscale in the z direction, for extra features.')
    parser.add_argument('--outputscale', type = float, required = True, help = 'Output scale.')
    parser.add_argument('--gaussian_noise', type = float, required = True, help = 'Gaussian noise.')
    parser.add_argument('--learned_noise', type = float, required = True, help = 'Learned noise.')
    parser.add_argument('--model_name', type = str, required = True, help = 'Name of the model.')
    parser.add_argument('--patience', type = float, required = True, help = 'Patience for early stopping.')
    parser.add_argument('--min_delta', type = float, required = True, help = 'Minimum change for early stopping.')
    parser.add_argument('--COMPUTE_VAR', type = str2bool, required = True, help = 'Compute the variance of the residuals.')
    parser.add_argument('--SAVE', type = str2bool, required = True, help = 'Save the metrics.')
    parser.add_argument('--SAVE_preds', type = str2bool, required = True, help = 'Save the corrected predictions.')
    parser.add_argument('--max_split_diff', type = float, required = True, help = 'Max difference in pct tolerated between splits.')
    parser.add_argument('--max_tries', type = int, required = True, help = 'Max number of tries for finding a valid fold.')
    parser.add_argument('--seed', type = int, default = 10, help = 'Random seed.')
    parser.add_argument('--composites', type = str2bool, required = True, help = 'Whether we are loading composites-derived predictions.')
    parser.add_argument('--ood', type = str2bool, required = True, help = 'Whether to remove OOD samples.')
    parser.add_argument('--agb', type = str2bool, required = True, help = 'Whether to use AGB values instead of residuals.')
    parser.add_argument('--offline', type = str2bool, default = False, help = 'Run wandb in offline mode (no network sync) to avoid hangs on clusters without internet.')
    args = parser.parse_args(argv)

    # Run wandb offline if requested. This must be set before wandb.init() is called, so that
    # initialization and logging write to a local ./wandb/ dir instead of hanging on the network.
    if args.offline : os.environ["WANDB_MODE"] = "offline"

    # Determine how to process the input features
    if args.norm_aux == 'false' : args.norm_aux = False
    elif args.norm_aux == 'min_max' : args.norm_aux = 'min_max'
    else: raise ValueError(f"norm_aux must be either 'false' or 'min_max', got {args.norm_aux}.")
    
    return args.s2_tile, args.year, args.arch, args.ens_models, args.test_holdout, args.val_holdout, args.stripe_size, args.path_predictions, args.path_gedi, args.path_geometries, args.path_dem, args.path_kriging, args.aux, args.extra_features, args.norm_aux, args.norm_coords, args.norm_res, args.coords, args.pred_vals, args.matern_nu, args.num_iterations, args.pos_loss, args.lr, args.max_train_footprints, args.mem_max_footprints, args.x_lengthscale, args.y_lengthscale, args.fix_x_y, args.z_aux_lengthscale, args.z_pred_lengthscale, args.eft_lengthscale, args.outputscale, args.gaussian_noise, args.learned_noise, args.model_name, args.patience, args.min_delta, args.COMPUTE_VAR, args.SAVE, args.SAVE_preds, args.max_split_diff, args.max_tries, args.seed, args.composites, args.ood, args.agb, args.ref_model


#######################################################################################################################
# Code execution

def main(argv=None):


    # Initialize everything #######################################################################

    program_start = time()

    # Parse the arguments
    s2_tile, year, arch, ens_models, test_holdout, val_holdout, stripe_size, path_predictions, path_gedi, path_geometries, path_dem, path_kriging, aux, extra_features, norm_aux, norm_coords, norm_res, coords, pred_vals, matern_nu, num_iterations, pos_loss, lr, max_train_footprints, mem_max_footprints, x_lengthscale, y_lengthscale, fix_x_y, z_aux_lengthscale, z_pred_lengthscale, eft_lengthscale, outputscale, gaussian_noise, learned_noise, model_name, patience, min_delta, COMPUTE_VAR, SAVE, SAVE_preds, max_split_diff, max_tries, _seed, composites, ood, agb, ref_model = parser(argv)
    pred_model_name='_'.join(ens_models)

    # Set up Weights & Biases logger
    if 'local' in model_name :
        run = init_run(project = 'kriging')
        if run.name :  # a real W&B run assigned a random name
            model_name = model_name.replace('local', run.name)
            run.name = model_name
    else:
        run = init_run(project = 'kriging', name = model_name)
    # Log the parameters
    run.config.update({'s2_tile': s2_tile, 'year': year, 'pred_model_name': pred_model_name,
                    'ens_models': '_'.join(ens_models), 'val_holdout': val_holdout, 'test_holdout': test_holdout, 
                    'matern_nu': matern_nu,
                    'lr': lr, 'num_iterations': num_iterations, 'max_train_footprints': max_train_footprints, 'mem_max_footprints':mem_max_footprints, 'x_lengthscale': x_lengthscale, 
                    'y_lengthscale': y_lengthscale, 'fix_x_y': fix_x_y, 'z_aux_lengthscale': z_aux_lengthscale, 'z_pred_lengthscale' : z_pred_lengthscale, 'eft_lengthscale': eft_lengthscale, 'outputscale': outputscale,
                    'gaussian_noise': gaussian_noise, 'learned_noise': learned_noise,  'model_name': model_name,
                    'patience': patience, 'min_delta': min_delta, 'COMPUTE_VAR': COMPUTE_VAR, 
                    'SAVE': SAVE, 'SAVE_preds': SAVE_preds, 'pos_loss': pos_loss, 'seed': _seed, 'norm_aux': norm_aux, 'norm_coords': norm_coords, 'norm_res': norm_res, 'coords': coords, 'pred_vals': pred_vals, 'stripe_size': stripe_size, 'aux': aux, 'extra_features': extra_features, 'max_split_diff': max_split_diff, 'max_tries': max_tries,
                    'composites': composites, 'ood': ood, 'agb': agb, 'ref_model': ref_model})

    # Set the random seeds for reproducibility
    random.seed(_seed), np.random.seed(_seed), torch.manual_seed(_seed), torch.cuda.manual_seed_all(_seed)

    # Define paths
    if composites : s2_path = join(path_predictions, f'{s2_tile}_{year}_composite.tif')
    else :
        s2_path = join(path_predictions, f'{s2_tile}_{year}_AGB_merged.tif')
        s2_path_ref = join(path_predictions, f'{s2_tile}_{year}_composite.tif')
    ckpt_path = path_kriging
    if not isdir(ckpt_path) : makedirs(ckpt_path, exist_ok=True)

    # Load and pre-process the data ###############################################################
    time_start = time()
    print('Loading data...')

    # Load the AGB predictions for the specified tile
    pred_agb, pred_std, pred_mask, meta, transform, upsampling_shape, nodataval = load_dense_preds(s2_path = s2_path, aux = aux) # shape (nrows, ncolumns) = (height, width)
    tile_geom = get_S2_bounds(s2_tile, path_geometries)
    if not composites : 
        # Fill any NaNs in pred_agb and pred_std with nearest neighbor interpolation
        pred_agb = fill_nan_with_nearest(pred_agb, pred_mask)
        pred_std = fill_nan_with_nearest(pred_std, pred_mask)
        # Load the reference composite to align the valid mask to the composites
        _, _, pred_mask, _, _, _, _  = load_dense_preds(s2_path = s2_path_ref, aux = aux)
        assert pred_agb.shape == pred_mask.shape, "Prediction and mask shapes do not match."

    # If needed, compute extra features
    if extra_features[0] != 'none' : extra_ft = get_extra_features(pred_agb, features = extra_features)
    else: extra_ft = None

    # Load the GEDI footprints within the geometry
    GEDI = gpd.read_file(path_gedi, engine = 'pyogrio', bbox = tile_geom.bounds)
    GEDI = GEDI[GEDI.intersects(tile_geom)]
    if GEDI.empty:
        run.log({'num_footprints': 0}, step = None)
        raise ValueError(f'No GEDI footprints in the tile geometry ({s2_tile}).')

    # Filter by the year of interest
    GEDI_year = filter_GEDI_dates(GEDI, year)
    if GEDI_year.empty or (len(GEDI_year) < 100) : 
        print(f'No GEDI footprints for the year {year} in the tile {s2_tile}. Using footprints from other timesteps instead.')
        run.config.update({'GEDI_year_empty': True})
    else: GEDI = GEDI_year
    
    # Reproject GEDI footprints to the local CRS
    crs = get_CRS_from_S2_tilename(s2_tile)
    GEDI = GEDI.to_crs(crs)
    print(f'Number of footprints: {GEDI.shape[0]}')

    # Get the row and column indices of the GEDI footprints
    with rs.open(s2_path) as src:
        width, height = src.width, src.height
        def get_idx(geom, src) :
            lon, lat = geom.x, geom.y
            row_index, col_index = src.index(lon, lat)
            return row_index, col_index
        GEDI[['row_idx', 'col_idx']] = GEDI.apply(lambda row: get_idx(row['geometry'], src), axis = 1).apply(pd.Series)

    # Filter the values that are outside of the width/height of the prediction
    GEDI = GEDI[(GEDI['row_idx'] < height) & (GEDI['row_idx'] >= 0) & (GEDI['col_idx'] < width) & (GEDI['col_idx'] >= 0)]
    
    # Remove the values where either the prediction or the STD are not defined
    valid_mask = (pred_mask == 0)
    if np.count_nonzero(~valid_mask) > 0 :
        GEDI = GEDI[valid_mask[GEDI['row_idx'], GEDI['col_idx']]]
        assert GEDI.shape[0] > 0, "No GEDI footprints left after filtering."

    # Dynamically calculate the lengthscales based on the furthest nearest neighbor
    print(x_lengthscale, y_lengthscale)
    if x_lengthscale == y_lengthscale == "dynamic" :
        print("Calculating dynamic lengthscales based on the maximum distance to footprint.")
        x_lengthscale, y_lengthscale = get_furthest_neighbor(GEDI, height, width, second_max = True, unit = 500, norm_coords = norm_coords)
        print(f'New lengthscales: x={x_lengthscale}, y={y_lengthscale}')
        run.config.update({'x_lengthscale': x_lengthscale, 'y_lengthscale': y_lengthscale}, allow_val_change=True)

    # If there are multiple footprints in the same pixel, take the median
    print(f'\nNumber of footprints before groupby: {GEDI.shape[0]}')
    GEDI = GEDI.groupby(['row_idx', 'col_idx'], as_index = False).median(numeric_only = True)
    print(f'Number of footprints after groupby: {GEDI.shape[0]}\n')

    # Calculate the overall performance metrics, pre-kriging ######################################
    og_Y, og_X = GEDI['row_idx'].values, GEDI['col_idx'].values
    og_indices = GEDI['idx'].values
    og_gedi_agb = GEDI['agbd'].values
    predictions = pred_agb[og_Y, og_X]

    # 1. Overall RMSE
    residuals = predictions - og_gedi_agb
    rmse = np.sqrt(np.nanmean(np.pow(residuals, 2)))
    run.log({'overall_rmse': rmse}, step = None)
    print(f'\nOverall RMSE: {rmse}\n')
    # 2. Binned RMSE
    bins = np.arange(0, 501, 50)
    lb, ub = bins[:-1], bins[1:]
    for l, u in zip(lb, ub) :
        mask = (og_gedi_agb >= l) & (og_gedi_agb < u)
        binned_residuals = residuals[mask]
        num = np.sum(mask)
        run.log({f'binned/num_footprints_{l}-{u}': num}, step = None)
        if num > 0 :
            binned_rmse = np.sqrt(np.nanmean(np.pow(binned_residuals, 2)))
        else: binned_rmse = np.nan
        run.log({f'binned/pre_rmse_{l}-{u}': binned_rmse}, step = None)
    
    # Number of footprints in the tile
    num_footprints = GEDI.shape[0]
    run.log({'num_footprints': num_footprints}, step = None)

    # Get the train/val/test sets #################################################################
    
    save_dir = path_kriging
    if not isdir(save_dir) : makedirs(save_dir, exist_ok=True)
    _id = model_name.split('-')[-1]

    ood = ood and (GEDI.shape[0] > 500)
    
    # Define paths
    if (not composites) and ood : basemap_suffix = '-merged_preds'
    else: basemap_suffix = ''

    fname = f"splits-{_id}_{test_holdout:.1f}_{val_holdout:.1f}_{max_train_footprints}_{max_split_diff:.1f}_{stripe_size}{'_ood' if ood else ''}{basemap_suffix}.pkl"
    SPLITS_EXIST = isfile(join(save_dir, fname))
    SKIP = False
    SUBSAMPLE = False

    # If it has already been computed, load them
    if SPLITS_EXIST :
        print("Loading existing splits.")
        with open(join(save_dir, fname), 'rb') as f: split_indices = pickle.load(f)
        print(len(split_indices['train']), len(split_indices['val']), len(split_indices['test']))
        GEDI_train = GEDI[GEDI['idx'].isin(split_indices['train'])]
        GEDI_val = GEDI[GEDI['idx'].isin(split_indices['val'])]
        GEDI_hold_out = GEDI[GEDI['idx'].isin(split_indices['test'])]
        residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val, \
            = get_data(GEDI_train, GEDI_val, GEDI_hold_out, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run)
        GEDI = GEDI_train
        print(f"Train/val/test sizes: {GEDI.shape[0]} | {GEDI_val.shape[0]} | {GEDI_hold_out.shape[0]}")
    
    # Otherwise, do the split
    else:

        # Basic configuration, with max. 25K train footprints in memory and no OOD/subsampling
        base_fname = f"splits-{_id}_{test_holdout:.1f}_{val_holdout:.1f}_{mem_max_footprints}_{max_split_diff:.1f}_{stripe_size}.pkl" # basic split with the max. 25K train footprints in memory
        base_ood_fname = f"splits-{_id}_{test_holdout:.1f}_{val_holdout:.1f}_{mem_max_footprints}_{max_split_diff:.1f}_{stripe_size}_ood{basemap_suffix}.pkl" 
        
        # If we need to compute a different split than the basic one, load it first
        if fname != base_fname :
            
            # If the basic split does not exist, raise an error
            if not isfile(join(save_dir, base_fname)) : raise Exception('The basic configuration split does not exists, cannot compute the required split. Run it without ood or reduced max_train_footprints first.')

            # If we need to subsample the train and val footprints
            if (max_train_footprints < mem_max_footprints) :

                # If the base configs already exist, we just need to directly subsample from them
                if (ood and isfile(join(save_dir, base_ood_fname))) or (not ood and isfile(join(save_dir, base_fname))) :
                    
                    if ood : base_fname = base_ood_fname
                    SKIP = True
                    print("Subsampling existing train and val splits.")
                    with open(join(save_dir, base_fname), 'rb') as f: split_indices = pickle.load(f)
                    GEDI_train = GEDI[GEDI['idx'].isin(split_indices['train'])]
                    GEDI_val = GEDI[GEDI['idx'].isin(split_indices['val'])]
                    GEDI_hold_out = GEDI[GEDI['idx'].isin(split_indices['test'])]
                    
                    # Subsample
                    if (GEDI_train.shape[0] + GEDI_val.shape[0]) > max_train_footprints :
                        num_val_footprints = min(int(val_holdout * max_train_footprints), GEDI_val.shape[0])
                        num_train_footprints = min(int((1 - val_holdout) * max_train_footprints), GEDI_train.shape[0])
                        GEDI_train = GEDI_train.sample(num_train_footprints, random_state = _seed)
                        GEDI_val = GEDI_val.sample(num_val_footprints, random_state = _seed)
                        print(f"Train/val/test sizes: {GEDI_train.shape[0]} | {GEDI_val.shape[0]} | {GEDI_hold_out.shape[0]}")
                        # Get the data
                        residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
                        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
                        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val, \
                            = get_data(GEDI_train, GEDI_val, GEDI_hold_out, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run)
                        GEDI = GEDI_train
                        # Save the indices
                        if SAVE:
                            split_indices = {'train' : GEDI['idx'].values, 'val' : GEDI_val['idx'].values, 'test' : GEDI_hold_out['idx'].values}
                            with open(join(save_dir, fname), 'wb') as f: pickle.dump(split_indices, f)

                    # We don't actually need to subsample, but we need to save the files as if we had done kriging with reduced footprints
                    else:

                        # Check that the ref_model's config matches the current one
                        check_config_match(ref_model, _id, run)

                        if SAVE:

                            # Indices
                            split_indices = {'train' : GEDI_train['idx'].values, 'val' : GEDI_val['idx'].values, 'test' : GEDI_hold_out['idx'].values}
                            with open(join(save_dir, fname), 'wb') as f: pickle.dump(split_indices, f)
                            
                            # Copy the .pkl file with the pre- and post-kriging predictions, the reference data and the indices
                            tif_path = path_kriging
                            source = join(tif_path, f"results-{ref_model}-{_id}.pkl")
                            destination = join(tif_path, f"results-{model_name}.pkl")
                            shutil.copy2(source, destination)

                        if SAVE_preds: 
                            
                            # Copy the model checkpoint
                            source = join(ckpt_path, f'{ref_model}-{_id}_{s2_tile}.ckpt')
                            destination = join(ckpt_path, f'{model_name}_{s2_tile}.ckpt')
                            shutil.copy2(source, destination)

                            # And the .pt file
                            source = join(ckpt_path, f'{ref_model}-{_id}_{s2_tile}.pt')
                            destination = join(ckpt_path, f'{model_name}_{s2_tile}.pt')
                            shutil.copy2(source, destination)
                             
                            # Save the corrected predictions as a GeoTIFF
                            source = join(tif_path, f"kriging-{ref_model}-{_id}.tif")
                            destination = join(tif_path, f"kriging-{model_name}.tif")
                            shutil.copy2(source, destination)
                        
                        exit(0)

                # If the OOD basefile does not exist, and we need to subsample from it, we need to create it first, and subsample at the very end
                elif ood and (not isfile(join(save_dir, base_ood_fname))) : SUBSAMPLE = True

            # If we need to do OOD only, or OOD + subsample
            if ood and ((max_train_footprints == mem_max_footprints) or SUBSAMPLE) :

                # Load the base config split and preserve the hold-out test set before any other filtering
                with open(join(save_dir, base_fname), 'rb') as f: test_indices = pickle.load(f)['test']
                _GEDI_holdout = GEDI[GEDI['idx'].isin(test_indices)]
                GEDI = GEDI[~GEDI['idx'].isin(test_indices)]
                test_holdout = 0.0                
                
                # Load the statistics of the residuals and filter out the outliers            
                with open(join(path_kriging, 'correction', f'stats_residuals_vs_true_agb_0-25{basemap_suffix}.pkl'), 'rb') as f: residuals_statistics = pickle.load(f)
                residuals_statistics = residuals_statistics['global']
                residuals = predictions - og_gedi_agb
                true_agb_bins = np.arange(0, 51, 25) # bins for which we drop the OOD values: 0-25, 25-50 t/ha
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
            
        else: # Compute the basic split, so no need to preserve any hold-out set
            test_indices = None
            _GEDI_holdout = None
            num_footprints = GEDI.shape[0]

        # If needed, actually split the data into train/val/test splits 
        if not SKIP :

            # Randomly subsample the data if there are too many footprints
            max_total_footprints = int(mem_max_footprints / (1 - test_holdout))
            if num_footprints > max_total_footprints :
                print(f"Subsampling {num_footprints} footprints to {max_total_footprints} footprints.")
                GEDI = GEDI.sample(max_total_footprints, random_state = 42)
                num_footprints = mem_max_footprints
            run.log({'num_footprints_subsampled': num_footprints}, step = None)
            all_GEDI = GEDI.copy()

            # If there are too few footprints, use a single pixel as a stripe size
            if num_footprints < 250 : 
                print(f"Using a stripe size of 1 pixel (only {num_footprints} footprints available).")
                stripe_size = 1
                run.config.update({'stripe_size': stripe_size}, allow_val_change=True)
            elif num_footprints < 600 :
                print(f"Using a stripe size of 50 pixels (only {num_footprints} footprints available).")
                stripe_size = 50
                run.config.update({'stripe_size': stripe_size}, allow_val_change=True)

            # Repeat until we have a satisfying split
            satisfied, s_id, split_differences = False, 0, []
            while not satisfied and s_id < max_tries :
                satisfied, s_id, GEDI, GEDI_val, GEDI_hold_out, max_diff, rest = get_fold(all_GEDI, run, False, s_id, val_holdout, test_holdout, stripe_size, mem_max_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_agb, pred_std, aux, extra_ft, width, height, test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
                if max_diff is not None: split_differences.append(max_diff)
                else: split_differences.append(np.inf)

            # If could not find a satisfying split after max_tries attempts, skip this fold
            if not satisfied :
                if len(split_differences) == 0 : raise Exception('No valid splits were found, exiting.')
                print(f'Could not find a satisfying split after {max_tries} attempts. Picking least worst.')
                s_id = np.argmin(split_differences)
                _, _, GEDI, GEDI_val, GEDI_hold_out, _, rest = get_fold(all_GEDI, run, True, s_id, val_holdout, test_holdout, stripe_size, mem_max_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_agb, pred_std, aux, extra_ft, width, height, test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
            if GEDI is None or rest is None: raise Exception('No valid split found, exiting.')

            # If need, subsample the training and validation sets to the specified max number of train footprints
            if SUBSAMPLE :
                num_val_footprints = int(max_train_footprints * val_holdout)
                num_train_footprints = max_train_footprints - num_val_footprints
                GEDI = GEDI.sample(num_train_footprints, random_state = _seed)
                GEDI_val = GEDI_val.sample(num_val_footprints, random_state = _seed)
                
            # If needed, load back the hold-out set
            if (test_indices is not None) or SUBSAMPLE :
                residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
                gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
                agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val, \
                    = get_data(GEDI, GEDI_val, _GEDI_holdout if (test_indices is not None) else GEDI_hold_out, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run)
                if test_indices is not None : GEDI_hold_out = _GEDI_holdout
            else:
                # Parse the results from the split
                residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test, \
                gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test, \
                agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val, norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val = rest            

            # Save the indices
            if SAVE:
                split_indices = {'train' : GEDI['idx'].values, 'val' : GEDI_val['idx'].values, 'test' : GEDI_hold_out['idx'].values}
                with open(join(save_dir, fname), 'wb') as f: pickle.dump(split_indices, f)

    run.log({f'num_train_footprints': GEDI.shape[0], f'num_val_footprints': GEDI_val.shape[0], f'num_test_footprints': GEDI_hold_out.shape[0]}, step = None)

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
    if SAVE_preds: torch.save({'train_x': train_x, 'train_y': train_y, 'likelihood_state': likelihood.state_dict(), 'ndims': train_x.shape[1], 'norm_values': norm_values, 'config': {'arch': arch, 'year': year, 'ens_models': '_'.join(ens_models), 'aux': aux, 'extra_features': extra_features, 'matern_nu': matern_nu, 'COMPUTE_VAR': COMPUTE_VAR, 'norm_res': norm_res, 'norm_coords': norm_coords, 'coords': coords, 'pred_vals': pred_vals, 'composites': composites, 'agb': agb, 'norm_aux': norm_aux}}, join(ckpt_path, f'{model_name}_{s2_tile}.pt'))
    

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
        if SAVE_preds: torch.save(model.state_dict(), join(ckpt_path, f'{model_name}_{s2_tile}.ckpt'))
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
            tile_x_crop = tile_x[i : min(i + N, pred_y.size)].cuda()
            observed_pred = likelihood(model(tile_x_crop))
            pred_y[i : min(i + N, pred_y.size)] = observed_pred.mean.cpu().numpy()
            if COMPUTE_VAR : pred_y_var[i : min(i + N, pred_y.size)] = observed_pred.variance.cpu().numpy()
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
    corrected[~valid_mask] = np.nan
    if COMPUTE_VAR : 
        residuals_var[~valid_mask] = np.nan
    del valid_mask

    # Now calculate the new RMSE on the test set
    predictions_test_corrected = corrected[Y_test, X_test]
    residuals_test_corrected = predictions_test_corrected - gedi_agb_test
    rmse_corrected = np.sqrt(np.nanmean(np.pow(residuals_test_corrected, 2)))
    run.log({f'rmse_test_post': rmse_corrected}, step = None)
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
        tif_path = path_kriging
        if not isdir(tif_path) : makedirs(tif_path, exist_ok=True)
        # Save the pre-kriging predictions and GTs at the footprints' locations
        with open(join(tif_path, f"results-{model_name}.pkl"), 'wb') as f:
            pickle.dump({'pre': pred_agb[og_Y, og_X], 'ref': og_gedi_agb, 'idx': og_indices, 'post': corrected[og_Y, og_X]}, f)
    del(pred_std)
    if not agb: del(pred_agb)

    ###################################################################################################################
    print('Calculating overall performance...')

    # Calculate the overall performance
    # 1. Overall RMSE
    res = corrected[og_Y, og_X] - og_gedi_agb
    rmse = np.sqrt(np.nanmean(np.pow(res, 2)))
    run.log({'overall_rmse_post': rmse}, step = None)
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
        run.log({f'binned/post_rmse_{l}-{u}': binned_rmse}, step = None)

    ###################################################################################################################
    # Save the dense predictions
    if SAVE_preds :
        start_time = time()
        print('Saving the corrected prediction...')
        count = 3 if COMPUTE_VAR else 2
        meta.update(count = count, dtype = 'float32', nodata = np.nan)
        with rs.open(join(tif_path, f"kriging-{model_name}.tif"), 'w', **meta) as dst:
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
    
    # Finish the Weights & Biases run
    print()
    print('Finishing the run...')
    run.finish()
    print('Done!')


    # Measure total time taken
    total_time = time() - program_start
    print(f'Total time taken: {total_time} seconds.')


if __name__ == '__main__':
    main()
