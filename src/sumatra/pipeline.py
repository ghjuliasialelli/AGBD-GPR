"""
The end-to-end Sumatra kriging pipeline, shared by the three entry-point scripts
(kriging_gedi.py, kriging_downsampled_gedi.py, kriging_ours_gedi.py).

Previously each of those scripts carried its own ~700-line copy of this logic. They now just
build the arguments and call `run_kriging_for_map()` here, which corrects ONE prediction raster:

    load prediction + GEDI  ->  geographic train/val/test split  ->  fit GP to residuals
      ->  predict residual field over the whole tile  ->  corrected = prediction - residual
      ->  report pre/post RMSE and save the GeoTIFF + metrics.

All file paths go through sumatra/paths.py; the per-split data is carried in a SplitBundle
(sumatra/common.py). Low-level helpers (get_data, feature computation, GP training) also live
in sumatra/common.py.
"""

import os
import gc
import pickle
import random
from os.path import isfile, join
from time import time
from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
import rasterio as rs
import geopandas as gpd
import torch
import gpytorch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from config import DATA_ROOT
from kriging.core import ExactGPModel, get_furthest_neighbor, get_tile_data, get_train_val_data
from sumatra import paths
from sumatra.common import (
    SplitBundle, get_data, get_extra_features, load_dense_preds, train_with_retries,
)

# Match the original scripts' global settings (TF32 off for reproducible GP fits).
torch.backends.cuda.matmul.allow_tf32 = False
os.environ.setdefault('WANDB_INIT_TIMEOUT', '300')


def set_seeds(seed):
    """Seed all the RNGs kriging touches, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def geographical_train_test_split(GEDI, model_name, order, width, height, val_size=0.2,
                                  test_size=0.3, stripe_size=100, max_train_footprints=25000,
                                  random_state=42):
    """
    Split the GEDI footprints into geographically-separated train/val/test sets. Uses a stripe
    (checkerboard) pattern of `stripe_size` pixels; `stripe_size == 1` falls back to a random split.

    Returns (GEDI_train, GEDI_val, GEDI_test), or (None, None, None) if a split came out empty.
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
        plt.savefig(paths.checkerboard_png(stripe_size, model_name, make=True))
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

    # Build a 2d array visualizing the split, and save it (read back by the notebook)
    split_map = np.zeros((height, width), dtype = np.uint8)
    for split in ['train', 'val', 'test'] :
        for patch in metrics[split]['stripes'] :
            row, col = patch
            rows, cols = list(range(row, min(row + stripe_size, height))), list(range(col, min(col + stripe_size, width)))
            if split == 'train' : split_map[np.ix_(rows, cols)] = 1
            elif split == 'val' : split_map[np.ix_(rows, cols)] = 2
            elif split == 'test' : split_map[np.ix_(rows, cols)] = 3

    # Plot the split map
    cmap = ListedColormap(['white', '#5B9BD5', '#F79646', '#70AD47'])
    plt.figure(figsize = (10,10))
    plt.imshow(split_map, cmap = cmap, interpolation = 'nearest')
    plt.title('Geographical split of the GEDI data')
    plt.xticks()
    plt.yticks()
    plt.colorbar(ticks = [0, 1, 2, 3], label = 'Split', orientation = 'horizontal')
    plt.clim(-0.5, 3.5)
    plt.savefig(paths.checkerboard_png(stripe_size, model_name, make=True))

    with open(paths.split_map_pkl(model_name, make=True), 'wb') as f : pickle.dump(split_map, f)

    # Return the data
    print(f'    Number of footprints: {GEDI_train.shape[0]} (train) | {GEDI_val.shape[0]} (validation) | {GEDI_test.shape[0]} (test)')
    return GEDI_train, GEDI_val, GEDI_test


def get_fold(all_GEDI, model_name, run, satisfied, s_id, val_holdout, test_holdout, stripe_size,
             width, height, max_train_footprints, path_dem, s2_tile, transform, upsampling_shape,
             pred_std, pred_agb, aux, extra_ft, test_indices=None,
             order=['train', 'test', 'train', 'val'], norm_coords=False, norm_aux=None, max_split_diff=15):
    """
    Draw one geographic split and, if its train/val/test RMSEs agree closely enough, package it
    into a SplitBundle. Returns (satisfied, next_s_id, GEDI, GEDI_val, GEDI_hold_out,
    max_pairwise_rmse_diff, SplitBundle). Called repeatedly until a satisfying split is found.
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

    # Assemble the auxiliary data if the fold is satisfied
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

    bundle = SplitBundle(
        residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test,
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test,
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val,
        norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val)
    return satisfied, s_id + 1, GEDI, GEDI_val, GEDI_hold_out, max(pairwise_differences), bundle


def _search_for_fold(all_GEDI, model_name, run, val_holdout, test_holdout, stripe_size, width, height,
                     max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std,
                     pred_agb, aux, extra_ft, test_indices, norm_coords, norm_aux, max_split_diff, max_tries):
    """Repeatedly call get_fold() until a satisfying split is found (or fall back to the least-bad)."""
    satisfied, s_id, split_differences = False, 0, []
    while not satisfied and s_id < max_tries :
        satisfied, s_id, GEDI, GEDI_val, GEDI_hold_out, max_diff, bundle = get_fold(
            all_GEDI, model_name, run, False, s_id, val_holdout, test_holdout, stripe_size, width, height,
            max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std, pred_agb, aux, extra_ft,
            test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
        if max_diff is not None: split_differences.append(max_diff)

    if not satisfied :
        if len(split_differences) == 0 : raise Exception('No valid splits were found, exiting.')
        print(f'Could not find a satisfying split after {max_tries} attempts. Picking least worst.')
        s_id = np.argmin(split_differences)
        _, _, GEDI, GEDI_val, GEDI_hold_out, _, bundle = get_fold(
            all_GEDI, model_name, run, True, s_id, val_holdout, test_holdout, stripe_size, width, height,
            max_train_footprints, path_dem, s2_tile, transform, upsampling_shape, pred_std, pred_agb, aux, extra_ft,
            test_indices = test_indices, norm_coords = norm_coords, norm_aux = norm_aux, max_split_diff = max_split_diff)
    if GEDI is None or bundle is None: raise Exception('No valid split found, exiting.')
    return GEDI, GEDI_val, GEDI_hold_out, bundle


def get_train_val_test_split(GEDI, model_name, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords,
                             path_dem, s2_tile, transform, upsampling_shape, width, height, run,
                             test_holdout, val_holdout, max_train_footprints, max_split_diff, stripe_size,
                             max_tries, test_indices=None, SAVE=True, SPLITS_EXIST=False, perform_ood=False,
                             split_path=None, ood_split_path=None, ood_extra=None):
    """
    Return (GEDI_train, GEDI_val, GEDI_hold_out, SplitBundle) for a run. Three cases:
      1. the split already exists and no OOD filtering is needed -> load `split_path`;
      2. the base split exists and OOD filtering is requested -> load the test set, drop
         out-of-distribution footprints, re-split train/val, and save to `ood_split_path`;
      3. otherwise -> search for a fresh geographic split and save it to `split_path`.
    """

    # Case 1: the split already exists and OOD is not needed -> just load it
    if SPLITS_EXIST and (not perform_ood) :
        print("Loading existing splits.")
        with open(split_path, 'rb') as f: split_indices = pickle.load(f)
        GEDI_train = GEDI[GEDI['idx'].isin(split_indices['train'])]
        GEDI_val = GEDI[GEDI['idx'].isin(split_indices['val'])]
        GEDI_hold_out = GEDI[GEDI['idx'].isin(split_indices['test'])]
        bundle = get_data(GEDI_train, GEDI_val, GEDI_hold_out, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run)
        GEDI = GEDI_train
        print(f"Train/val/test sizes: {GEDI.shape[0]} | {GEDI_val.shape[0]} | {GEDI_hold_out.shape[0]}")

    # Case 2: the base split exists but we need to drop OOD samples first
    elif SPLITS_EXIST and perform_ood :
        print("Loading existing test split and removing OOD samples to get new train and val splits.")
        with open(split_path, 'rb') as f: test_indices = pickle.load(f)['test']
        _GEDI_holdout = GEDI[GEDI['idx'].isin(test_indices)]
        GEDI = GEDI[~GEDI['idx'].isin(test_indices)]
        test_holdout = 0.0

        # Load the statistics of the residuals and filter out the outliers
        with open(join(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/correction', 'stats_residuals_vs_true_agb_0-25.pkl'), 'rb') as f: residuals_statistics = pickle.load(f)
        residuals_statistics = residuals_statistics['global']
        og_indices, og_gedi_agb, predictions = ood_extra
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

        # Re-split the remaining footprints into train/val (test is fixed above)
        num_footprints = GEDI.shape[0]
        max_total_footprints = int(max_train_footprints / (1 - test_holdout))
        if num_footprints > max_total_footprints :
            print(f"Subsampling {num_footprints} footprints to {max_total_footprints} footprints.")
            GEDI = GEDI.sample(max_total_footprints, random_state = 42)
        all_GEDI = GEDI.copy()
        GEDI, GEDI_val, _, bundle = _search_for_fold(
            all_GEDI, model_name, run, val_holdout, test_holdout, stripe_size, width, height, max_train_footprints,
            path_dem, s2_tile, transform, upsampling_shape, pred_std, pred_agb, aux, extra_ft, test_indices,
            norm_coords, norm_aux, max_split_diff, max_tries)

        # Re-attach the fixed hold-out set
        bundle = get_data(GEDI, GEDI_val, _GEDI_holdout, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run)
        GEDI_hold_out = _GEDI_holdout

        # Save the OOD split indices
        if SAVE:
            split_indices = {'train' : GEDI['idx'].values, 'val' : GEDI_val['idx'].values, 'test' : GEDI_hold_out['idx'].values}
            with open(ood_split_path, 'wb') as f: pickle.dump(split_indices, f)

    # Case 3: no existing split -> search for a fresh geographic one
    else:
        # Randomly subsample the data if there are too many footprints
        num_footprints = GEDI.shape[0]
        max_total_footprints = int(max_train_footprints / (1 - test_holdout))
        if num_footprints > max_total_footprints :
            print(f"Subsampling {num_footprints} footprints to {max_total_footprints} footprints.")
            GEDI = GEDI.sample(max_total_footprints, random_state = 42)
        all_GEDI = GEDI.copy()
        GEDI, GEDI_val, GEDI_hold_out, bundle = _search_for_fold(
            all_GEDI, model_name, run, val_holdout, test_holdout, stripe_size, width, height, max_train_footprints,
            path_dem, s2_tile, transform, upsampling_shape, pred_std, pred_agb, aux, extra_ft, test_indices,
            norm_coords, norm_aux, max_split_diff, max_tries)

        # Save the split indices
        if SAVE:
            split_indices = {'train' : GEDI['idx'].values, 'val' : GEDI_val['idx'].values, 'test' : GEDI_hold_out['idx'].values}
            with open(split_path, 'wb') as f: pickle.dump(split_indices, f)

    return GEDI, GEDI_val, GEDI_hold_out, bundle


def run_kriging_for_map(cfg, path_predictions, model_name, s2_tile, reference='gedi'):
    """
    Correct ONE prediction raster and (optionally) save the result. `cfg` is the argparse
    Namespace returned by common.parser(); `path_predictions` is the raster to correct;
    `model_name` names all outputs. Returns the post-kriging test-set RMSE.
    """

    program_start = time()
    run = None
    aux, extra_features = cfg.aux, cfg.extra_features
    x_lengthscale, y_lengthscale = cfg.x_lengthscale, cfg.y_lengthscale

    # --- Load and pre-process the data -------------------------------------------------
    print('Loading data...')
    pred_agb, pred_std, pred_mask, meta, transform, upsampling_shape, nodataval, target_crs = \
        load_dense_preds(s2_path = path_predictions, model_name = model_name, aux = aux)  # (height, width)

    # If needed, compute extra features
    if extra_features[0] != 'none' : extra_ft = get_extra_features(pred_agb, features = extra_features)
    else: extra_ft = None

    # Load the GEDI footprints and reproject them to the prediction's CRS
    GEDI = gpd.read_file(cfg.path_gedi, engine = 'pyogrio')
    GEDI = GEDI.rename(columns={"index": "idx"})
    if GEDI.empty: raise ValueError('No GEDI footprints in the geometry.')
    GEDI = GEDI.to_crs(target_crs)
    print(f'Number of footprints: {GEDI.shape[0]}')

    # Get the row/column indices of the GEDI footprints on the prediction grid
    width, height = pred_agb.shape[1], pred_agb.shape[0]
    print('height, width:', height, width)
    with rs.open(path_predictions) as src:
        def get_idx(geom, src) :
            lon, lat = geom.x, geom.y
            row_index, col_index = src.index(lon, lat)
            return row_index, col_index
        GEDI[['row_idx', 'col_idx']] = GEDI.apply(lambda row: get_idx(row['geometry'], src), axis = 1).apply(pd.Series)

    # Drop footprints outside the prediction, or where prediction/STD are undefined
    GEDI = GEDI[(GEDI['row_idx'] < height) & (GEDI['row_idx'] >= 0) & (GEDI['col_idx'] < width) & (GEDI['col_idx'] >= 0)]
    valid_mask = (pred_mask == 0)
    if np.count_nonzero(~valid_mask) > 0 :
        GEDI = GEDI[valid_mask[GEDI['row_idx'], GEDI['col_idx']]]
        assert GEDI.shape[0] > 0, "No GEDI footprints left after filtering."

    # Dynamically calculate the lengthscales based on the furthest nearest neighbor
    if x_lengthscale == y_lengthscale == "dynamic" :
        print("Calculating dynamic lengthscales based on the maximum distance to footprint.")
        x_lengthscale, y_lengthscale = get_furthest_neighbor(GEDI, height, width, second_max = True, unit = 10, norm_coords = cfg.norm_coords)
        print(f'New lengthscales: x={x_lengthscale}, y={y_lengthscale}')

    # If there are multiple footprints in the same pixel, take the median
    print(f'\nNumber of footprints before groupby: {GEDI.shape[0]}')
    GEDI = (GEDI.groupby(['row_idx', 'col_idx'], as_index=False).agg({'agbd': 'median', 'idx': list, **{col: 'first' for col in GEDI.select_dtypes(include='number').columns if col not in ['row_idx', 'col_idx', 'agbd', 'idx']}}))
    print(f'Number of footprints after groupby: {GEDI.shape[0]}\n')

    # --- Pre-kriging performance -------------------------------------------------------
    og_Y, og_X = GEDI['row_idx'].values.astype(int), GEDI['col_idx'].values.astype(int)
    og_indices = GEDI['idx'].values
    og_gedi_agb = GEDI['agbd'].values
    predictions = pred_agb[og_Y, og_X]
    residuals = predictions - og_gedi_agb
    rmse = np.sqrt(np.nanmean(np.pow(residuals, 2)))
    print(f'\nOverall RMSE: {rmse}\n')

    # --- Build (or load) the train/val/test split --------------------------------------
    split_path_base = paths.split_pkl(model_name, cfg.test_holdout, cfg.val_holdout, cfg.max_train_footprints, cfg.max_split_diff, cfg.stripe_size, reference, ood=False, make=True)
    split_path_ood = paths.split_pkl(model_name, cfg.test_holdout, cfg.val_holdout, cfg.max_train_footprints, cfg.max_split_diff, cfg.stripe_size, reference, ood=True)

    ood = cfg.ood and (GEDI.shape[0] > 500)
    if ood :
        if isfile(split_path_ood) :
            SPLITS_EXIST, perform_ood, split_path = True, False, split_path_ood
        elif isfile(split_path_base) :
            SPLITS_EXIST, perform_ood, split_path = True, True, split_path_base
        else:
            raise Exception('OOD splits requested but base split file does not exist. Please run without OOD first.')
    else:
        split_path = split_path_base
        SPLITS_EXIST, perform_ood = isfile(split_path_base), False

    GEDI, GEDI_val, GEDI_hold_out, b = get_train_val_test_split(
        GEDI, model_name, pred_agb, pred_std, extra_ft, aux, cfg.norm_aux, cfg.norm_coords, cfg.path_dem,
        s2_tile, transform, upsampling_shape, width, height, run, cfg.test_holdout, cfg.val_holdout,
        cfg.max_train_footprints, cfg.max_split_diff, cfg.stripe_size, cfg.max_tries, test_indices = None,
        SAVE = cfg.SAVE, SPLITS_EXIST = SPLITS_EXIST, perform_ood = perform_ood, split_path = split_path,
        ood_split_path = split_path_ood, ood_extra = (og_indices, og_gedi_agb, predictions))

    # Unpack the split bundle into locals used below
    residuals, residuals_val, residuals_test = b.residuals, b.residuals_val, b.residuals_test
    X, Y, X_val, Y_val, X_test, Y_test = b.X, b.Y, b.X_val, b.Y_val, b.X_test, b.Y_test
    gedi_agb, gedi_agb_val, gedi_agb_test = b.gedi_agb, b.gedi_agb_val, b.gedi_agb_test
    predictions, predictions_val = b.predictions, b.predictions_val
    std, std_val, dem, eft, eft_val, norm_values = b.std, b.std_val, b.dem, b.eft, b.eft_val, b.norm_values

    # --- Pre-process the residuals -----------------------------------------------------
    # If regressing the AGB values instead of the residuals, use the AGB values as targets
    if cfg.agb : residuals, residuals_val, residuals_test = gedi_agb, gedi_agb_val, gedi_agb_test

    # Normalize the residuals (z-scoring) if requested
    if cfg.norm_res :
        res_mu, res_std = np.mean(residuals), np.std(residuals)
        norm_values['res'] = {'mean': res_mu, 'std': res_std}
        print('Residuals normalizations values: mean = %.4f, std = %.4f' % (res_mu, res_std))
        if res_std == 0 : res_std = 1  # Avoid division by zero
        residuals = (residuals - res_mu) / res_std
        residuals_val = (residuals_val - res_mu) / res_std
        residuals_test = (residuals_test - res_mu) / res_std

    # Assemble the GP training/validation tensors
    train_x, train_y, val_x, val_y, norm_values = get_train_val_data(X, Y, X_val, Y_val, predictions, predictions_val, std, std_val, eft, eft_val, aux, cfg.norm_aux, cfg.norm_coords, cfg.coords, cfg.pred_vals, norm_values, residuals, residuals_val)

    # --- Define + train the GP model ---------------------------------------------------
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if cfg.SAVE_preds:
        torch.save({'train_x': train_x, 'train_y': train_y, 'likelihood_state': likelihood.state_dict(),
                    'ndims': train_x.shape[1], 'norm_values': norm_values}, paths.checkpoint_data(model_name, make=True))

    model = ExactGPModel(train_x, train_y, likelihood, ndims = train_x.shape[1], matern_nu = cfg.matern_nu)
    model.likelihood.noise = cfg.gaussian_noise

    # Initialize the kernel lengthscales / outputscale
    lengthscales = []
    if cfg.coords: lengthscales.extend([x_lengthscale, y_lengthscale])
    if aux != 'none' : lengthscales.append(cfg.z_aux_lengthscale)
    if cfg.pred_vals : lengthscales.append(cfg.z_pred_lengthscale)
    if extra_features[0] != 'none' :
        lengthscales.extend([deepcopy(cfg.eft_lengthscale) for _ in range(len(extra_features))])
    model.covar_module.base_kernel.lengthscale = [[lengthscales]]
    model.covar_module.outputscale = cfg.outputscale

    # Train
    time_start = time()
    print('    Training the model...')
    model.cuda().to(torch.float32)
    likelihood.cuda()
    model, best_model_state, optimizer, mll = train_with_retries(model, likelihood, train_x, train_y, val_x, val_y, run, cfg.num_iterations, cfg.patience, cfg.min_delta, cfg.pos_loss, cfg.lr, aux, cfg.coords, cfg.pred_vals, cfg.fix_x_y)
    train_x, train_y, val_x, val_y = train_x.cpu().detach(), train_y.cpu().detach(), val_x.cpu().detach(), val_y.cpu().detach()
    del(train_x, train_y, val_x, val_y)
    torch.cuda.empty_cache()
    gc.collect()

    # Restore best model for evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
        if cfg.SAVE_preds: torch.save(model.state_dict(), paths.checkpoint_state(model_name, make=True))
    model.eval()
    likelihood.eval()
    print(f'    Done! In {time() - time_start} seconds.')

    # --- Apply kriging to the entire tile ----------------------------------------------
    start_time = time()
    print('    Predicting...')
    tile_x = get_tile_data(pred_agb, dem, pred_std, extra_ft, norm_values, cfg.coords, cfg.norm_coords, aux, cfg.norm_aux, cfg.pred_vals, extra_features)
    pred_y = np.zeros_like(tile_x[:, 0], dtype='float')
    if cfg.COMPUTE_VAR : pred_y_var = np.zeros_like(tile_x[:, 0], dtype='float')
    model = model.to(torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        N = 500
        for i in range(0, pred_y.size, N):
            tile_x_crop = tile_x[i : min(i + N, pred_y.size)]
            local_pred = np.zeros(tile_x_crop.shape[0], dtype='float')
            if cfg.COMPUTE_VAR : local_var = np.zeros(tile_x_crop.shape[0], dtype='float')
            _valid_mask = ~torch.any(torch.isnan(tile_x_crop), dim=1)
            if not torch.any(_valid_mask) : continue
            tile_x_crop = tile_x_crop[_valid_mask].cuda()
            observed_pred = likelihood(model(tile_x_crop))
            local_pred[_valid_mask.cpu().numpy()] = observed_pred.mean.cpu().numpy()
            pred_y[i : min(i + N, pred_y.size)] = local_pred
            if cfg.COMPUTE_VAR :
                local_var[_valid_mask.cpu().numpy()] = observed_pred.variance.cpu().numpy()
                pred_y_var[i : min(i + N, pred_y.size)] = local_var

    tile_x_crop = tile_x_crop.cpu().detach()
    tile_x = tile_x.cpu().detach()
    del(tile_x_crop, tile_x)
    torch.cuda.empty_cache()
    gc.collect()
    print(f'    Done! In {time() - start_time} seconds.')

    # Reshape the residual field and de-normalize
    residuals = pred_y.reshape(pred_agb.shape)
    if cfg.norm_res : residuals = residuals * res_std + res_mu
    del(pred_y)
    if cfg.COMPUTE_VAR :
        residuals_var = pred_y_var.reshape(pred_agb.shape)
        if cfg.norm_res : residuals_var *= np.pow(res_std, 2)
        residuals_var = np.clip(residuals_var, 0, None)  # Ensure non-negativity
        del(pred_y_var)

    # Subtract the residual field from the prediction
    if cfg.agb : corrected = np.clip(residuals, 0, None)
    else: corrected = np.clip(pred_agb - residuals, 0, None)
    if not cfg.SAVE_preds : del(residuals)

    # Mask out where the input was undefined
    corrected[~valid_mask] = np.nan
    if cfg.COMPUTE_VAR : residuals_var[~valid_mask] = np.nan
    del valid_mask

    # Post-kriging test-set RMSE
    residuals_test_corrected = corrected[Y_test, X_test] - gedi_agb_test
    rmse_corrected = np.sqrt(np.nanmean(np.pow(residuals_test_corrected, 2)))
    print(f'    RMSE test corrected: {rmse_corrected}\n')
    del(residuals_test_corrected)

    # Cleanup the GPU
    model, likelihood = model.to('cpu'), likelihood.to('cpu')
    del(model, optimizer, likelihood, mll)
    torch.cuda.empty_cache()
    gc.collect()

    # --- Save the per-footprint results (pre/post) -------------------------------------
    if cfg.SAVE :
        with open(paths.results_pkl(cfg.arch, cfg.year, cfg.ens_models, model_name, reference, cfg.stripe_size, ood=ood, make=True), 'wb') as f:
            pickle.dump({'pre': pred_agb[og_Y, og_X], 'ref': og_gedi_agb, 'idx': og_indices, 'post': corrected[og_Y, og_X]}, f)
    del(pred_std)
    if not cfg.agb: del(pred_agb)

    # Overall + test-set post-kriging RMSE
    res = corrected[og_Y, og_X] - og_gedi_agb
    print(f'\nOverall RMSE (post-kriging): {np.sqrt(np.nanmean(np.pow(res, 2)))}\n')
    res_test = corrected[Y_test, X_test] - gedi_agb_test
    rmse_test = np.sqrt(np.nanmean(np.pow(res_test, 2)))
    print(f'\nTest set RMSE (post-kriging): {rmse_test}\n')

    # --- Save the corrected raster -----------------------------------------------------
    if cfg.SAVE_preds :
        start_time = time()
        print('Saving the corrected prediction...')
        count = 3 if cfg.COMPUTE_VAR else 2
        meta.update(count = count, dtype = 'float32', nodata = np.nan)
        if 'cci' in model_name : meta.update({"height": height, "width": width})
        with rs.open(paths.corrected_tif(cfg.arch, cfg.year, cfg.ens_models, model_name, reference, cfg.stripe_size, ood=ood, make=True), 'w', **meta) as dst:
            dst.write(corrected, 1)
            dst.set_band_description(1, 'AGB')
            if cfg.agb : residuals = pred_agb - residuals
            dst.write(residuals, 2)
            dst.set_band_description(2, 'Residuals')
            if cfg.COMPUTE_VAR :
                dst.write(np.sqrt(residuals_var), 3)
                dst.set_band_description(3, 'STD')
        print(f'Done! In {time() - start_time} seconds.')

    print(f'Total time taken: {time() - program_start} seconds.')
    return rmse_test
