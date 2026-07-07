"""
Low-level helpers shared by the Sumatra GEDI kriging pipeline.

This module holds the pieces that operate on a single split or a single array:
  - SplitBundle     : dataclass carrying one train/val/test split (was a 30-element tuple)
  - get_data        : build a SplitBundle from pre-computed splits
  - get_extra_features / load_dense_preds : feature computation + prediction loading
  - parser          : the command-line argument parser (returns an argparse.Namespace)
  - train_loop / train_with_retries : the GP training loop

The higher-level orchestration (geographic splitting + the end-to-end per-map pipeline)
lives in sumatra/pipeline.py, and all file paths live in sumatra/paths.py.
Depends on kriging.core for the GP model and geometry helpers.
"""

import argparse
import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import rasterio as rs
import torch
import gpytorch
from scipy.ndimage import distance_transform_edt, uniform_filter, sobel, laplace, maximum_filter, minimum_filter
from linear_operator.utils.errors import NotPSDError
from linear_operator.utils.warnings import NumericalWarning
from rasterio.transform import Affine
from rasterio.enums import Resampling

from config import DATA_ROOT
from kriging.core import equals, float_or_str, str2bool, zero_first_two  # shared helpers (see src/kriging/core.py)


@dataclass
class SplitBundle:
    """
    Everything the GP needs about one train/val/test split, in one object (previously passed
    around as a bare 30-element tuple). Built by get_data() and get_fold(); consumed by
    run_kriging_for_map() in sumatra/pipeline.py. Fields are grouped train / val / test.

    Any field may be None when it does not apply (e.g. `std*` only when aux == 'STD',
    `og_*` only when coordinates were normalized, the `*_test` fields only when test_holdout > 0).
    """
    residuals: object = None          # prediction - GEDI AGB, per split
    residuals_val: object = None
    residuals_test: object = None
    X: object = None                  # footprint col indices (scaled if norm_coords)
    Y: object = None                  # footprint row indices (scaled if norm_coords)
    X_val: object = None
    Y_val: object = None
    X_test: object = None             # test indices are never scaled
    Y_test: object = None
    gedi_agb: object = None           # reference GEDI AGB, per split
    gedi_agb_val: object = None
    gedi_agb_test: object = None
    predictions: object = None        # model AGB sampled at the footprints, per split
    predictions_val: object = None
    predictions_test: object = None
    agbd_se: object = None            # GEDI standard errors (currently always None)
    agbd_se_val: object = None
    agbd_se_test: object = None
    gedi_dem: object = None           # DEM at footprints (None unless aux uses DEM)
    gedi_dem_val: object = None
    std: object = None                # prediction STD at footprints (None unless aux == 'STD')
    std_val: object = None
    dem: object = None                # dense DEM array (None here)
    eft: object = None                # extra-feature vectors at footprints (None if no extra features)
    eft_val: object = None
    norm_values: object = None        # dict of normalization stats, for de-normalizing later
    og_X_train: object = None         # un-scaled pixel indices (kept when norm_coords)
    og_Y_train: object = None
    og_X_val: object = None
    og_Y_val: object = None

def get_data(GEDI, GEDI_val, GEDI_hold_out, pred_agb, pred_std, extra_ft, aux, norm_aux, norm_coords, path_dem, s2_tile, transform, upsampling_shape, width, height, run) :
    """
    Extract, from the train/val/test GEDI splits, everything the GP needs, and return it as the
    "split bundle" -- a 30-element tuple threaded through get_fold() / get_train_val_test_split()
    and unpacked in each script's __main__.

    Args:
    - GEDI, GEDI_val, GEDI_hold_out: geopandas dataframes for the train / val / test splits.
    - pred_agb, pred_std: dense AGB prediction and its STD (2D arrays).
    - extra_ft: (H, W, C) array of extra features, or None.
    - aux: which auxiliary variable to attach ('STD' or 'none').
    - norm_aux: 'min_max' to min-max-scale aux/extra features, else False.
    - norm_coords: if True, divide pixel coords by (width, height).

    Returns (the "split bundle", in order):
        residuals, residuals_val, residuals_test   prediction - GEDI AGB, per split
        X, Y                                        train footprint col/row pixel indices (scaled if norm_coords)
        X_val, Y_val                                val footprint indices
        X_test, Y_test                              test footprint indices (never scaled)
        gedi_agb, gedi_agb_val, gedi_agb_test       reference GEDI AGB, per split
        predictions, predictions_val, predictions_test   model AGB sampled at the footprints, per split
        agbd_se, agbd_se_val, agbd_se_test          GEDI standard errors (currently always None)
        gedi_dem, gedi_dem_val                       DEM at footprints (None unless aux uses DEM)
        std, std_val                                 prediction STD at footprints (None unless aux == 'STD')
        dem                                          dense DEM array (None here)
        eft, eft_val                                 extra-feature vectors at footprints (None if no extra_ft)
        norm_values                                  dict of normalization stats, for de-normalizing later
        og_X_train, og_Y_train, og_X_val, og_Y_val   un-scaled pixel indices (kept when norm_coords)
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

    return SplitBundle(
        residuals, residuals_val, residuals_test, X, Y, X_val, Y_val, X_test, Y_test,
        gedi_agb, gedi_agb_val, gedi_agb_test, predictions, predictions_val, predictions_test,
        agbd_se, agbd_se_val, agbd_se_test, gedi_dem, gedi_dem_val, std, std_val, dem, eft, eft_val,
        norm_values, og_X_train, og_Y_train, og_X_val, og_Y_val)


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
    if np.count_nonzero(np.isnan(pred_agb)) != 0 :
        # Temporarily fill NaNs with nearest neighbor interpolation for feature computation
        nan_mask = np.isnan(pred_agb)
        filled_pred_agb = pred_agb.copy()
        inds = distance_transform_edt(nan_mask, return_distances=False, return_indices=True) # (2,W,H)
        inds = tuple(inds[:, nan_mask])
        filled_pred_agb[nan_mask] = filled_pred_agb[inds]
        pred_agb = filled_pred_agb
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


def load_dense_preds(s2_path, model_name, aux = None, bounds = None) :
    """
    This function loads the dense predictions of the model for the Sentinel-2 tile.

    Args:
    - s2_path: string, path to the best model's prediction for the S2 tile.
    - model_name: string, name of the model.
    - aux: string, auxiliary variable to use. If 'STD', the function will load the STD of the ensemble.
    - bounds: list of floats, bounds of the tile to load. If None, the whole tile is loaded.

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

        if bounds is not None : 
            window = rs.windows.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], src.transform)
            window = window.round_offsets().round_lengths()
            _transform = src.window_transform(window)
        else: 
            window = None
            _transform = src.transform

        # We need to resample even though CCI has a 100m resolution because their pixel grids are not perfectly aligned
        if "cci" in model_name : 
            with rs.open(f'{DATA_ROOT}/Sumatra-AGB/pred_rasters/agbd_100m.tif', 'r') as _src:
                if bounds is not None : 
                    sumatra_window = rs.windows.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], _src.transform)
                    sumatra_window = sumatra_window.round_offsets().round_lengths()
                else: sumatra_window = None
                sumatra_1_shape = _src.read(1, window = sumatra_window).shape
                print('sumatra_1_shape', sumatra_1_shape)
            data = src.read(window = window, out_shape = (src.count, sumatra_1_shape[0], sumatra_1_shape[1]), resampling = Resampling.bilinear)
        else:
            data = src.read(window = window)
        
        # Get the predictions and the STD
        pred_agb = data[0, :, :]
        print('pred_agb shape', pred_agb.shape)
        pred_std = data[1, :, :] if aux == 'STD' else None

        # Get the metadata of the file (to later save the Kriging results)
        meta = src.meta
        upsampling_shape = pred_agb.shape
        nodataval = src.nodata
        target_crs = src.crs
    
    # Get the mask
    if nodataval is not None:
        if aux == 'STD' : pred_mask = (equals(pred_agb, nodataval) | equals(pred_std, nodataval)).astype(np.uint8)
        else: pred_mask = (equals(pred_agb, nodataval)).astype(np.uint8)
    else: pred_mask = np.zeros(upsampling_shape, dtype = np.uint8)

    return pred_agb, pred_std, pred_mask, meta, _transform, upsampling_shape, nodataval, target_crs


def parser():
    """ 
    Returns an `ArgumentParser()` object containing the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2_tile', type = str, required = True, help = 'S2 tile.')
    parser.add_argument('--year', type = int, required = True, help = 'Year of the prediction.')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--ens_models', type = str, nargs = '+', required = True, help = 'Models for ensemble STD.')
    parser.add_argument('--test_holdout', type = float, required = True, help = '% of footprints to hold out for test.')
    parser.add_argument('--val_holdout', type = float, required = True, help = '% of footprints to hold out for validation.')
    parser.add_argument('--stripe_size', type = int, required = True, help = 'Size (in pixels) of a stripe.')
    parser.add_argument('--path_predictions', type = str, required = True, help = 'Directory with the predictions.')
    parser.add_argument('--path_gedi', type = str, required = True, help = 'Directory with the GEDI footprints.')
    parser.add_argument('--path_dem', type = str, required = True, help = 'Directory with the DEM.')
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
    args = parser.parse_args()

    # Determine how to process the input features
    if args.norm_aux == 'false' : args.norm_aux = False
    elif args.norm_aux == 'min_max' : args.norm_aux = 'min_max'
    else: raise ValueError(f"norm_aux must be either 'false' or 'min_max', got {args.norm_aux}.")

    # Return the parsed arguments as a single namespace (access fields as args.<name>).
    return args


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



            model.train(), likelihood.train()
    
    return model, best_model_state, optimizer, mll


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
        if _lr != lr : print(f"    Reducing learning rate to {_lr}.")
        try:
            with warnings.catch_warnings() :
                warnings.filterwarnings("error", category=NumericalWarning) # treat the NumericalWarning as an error
                model, best_model_state, optimizer, mll = train_loop(og_model, og_likelihood, train_x, train_y, val_x, val_y, run, num_iterations, patience, min_delta, pos_loss, _lr, aux, coords, pred_vals, fix_x_y)
            return model, best_model_state, optimizer, mll
        except NotPSDError as e:
            print(f"Training failed at lr={_lr} with NotPSDError: {e}")
        except NumericalWarning as e:
            print(f"Training failed at lr={_lr} with NumericalWarning: {e}")
        except Exception as e:
            raise Exception(f"Training failed with unexpected error: {e}")
    raise RuntimeError("Training failed for all learning rates.")

__all__ = ['SplitBundle', 'get_data', 'get_extra_features', 'load_dense_preds', 'parser', 'train_loop', 'train_with_retries']
