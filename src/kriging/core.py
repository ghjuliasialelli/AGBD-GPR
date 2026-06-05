"""
Shared core for the kriging / GPR calibration pipeline.

Contains the GP model classes and the helper functions that are byte-identical
across kriging.py and the Sumatra GEDI scripts (kriging_gedi / kriging_ours_gedi /
kriging_downsampled_gedi). Extracted verbatim; no behavioural changes.
"""

import numpy as np
import datetime as dt
from rasterio.crs import CRS
import geopandas as gpd
import torch
from os.path import join
from rasterio.transform import AffineTransformer
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt
import argparse
from os.path import join
import pickle
from scipy.spatial import cKDTree
from rasterio.transform import Affine

# ---- shared constants ----
NODATAVALS = {'S2' : 0, 'CH': 255, 'ALOS': 0, 'LC': 255, 'DEM': -9999, 'LC': 255}

def equals(x, nodata) :
    """
    This function checks if the value x is equal to the nodata value, taking into account the case where nodata is NaN.

    Args:
    - x (float): The value to check.
    - nodata (float): The nodata value. If it is NaN, the function will check if x is NaN.

    Returns:
    - bool: True if x is equal to nodata, False otherwise.
    """
    if np.isnan(nodata) : return np.isnan(x)
    else: return x == nodata


def str2bool(v):
    """ 
    Helper function to parse a string into a boolean.
    """
    if v == 'true': return True
    elif v == 'false': return False
    else: raise argparse.ArgumentTypeError(f"Either 'true' or 'false' expected, got {v}.")


def float_or_str(v) :
    """
    Helper function to parse a string into a float or a string.
    """
    try: return float(v)
    except ValueError: return v


def get_region(s2_tile, path_kriging) :
    """
    This function returns the region corresponding to the given S2 tile.

    Args:
    - s2_tile: str, the S2 tile.
    - path_kriging: str, path to the kriging directory.

    Returns:
    - region: str, the region corresponding to the S2 tile.
    """
    with open(join(path_kriging, 'helper', 'per_tile', 'region_per_tile.pkl'), 'rb') as f:
        region_per_tile = pickle.load(f)
    return region_per_tile[s2_tile]


def get_tile(data, s2_transform, upsampling_shape, data_source, data_attrs) :
    """
    This function extracts the data for the Sentinel-2 L2A product at hand, crops it so as to perfectly match
    the Sentinel-2 tile, resamples it to 10m resolution when necessary, and returns it.

    Args:
    - data: dict, with the attributes as keys and the corresponding 2d arrays as values.
    - s2_transform: affine.Affine, transform of the Sentinel-2 tile.
    - upsampling_shape: tuple of ints, shape of the Sentinel-2 tile.
    - data_source: string, source of the data. One of 'S2', 'CH', 'ALOS', 'LC', 'DEM'.
    - data_attrs: dict, with the attributes as keys and the corresponding data types as values.

    Returns:
    - res: dict, with the attributes as keys and the corresponding 2d arrays as values.
    """

    if data == {} : return None

    # Get the transforms
    s2_transformer, data_transformer = AffineTransformer(s2_transform), AffineTransformer(data['transform'])    

    # Upper left corner
    ul_x, ul_y = s2_transformer.xy(0, 0)
    ul_row, ul_col = data_transformer.rowcol(ul_x, ul_y)
    ul_row, ul_col = int(ul_row), int(ul_col)

    # Lower right corner
    lr_x, lr_y = s2_transformer.xy(upsampling_shape[0] - 1, upsampling_shape[1] - 1)
    lr_row, lr_col = data_transformer.rowcol(lr_x, lr_y)
    lr_row, lr_col = int(lr_row), int(lr_col)

    # Crop the data to the same bounds, padding the data if necessary
    res = {}
    for data_attr in data_attrs.keys() :
        res[data_attr] = crop_and_pad_arrays(data[data_attr], ul_row, lr_row, ul_col, lr_col, invalid = 0 if data_source == 'ALOS' else NODATAVALS[data_source])

        # Resample to 10m resolution if necessary, i.e. for data sources with resolution lower than 10m per pixel
    
        if data_source == 'ALOS' :
            res[data_attr] = upsampling_with_nans(res[data_attr].astype(np.float32), upsampling_shape, NODATAVALS[data_source], 1).astype(data_attrs[data_attr])
        
        if data_source == 'DEM' :
            res[data_attr] = upsampling_with_nans(res[data_attr].astype(np.float32), upsampling_shape, NODATAVALS[data_source], 1).astype(data_attrs[data_attr])
        
        elif data_source == 'LC' :
            res[data_attr] = upsampling_with_nans(res[data_attr], upsampling_shape, NODATAVALS[data_source], 0).astype(data_attrs[data_attr])
        
        assert res[data_attr].shape == upsampling_shape, f'{data_source} | {data_attr} | {data[data_attr].shape} | {res[data_attr].shape} | {upsampling_shape} | {ul_row} | {lr_row} | {ul_col} | {lr_col}'

    return res


def get_CRS_from_S2_tilename(tname) :
    """
    Get the CRS of the Sentinel-2 tile from its name. The tiles are named as DDCCC (where D is a digit and C a character).
    MGRS tiles are in UTM projection, which means the CRS will be EPSG=326xx in the Northern Hemisphere, and 327xx in the
    Southern. The first character of the tile name gives you the hemisphere (C to M is South, N to X is North); and the
    two digits give you the UTM zone number.

    Args:
    - tname: str, name of the Sentinel-2 tile

    Returns:
    - rasterio.crs.CRS, the CRS of the Sentinel-2 tile
    """

    tile_code, hemisphere = tname[:2], tname[2]

    if 'C' <= hemisphere <= 'M':
        crs = f'EPSG:327{tile_code}'
    elif 'N' <= hemisphere <= 'X':
        crs = f'EPSG:326{tile_code}'
    else:
        raise ValueError(f'Invalid hemisphere code: {hemisphere}')
    
    return CRS.from_string(crs)


def get_S2_bounds(tile_name, path_shp) :
    """
    Get the bounds of a Sentinel-2 tile from its name.

    Args:
    - tile_name: str, name of the Sentinel-2 tile.
    - path_shp: str, path to the shapefile containing the Sentinel-2 grid.

    Returns:
    - tile_geom: shapely.geometry.Polygon, the geometry of the Sentinel-2 tile.
    """

    # Read the Sentinel-2 grid shapefile
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio')

    # Get the geometry of the tile
    tile_geom = grid_df[grid_df['Name'] == tile_name]['geometry'].values[0]

    return tile_geom


def crop_and_pad_arrays(data, ul_row, lr_row, ul_col, lr_col, invalid):
    """
    This function crops (and pads if necessary) the data to match the shape provided by
    the upper left and lower right indices.

    Args:
    - data: 2d array, data.
    - (ul_row, ul_col) : tuple of ints, indices of the pixel corresponding to the upper
        left corner of the Sentinel-2 tile.
    - (lr_row, lr_col) : tuple of ints, indices of the pixel corresponding to the lower
        right corner of the Sentinel-2 tile.
    - invalid: int/float, value to use for the padding.
    
    Returns:
    - data: 2d array, cropped (and padded) data.
    """

    # Get the dimensions of the arrays
    height, width = data.shape

    # If any of the slicing indices are out of bounds, pad with zeros
    if ul_row < 0 or lr_row >= height or ul_col < 0 or lr_col >= width:

        print('(padding)')

        # Calculate the new shape after padding
        new_height = lr_row - ul_row + 1
        new_width = lr_col - ul_col + 1

        # Create new arrays to store the padded data
        padded_data = np.full(shape = (new_height, new_width), fill_value = invalid, dtype = data.dtype)

        # Compute the region of interest in the new padded arrays
        start_row = max(0, -ul_row)
        end_row = min(height - ul_row, lr_row - ul_row + 1)
        start_col = max(0, -ul_col)
        end_col = min(width - ul_col, lr_col - ul_col + 1)

        # Copy the original data to the new padded arrays
        padded_data[start_row : end_row, start_col : end_col] = data[max(0, ul_row) : min(height, lr_row + 1), max(0, ul_col) : min(width, lr_col + 1)]

        # Update the variables to point to the new padded arrays
        data = padded_data

    # Otherwise, simply perform the slicing operation
    else: data = data[ul_row : lr_row + 1, ul_col : lr_col + 1]

    return data


def fill_nan_with_nearest(image, nan_mask):
    """
    This function fills the NaN values in the image with the nearest non-NaN value.

    Args:
    - image: 2d array, image with NaN values.
    - nan_mask: 2d array, mask of the NaN values in the image.

    Returns:
    - filled_image: 2d array, image with NaN values filled.
    """
    
    indices = distance_transform_edt(nan_mask, return_distances = False, return_indices = True)
    filled_image = image[tuple(indices)]
    
    return filled_image


def upsampling_with_nans(image, upsampling_shape, nan_value, order) :
    """
    This function upsamples the image to the `upsampling_shape`, and fills the NaN values with the nearest non-NaN value.

    Args:
    - image: 2d array, image to upsample.
    - upsampling_shape: tuple of ints, shape of the upsampled image.
    - nan_value: int, value to use for the NaN values.
    - order: int, order of the interpolation.
        order = 0 : nearest neighbor interpolation
        order = 1 : bilinear interpolation
        order = 2 : bi-quadratic interpolation
        order = 3 : bicubic interpolation
        cf. https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.warp

    Returns:
    - upsampled_image_with_nans: 2d array, upsampled image with NaN values filled.
    """

    # Check that there are no inf values in the data
    assert not np.isinf(image).any(), 'There are inf values in the data.'

    # Create a mask for the non-defined values
    if np.isnan(nan_value) : nan_mask = np.isnan(image)
    else: nan_mask = (image == nan_value)

    # If there are no undefined values, simply resize
    if np.count_nonzero(nan_mask) == 0 :
        return resize(image, upsampling_shape, order = order, mode = 'edge', preserve_range = True)

    # Otherwise, take care of the undefined values
    else:

        # In the original image, fill the NaN values with the nearest non-NaN value
        non_nan_image = fill_nan_with_nearest(image, nan_mask)

        # Upsample the original image
        upsampled_image = resize(non_nan_image, upsampling_shape, order = order, mode = 'edge', preserve_range = True)

        # Upsample the NaN mask
        upsampled_nan_mask = resize(nan_mask.astype(float), upsampling_shape, order = 0, mode = 'edge') > 0.5

        # Replace the NaN values in the upsampled image with NaN
        upsampled_image_with_nans = np.where(upsampled_nan_mask, nan_value, upsampled_image)

        return upsampled_image_with_nans


def zero_first_two(grad):
    grad = grad.clone()
    grad[0, :2] = 0
    return grad






def get_furthest_neighbor(gdf, height, width, second_max = False, unit = 500, norm_coords = False) :

    points_mask = np.zeros((height, width), dtype = np.uint8)
    points_mask[gdf.row_idx, gdf.col_idx] = 1
    _, indices = distance_transform_edt(points_mask == 0, return_distances=True, return_indices=True)

    # index of nearest point along each axis
    nearest_row = indices[0]  # row index of nearest point
    nearest_col = indices[1]  # col index of nearest point

    # Now compute per-axis distances in pixels
    delta_row = nearest_row - np.arange(points_mask.shape[0])[:, None]
    delta_col = nearest_col - np.arange(points_mask.shape[1])[None, :]

    # Maximum along each axis
    max_dx = delta_row.max()
    max_dy = delta_col.max()

    if second_max :
        delta_row[delta_row == max_dx] = 0
        delta_col[delta_col == max_dy] = 0
        max_dx = delta_row.max()
        max_dy = delta_col.max()

    # Make positive
    max_dx = np.abs(max_dx)
    max_dy = np.abs(max_dy)

    # Round up to nearest integer
    max_dx = np.ceil(max_dx / unit) * unit
    max_dy = np.ceil(max_dy / unit) * unit

    print("new lengthscales before scaling:", max_dx, max_dy)

    # If coordinates are normalized, scale accordingly
    if norm_coords :
        max_dx = max_dx / 10980
        max_dy = max_dy / 10980

    return max_dx, max_dy


def get_furthest_neighbor_v0(gdf, unit = 1000) :
    """
    This function calculates the distance to the furthest nearest neighbor in a geopandas dataframe.

    Args:
    - gdf: geopandas dataframe, the input data.

    Returns:
    - max_dx: float, maximum distance (in pixels) in the x direction to the nearest neighbor, rounded to the nearest multiple of `unit`.
    - max_dy: float, maximum distance (in pixels) in the y direction to the nearest neighbor, rounded to the nearest multiple of `unit`.
    """

    assert gdf.crs.is_projected, "The input geopandas dataframe must be in a projected coordinate system."
    
    # Extract coordinates
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    # Build KDTree
    tree = cKDTree(coords)
    # Query the two nearest points (itself + nearest neighbor)
    distances, indices = tree.query(coords, k=2)
    # Indices of each point's nearest neighbor
    nearest_idx = indices[:, 1]
    # Compute Δx and Δy to the nearest neighbor
    dx = coords[nearest_idx, 0] - coords[:, 0]
    dy = coords[nearest_idx, 1] - coords[:, 1]
    # Get maximum per-axis displacements
    max_dx = np.max(np.abs(dx)) / 10
    max_dy = np.max(np.abs(dy)) / 10
    # Round up to the closest multiple of unit
    max_dx = np.ceil(max_dx / unit) * unit
    max_dy = np.ceil(max_dy / unit) * unit
    return max_dx, max_dy


def filter_GEDI_dates(GEDI, year) :
    """
    This function filters the GEDI data to have only the footprints from the year of interest.
    The `date` attribute of the GEDI data is the number of days since the beginning of the 
    GEDI mission, launched on April 17th, 2019.

    Args:
    - GEDI: geopandas dataframe, GEDI data.
    - year: string, year of interest.

    Returns:
    - GEDI: geopandas dataframe, GEDI data.
    """

    start_of_mission = dt.datetime.strptime('2019-04-17', '%Y-%m-%d')
    first_day_year = dt.datetime.strptime(f'{year}-01-01', '%Y-%m-%d')
    last_day_year = dt.datetime.strptime(f'{year}-12-31', '%Y-%m-%d')
    min_num_days = max((first_day_year - start_of_mission).days, 0)
    max_num_days = (last_day_year - start_of_mission).days

    return GEDI[(GEDI['date'] >= min_num_days) & (GEDI['date'] <= max_num_days)]




def get_tile_data(pred_agb, dem, pred_std, extra_ft, norm_values, coords, norm_coords, aux, norm_aux, pred_vals, extra_features) :
    """
    This function prepares the input data for the GP model for a given tile, including normalization and feature stacking.

    Args:
    - pred_agb: numpy array, predicted AGB values for the tile.
    - dem: numpy array, DEM values for the tile.
    - pred_std: numpy array, predicted standard deviation values for the tile.
    - extra_ft: numpy array, extra features for the tile.
    - norm_values: dict, dictionary with normalization values.
    - coords: bool, whether to include spatial coordinates as input features.
    - aux: str, auxiliary variable to use ('none', 'DEM', or 'STD').
    - pred_vals: bool, whether to include the predicted values as input features.
    - extra_features: str, whether to include extra features ('none' or other).

    Returns:
    - tile_x: torch.Tensor, the prepared input data for the tile.
    """

    tile_x = []

    # coordinates
    if coords : 
        xx, yy = np.meshgrid(range(pred_agb.shape[1]), range(pred_agb.shape[0]))
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        if norm_coords : # min max normalization of the coordinates
            xx = (xx - norm_values['coords_X']['min']) / (norm_values['coords_X']['max'] - norm_values['coords_X']['min'])
            yy = (yy - norm_values['coords_Y']['min']) / (norm_values['coords_Y']['max'] - norm_values['coords_Y']['min'])
        tile_x.extend([xx[:, None], yy[:, None]])
    

    # auxiliary variable
    if aux != 'none' :
        if aux == 'STD' : 
            zz = pred_std.reshape(-1)
        if norm_aux == 'min_max' : zz = (zz - norm_values[aux]['min']) / (norm_values[aux]['max'] - norm_values[aux]['min'])
        tile_x.append(zz[:, None])


    # predicted values
    if pred_vals :
        zz = pred_agb.reshape(-1)
        if norm_aux == 'min_max' : zz = (zz - norm_values['preds']['min']) / (norm_values['preds']['max'] - norm_values['preds']['min'])
        tile_x.append(zz[:, None])

    # extra features
    if extra_features[0] != 'none' :
        zz = extra_ft.reshape(-1, extra_ft.shape[-1])
        if norm_aux == 'min_max' : zz = (zz - norm_values['EFT']['min']) / (norm_values['EFT']['max'] - norm_values['EFT']['min'])
        tile_x.append(zz)
    
    tile_x = torch.from_numpy(np.concatenate(tile_x, axis = 1)).float()
    return tile_x


def get_train_val_data(X, Y, X_val, Y_val, predictions, predictions_val, std, std_val, eft, eft_val, aux, norm_aux, norm_coords, coords, pred_vals, norm_values, residuals, residuals_val) :
    """
    This function prepares the training and validation data for the GP model, including normalization and feature stacking.

    Args:
    - (X, Y, X_val, Y_val): numpy arrays, coordinates of the training and validation points.
    - (predictions, predictions_val): numpy arrays, predictions at the training and validation points.
    - (std, std_val): numpy arrays, standard deviation at the training and validation points.
    - (eft, eft_val): numpy arrays, extra features at the training and validation points.
    - aux: str, auxiliary variable to use ('none', 'DEM', or 'STD').
    - norm_aux: str or None, normalization method for the auxiliary variable ('mean_std', 'min_max', or None).
    - norm_coords: bool, whether to normalize the coordinates with min-max scaling.
    - mean_function: str, type of mean function to use ('constant' or 'linear').
    - coords: bool, whether to use spatial coordinates as input features.
    - pred_vals: bool, whether to use the predictions as input features.
    - norm_values: dict, dictionary to store normalization values.
    - (residuals, residuals_val): numpy arrays, residuals at the training and validation points.

    Returns:
    - train_x, train_y, val_x, val_y: torch.Tensors, training and validation inputs and targets.
    - norm_values: dict, updated dictionary with normalization values.

    """

    train_x, train_y = [], []
    val_x, val_y = [], []
    
    # coordinates
    if coords : train_x.append(X[:, None]), train_x.append(Y[:, None]), val_x.append(X_val[:, None]), val_x.append(Y_val[:, None])
    
    # auxiliary variable: DEM or STD
    elif aux == 'STD': train_x.append(std[:, None]), val_x.append(std_val[:, None])
    elif aux == 'none' : pass

    # predicted AGB values (that we need to process on the fly)
    if pred_vals :
        if norm_aux :
            if norm_aux == 'min_max' :
                _preds_min, _preds_max = np.min(predictions), np.max(predictions)
                norm_values['preds'] = {'min': _preds_min, 'max': _preds_max}
                predictions = (predictions - _preds_min) / (_preds_max - _preds_min)
                predictions_val = (predictions_val - _preds_min) / (_preds_max - _preds_min)
        train_x.append(predictions[:, None]), val_x.append(predictions_val[:, None])

    # extra features
    if eft is not None : train_x.append(eft), val_x.append(eft_val)

    # Stack all features
    train_x = torch.from_numpy(np.concatenate(train_x, axis=1)).float().cuda()
    val_x = torch.from_numpy(np.concatenate(val_x, axis=1)).float().cuda()
    train_y = torch.from_numpy(residuals).float().cuda()
    val_y = torch.from_numpy(residuals_val).float().cuda()
    
    return train_x, train_y, val_x, val_y, norm_values






from kriging.gp import ExactGPModel  # GP model (kept in kriging/gp.py; re-exported here)







__all__ = ['ExactGPModel', 'NODATAVALS', 'crop_and_pad_arrays', 'equals', 'fill_nan_with_nearest', 'filter_GEDI_dates', 'float_or_str', 'get_CRS_from_S2_tilename', 'get_S2_bounds', 'get_furthest_neighbor', 'get_furthest_neighbor_v0', 'get_region', 'get_tile', 'get_tile_data', 'get_train_val_data', 'str2bool', 'upsampling_with_nans', 'zero_first_two']
