"""

This script applies a threshold-based correction to the dense AGB predictions of a model using GEDI footprints.
The correction is based on pre-computed statistics of the residuals in different prediction bins.

"""

#######################################################################################################################
# Imports

import numpy as np
import rasterio as rs
import datetime as dt
import pandas as pd
from rasterio.crs import CRS
import geopandas as gpd
from os.path import join
from time import time
import argparse
from os.path import join, isdir
from os import makedirs
import pickle
import random
import os

os.environ['WANDB_INIT_TIMEOUT'] = '300'

#######################################################################################################################
# Helper functions 

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


def str2bool(v):
    """ 
    Helper function to parse a string into a boolean.
    """
    if v == 'true': return True
    elif v == 'false': return False
    else: raise argparse.ArgumentTypeError(f"Either 'true' or 'false' expected, got {v}.")


def parser():
    """ 
    Returns an `ArgumentParser()` object containing the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2_tile', type = str, required = True, help = 'S2 tile.')
    parser.add_argument('--year', type = int, required = True, help = 'Year of the prediction.')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--ens_models', type = str, nargs = '+', required = True, help = 'Models for ensemble STD.')
    parser.add_argument('--path_predictions', type = str, required = True, help = 'Directory with the predictions.')
    parser.add_argument('--path_gedi', type = str, required = True, help = 'Directory with the GEDI footprints.')
    parser.add_argument('--path_geometries', type = str, required = True, help = 'Directory with the S2 tiles geometries.')
    parser.add_argument('--path_kriging', type = str, required = True, help = 'Directory for Kriging.')
    parser.add_argument('--SAVE', type = str2bool, required = True, help = 'Save the metrics.')
    parser.add_argument('--SAVE_preds', type = str2bool, required = True, help = 'Save the corrected predictions.')
    parser.add_argument('--seed', type = int, default = 10, help = 'Random seed.')
    parser.add_argument('--composites', type = str2bool, required = True, help = 'Whether we are loading composites-derived predictions.')
    parser.add_argument('--ood', type = str2bool, required = True, help = 'Whether to remove OOD samples.')
    args = parser.parse_args()
    return args.s2_tile, args.year, args.arch, args.ens_models, args.path_predictions, args.path_gedi, args.path_geometries, args.path_kriging, args.SAVE, args.SAVE_preds, args.seed, args.composites, args.ood


#######################################################################################################################
# Code execution

if __name__ == '__main__':

    # Initialize everything #######################################################################

    # Parse the arguments
    s2_tile, year, arch, ens_models, path_predictions, path_gedi, path_geometries, path_kriging, SAVE, SAVE_preds, _seed, composites, ood = parser()
    pred_model_name='_'.join(ens_models)

    # Set the random seeds for reproducibility
    random.seed(_seed), np.random.seed(_seed)

    # Define paths
    if composites : s2_path = join(path_predictions, f'{s2_tile}_{year}_composite.tif')
    else : s2_path = join(path_predictions, f'{s2_tile}_{year}_AGB_merged.tif')

    # Load and pre-process the data ###############################################################
    time_start = time()
    print('    Loading data...')

    # Load the AGB predictions for the specified tile
    pred_agb, _, pred_mask, meta, transform, upsampling_shape, nodataval = load_dense_preds(s2_path = s2_path, aux = 'STD') # shape (nrows, ncolumns) = (height, width)
    tile_geom = get_S2_bounds(s2_tile, path_geometries)

    # Load the GEDI footprints within the geometry
    GEDI = gpd.read_file(path_gedi, engine = 'pyogrio', bbox = tile_geom.bounds)
    GEDI = GEDI[GEDI.intersects(tile_geom)]
    if GEDI.empty: raise ValueError(f'No GEDI footprints in the tile geometry ({s2_tile}).')

    # Filter by the year of interest
    GEDI_year = filter_GEDI_dates(GEDI, year)
    if GEDI_year.empty or (len(GEDI_year) < 100) : 
        print(f'    No GEDI footprints for the year {year} in the tile {s2_tile}. Using footprints from other timesteps instead.')
    else: GEDI = GEDI_year
    
    # Reproject GEDI footprints to the local CRS
    crs = get_CRS_from_S2_tilename(s2_tile)
    GEDI = GEDI.to_crs(crs)
    print(f'    Number of footprints: {GEDI.shape[0]}')

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

    # If there are multiple footprints in the same pixel, take the median
    print(f'    Number of footprints before groupby: {GEDI.shape[0]}')
    GEDI = GEDI.groupby(['row_idx', 'col_idx'], as_index = False).median(numeric_only = True)
    print(f'    Number of footprints after groupby: {GEDI.shape[0]}\n')

    # Apply the correction to the predictions #####################################################

    og_Y, og_X = GEDI['row_idx'].values, GEDI['col_idx'].values
    og_indices = GEDI['idx'].values
    og_gedi_agb = GEDI['agbd'].values
    predictions = pred_agb[og_Y, og_X]

    # Load the corrections statistics
    if ood : 
        path_corr = join(path_kriging, 'correction', 'corrections_mean_std_0-25.pkl')
        fname = 'results-baseline_mean_std'
    else: 
        path_corr = join(path_kriging, 'correction', 'corrections_all_0-25.pkl')
        fname = 'results-baseline_all'
    with open(path_corr, 'rb') as f:
        corrections = pickle.load(f)

    # Map each pixel value to a bin
    bins = range(0, 501, 25)
    lbs, ubs = bins[:-1], bins[1:]
    bin_map = np.digitize(pred_agb, bins) - 1
    bin_map = np.clip(bin_map, 0, len(corrections['mean']) - 1)

    # Apply the correction (mean or median)
    values = np.array(list(corrections['mean'].values()))
    correction_values = values[bin_map]
    corrected_AGB = pred_agb + correction_values
    corrected_AGB = np.clip(corrected_AGB, 0, None) # AGB cannot be negative

    if SAVE :
        # Path to the output directory
        tif_path = path_kriging
        if not isdir(tif_path) : makedirs(tif_path, exist_ok=True)
        # Save the pre-kriging predictions and GTs at the footprints' locations
        with open(join(tif_path, f"{fname}.pkl"), 'wb') as f:
            pickle.dump({'pre': pred_agb[og_Y, og_X], 'post' : corrected_AGB[og_Y, og_X], 'ref': og_gedi_agb, 'idx': og_indices}, f)

        # Save the dense predictions
        if SAVE_preds :
            start_time = time()
            print('    Saving the corrected prediction...')
            meta.update(count = 2, dtype = 'float32', nodata = np.nan)
            with rs.open(join(tif_path, f"{fname}.tif"), 'w', **meta) as dst:
                # Write the corrected AGB predictions
                dst.write(corrected_AGB, 1)
                dst.set_band_description(1, 'AGB')
                # Write the residuals
                dst.write(correction_values, 2)
                dst.set_band_description(2, 'Residuals')
            print(f'    Done! In {time() - start_time} seconds.')
