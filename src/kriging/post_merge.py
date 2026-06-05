"""

This script is used to merge the post-Kriging Sentinel-2 tile level predictions at the ESA tile level.

"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
import argparse
from os.path import join, basename
import geopandas as gpd
from kriging.kriging import str2bool
from kriging.kriging import filter_GEDI_dates
import rasterio as rs
from rasterio.merge import merge
import numpy as np
from pyproj import CRS
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import transform as window_transform
import glob
from os import makedirs
from os.path import isdir, isfile
import pandas as pd
import pickle
import gc
import psutil
import os
from shapely import union_all
from shapely.geometry import box
from rasterio import windows
from rasterio.transform import from_bounds

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


process = psutil.Process(os.getpid())
def print_current_RAM() :
    mem = process.memory_info().rss  # in bytes
    print(f"RAM usage: {mem / (1024**3):.2f} GB")

def gaus2d(x = 0, y = 0, mx = 0, my = 0, sx = 1, sy = 1): 
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def offset_coord(coord, lat_avg, coord_type, sign, offset) :
    """
    This function offsets the coordinate (coord) by a given offset (in meters).

    Args:
    - coord: float, coordinate to offset.
    - lat_avg: float, average latitude of the tile.
    - coord_type: string, type of coordinate. Either 'lat' or 'lon'.
    - sign: int, sign of the offset. Either 1 or -1.
    - offset: int, offset in meters.

    Returns:
    - coord: float, offsetted coordinate.
    """
    
    # Calculate the offsetted coordinates
    # cf. https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-latitude-longitude-by-some-amount-of-meters
    R = 6371000 # Earth's radius

    if coord_type == 'lat' :
        dLat = sign * offset / R
        coord = (coord + dLat * 180 / np.pi)
        coord = (coord + 90) % 180 - 90
    
    elif coord_type == 'lon' :
        dLon = sign * offset / (R * np.cos(np.pi * lat_avg / 180))
        coord = (coord + dLon * 180 / np.pi)
        coord = (coord + 180) % 360 - 180
    
    else: raise ValueError(f"coord_type {coord_type} not recognized. Use either 'lat' or 'lon'.")
    
    return coord

def buffered_geometry(geometry, offset = 5000) :
    """
    This function buffers the geometry by a given offset (in meters).

    Args:
    - geometry: shapely.geometry.Polygon, geometry to buffer.
    - offset: int, offset in meters.

    Returns:
    - geometry: shapely.geometry.Polygon, buffered geometry.
    """

    # Get the bounding coordinates of the geometry
    lon_min, lat_min, lon_max, lat_max = geometry.bounds

    # Average latitude of the tile
    lat_avg = (lat_min + lat_max) / 2
    
    # Offset the coordinates by the given offset (in meters)
    lon_min = offset_coord(lon_min, lat_avg, 'lon', -1, offset)
    lon_max = offset_coord(lon_max, lat_avg, 'lon', 1, offset)
    lat_min = offset_coord(lat_min, lat_avg, 'lat', -1, offset)
    lat_max = offset_coord(lat_max, lat_avg, 'lat', 1, offset)

    # Make a new geometry with the new coordinates
    geometry = box(lon_min, lat_min, lon_max, lat_max)

    return geometry


def process_memfile(s2_path, border, target_crs, esa_bounds, method, tmp_dir) :

    with rs.open(s2_path) as src :
        src_crs = src.crs
        reproj = (src_crs != target_crs)
        num_bands = src.count
        assert num_bands == 3, f"Expected 3 bands (AGB, residuals, STD), but got {num_bands}."
        nodata = src.nodata
        if np.isnan(nodata) : nodata = np.nan
        profile = src.meta.copy()
        height, width = src.height, src.width

        # Crop the borders ###############################################
        if border > 0 :
            # Calculate the new window
            crop_width = width - 2 * border
            crop_height = height - 2 * border
            window = Window(border, border, crop_width, crop_height)
            # Read cropped data
            data = src.read(window = window)
            assert data.shape[1] == crop_height and data.shape[2] == crop_width, f"Expected data shape ({crop_height}, {crop_width}), but got {data.shape[1:3]}."
            # Get new transform and bounds
            crop_transform = src.window_transform(window)
            crop_bounds = src.window_bounds(window)
            # Update metadata
            profile.update({"height": crop_height, "width": crop_width, "transform": crop_transform})
        else:
            data = src.read()
            crop_transform = src.transform
            crop_bounds = src.bounds
            crop_width, crop_height = width, height

    # Reproject the data to the target CRS ###############################
    if reproj :
        transform, width, height = calculate_default_transform(src_crs, target_crs, crop_width, crop_height, *crop_bounds)
        profile.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
        dest = np.full(shape = (num_bands, height, width), fill_value = nodata, dtype = data.dtype)
        reproject(
            source = data,
            destination = dest,
            src_transform = crop_transform,
            src_crs = src_crs,
            dst_transform = transform,
            dst_crs = target_crs,
            dst_nodata = nodata,
            resampling = Resampling.bilinear
        )
    else:
        dest = data
        transform = crop_transform

    del data
    gc.collect()


    # Calculate weights ##################################################
    _, src_height, src_width = dest.shape

    if method == 'gaussian' :
        xmin, xmax = - src_height // 2, src_height // 2
        ymin, ymax = - src_width // 2, src_width // 2
        x = np.linspace(xmin, xmax, src_width)
        y = np.linspace(ymin, ymax, src_height)
        x, y = np.meshgrid(x, y)
        sx = (x.max() - x.min()) / 5
        sy = (y.max() - y.min()) / 5
        weights = gaus2d(x, y, sx=sx, sy=sy)
    elif method == 'cosine' :
        floor = 0.05
        wx = floor + (1 - floor) * 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, 1, src_width)))
        wy = floor + (1 - floor) * 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, 1, src_height)))
        weights = np.outer(wy, wx)
    else: raise NotImplementedError(f"Weighting method '{method}' not implemented.")

    # Multiply each dimension of dest by the weights
    res = np.full(shape = (3, src_height, src_width), fill_value = nodata, dtype = dest.dtype)
    mask = equals(dest[0, :, :], nodata) | equals(dest[2, :, :], nodata)
    res[0, :, :] = dest[0, :, :] * weights # first dimension: AGB
    res[1, :, :] = dest[2, :, :] * weights # third dimension: STD
    res[2, :, :] = weights

    del dest, weights
    gc.collect()

    # Where there are nodata values, set the corresponding values and weights to 0
    res[0, mask] = nodata
    res[1, mask] = nodata
    res[2, mask] = nodata

    del mask
    gc.collect()

    # Crop to ESA bounds ##########################################
    window = windows.from_bounds(*esa_bounds, transform = transform)
    row_off, col_off, height, width = int(np.floor(window.row_off)), int(np.floor(window.col_off)), int(np.ceil(window.height)), int(np.ceil(window.width))
    r0, c0, r1, c1 = max(0, row_off), max(0, col_off), min(src_height, row_off + height), min(src_width, col_off + width)
    if r0 >= r1 or c0 >= c1 :
        print("ESA bounds do not intersect with the S2 tile bounds after reprojection. Skipping this tile.")
        return
    windowed_data = res[:, r0 : r1, c0 : c1]
    
    del res
    gc.collect()
    
    int_window = Window(c0, r0, c1 - c0, r1 - r0)
    new_transform = window_transform(int_window, transform)
    profile.update({'height': windowed_data.shape[1], 'width': windowed_data.shape[2], 'transform': new_transform, 'count': 3})

    if not isdir(tmp_dir) : makedirs(tmp_dir, exist_ok = True)
    fname = join(tmp_dir, f'temp_{basename(s2_path)}')
    with rs.open(fname, 'w', **profile) as dst:
        dst.write(windowed_data)
    
    return fname


def calculate_metrics(true_agb, pred_agb) :

    # First, we calculate the overall metrics
    diff = pred_agb - true_agb
    results = {
        'ME': np.mean(diff),
        'MAE': np.mean(np.abs(diff)),
        'RMSE': np.sqrt(np.mean(np.pow(diff, 2))),
        'N': len(true_agb)
    }

    # Then, we calculate the binned metrics
    bins = np.arange(0, 501, 50)
    lbs, ubs = bins[:-1], bins[1:]
    binned_results = {'ME': {}, 'MAE': {}, 'RMSE': {},  'N': {}}
    for lb, ub in zip(lbs, ubs) :
        mask = (true_agb >= lb) & (true_agb < ub)
        if np.sum(mask) == 0: continue
        binned_true = true_agb[mask]
        binned_pred = pred_agb[mask]
        binned_diff = binned_pred - binned_true
        binned_results['ME'][f'{lb}-{ub}'] = np.mean(binned_diff)
        binned_results['MAE'][f'{lb}-{ub}'] = np.mean(np.abs(binned_diff))
        binned_results['RMSE'][f'{lb}-{ub}'] = np.sqrt(np.mean(np.pow(binned_diff, 2)))
        binned_results['N'][f'{lb}-{ub}'] = len(binned_true)

    results['binned'] = binned_results
    return results, diff


def compute_metrics(path_GEDI, epsg4326_esa_geom, year, target_crs, save_path) :
    """
    This function computes the metrics between the post-Kriging predictions and the GEDI footprints.
    It loads the GEDI footprints within the ESA tile's geometry, reprojects them to the target CRS,
    finds the correspondance between the GEDI footprints and the predictions, and computes the metrics.

    Args:
    - path_GEDI (str): Path to the GEDI footprints file.
    - epsg4326_esa_geom (geometry): The geometry of the ESA tile in EPSG:4326.
    - year (int): The year for which to compute the metrics.
    - target_crs (CRS): The target CRS to which the GEDI footprints will be reprojected.
    - save_path (str): Path to the post-Kriging predictions

    Returns:
    - None: The results are saved to a file.
    """

    print('Computing metrics...')

    # Load the GEDI footprints within the geometry
    print('Loading GEDI data...')
    GEDI = gpd.read_file(path_GEDI, engine = 'pyogrio', bbox = epsg4326_esa_geom.bounds)
    assert GEDI.crs.to_string() == 'EPSG:4326', "The GEDI geometries must be in EPSG:4326 for metrics computation."
    GEDI = GEDI[GEDI.intersects(epsg4326_esa_geom)]
    GEDI_year = filter_GEDI_dates(GEDI, year)
    if GEDI_year.empty or (len(GEDI_year) < 100) : 
        print(f'No GEDI footprints for the year {year} in the tile {s2_tile}. Using footprints from other timesteps instead.')
        del GEDI_year
    else: GEDI = GEDI_year
    print('done.')
    print_current_RAM()

    # Reproject GEDI footprints to the target CRS
    GEDI = GEDI.to_crs(target_crs)

    # Find the correspondance between the GEDI footprints and the predictions
    print('Getting correspondance...')
    with rs.open(save_path) as src:
        width, height = src.width, src.height
        def get_idx(geom, src) :
            lon, lat = geom.x, geom.y
            row_index, col_index = src.index(lon, lat)
            return row_index, col_index
        GEDI[['row_idx', 'col_idx']] = GEDI.apply(lambda row: get_idx(row['geometry'], src), axis = 1).apply(pd.Series)
        agb_merged = src.read(1)
        nodata = src.nodata
    print('done.')
    print_current_RAM()
    # Filter the values that are outside of the width/height of the prediction
    print(f'Filtering GEDI footprints outside of the prediction bounds ({height} x {width})...')
    GEDI = GEDI[(GEDI['row_idx'] < height) & (GEDI['row_idx'] >= 0) & (GEDI['col_idx'] < width) & (GEDI['col_idx'] >= 0)]
    print('done.')
    print_current_RAM()
    
    # Remove the values where either the prediction is not defined
    valid_mask = ~equals(agb_merged, nodata)
    if np.count_nonzero(~valid_mask) > 0 :
        print('Removing GEDI footprints with no corresponding prediction...')
        GEDI = GEDI[valid_mask[GEDI['row_idx'], GEDI['col_idx']]]
        assert GEDI.shape[0] > 0, "No GEDI footprints left after filtering."
    del valid_mask
    gc.collect()
    print('done.')
    print_current_RAM()

    # Get the predictions
    print('Getting predictions...')
    og_Y, og_X = GEDI['row_idx'].values, GEDI['col_idx'].values
    og_gedi_agb = GEDI['agbd'].values
    predictions = agb_merged[og_Y, og_X]
    print('done.')
    del agb_merged, og_Y, og_X, GED
    gc.collect()
    print_current_RAM()
    
    # Compute the tile's metrics, such that we can later on compute the metrics for the whole coverage
    print('Calculating metrics...')
    results, residuals = calculate_metrics(og_gedi_agb, predictions)
    print('done.')
    print_current_RAM()

    # Save the results to file
    with open(join(save_path.replace('.tif', '_metrics.pkl')), 'wb') as f: pickle.dump(results, f)
    with open(join(save_path.replace('.tif', '_residuals.pkl')), 'wb') as f: pickle.dump(residuals, f)
    print('done!')


def resize_merged_data(agb_merged, std_merged, meta, ESA_shape, ESA_bounds):
    """
    This function resizes the merged data to match the shape of the ESA mask.

    Args:
    - agb_merged (np.ndarray): The merged AGB data.
    - std_merged (np.ndarray): The merged standard deviation data.
    - meta (dict): The metadata of the merged data.
    - ESA_shape (tuple): The shape of the ESA mask.
    - ESA_bounds (tuple): The bounds of the ESA mask.

    Returns:
    - dst_data (np.ndarray): The resized merged data.
    - dst_profile (dict): The updated metadata of the resized merged data.
    """
   
    target_height, target_width = ESA_shape
    src_data = np.asarray([agb_merged, std_merged])
    src_transform = meta['transform']
    src_crs = meta['crs']
    
    # Spatial resampling
    dst_transform = from_bounds(*ESA_bounds, width=target_width, height=target_height)
    dst_data = np.empty((2, target_height, target_width), dtype=src_data.dtype)
    reproject(
        source=src_data, #data_windowed,
        destination=dst_data,
        src_transform=src_transform, #window_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        resampling=Resampling.bilinear
    )

    # Update profile
    dst_profile = meta.copy()
    dst_profile.update({
        'height': target_height,
        'width': target_width,
        'transform': dst_transform
    })

    return dst_data, dst_profile


def apply_ESA_mask(path_mask, year, tile_name, target_crs, agb_merged, std_merged, meta) :
    """
    This function applies the ESA WorldCover mask to the merged data. If necessary, it
    reprojects the ESA mask to the target CRS and resizes the merged data to match the
    shape of the ESA mask.

    Args:
    - path_mask (str): Path to the ESA WorldCover mask files.
    - year (int): Year of the ESA WorldCover mask.
    - tile_name (str): Name of the ESA tile.
    - target_crs (CRS): Target CRS to which the ESA mask will be reprojected.
    - agb_merged (np.ndarray): Merged AGB data.
    - std_merged (np.ndarray): Merged standard deviation data.
    - meta (dict): Metadata of the merged data.

    Returns:
    - agb_merged (np.ndarray): Merged AGB data with ESA mask applied.
    - std_merged (np.ndarray): Merged standard deviation data with ESA mask applied.
    - meta (dict): Updated metadata of the merged data.
    """

    # Find the file
    print('Loading ESA WorldCover mask...')
    if isfile(join(path_mask, f'ESA_WorldCover_10m_{year}_v100_{tile_name}_Map.tif')):
        fpath = join(path_mask, f'ESA_WorldCover_10m_{year}_v100_{tile_name}_Map.tif')
    else: 
        print(f'ESA WorldCover mask not found for tile {tile_name}.')
        return agb_merged, std_merged
    
    # Read the data
    with rs.open(fpath) as src:
        worldcover = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_bounds = src.bounds
        src_height, src_width = src.height, src.width
    
    # Reproject the ESA mask to the target CRS
    if src_crs != target_crs :
        print(f'Reprojecting ESA WorldCover mask from {src_crs} to {target_crs}...')
        transform, width, height = calculate_default_transform(src_crs, target_crs, src_width, src_height, *src_bounds)
        reproj_worldcover = np.zeros((height, width), np.uint8)
        reproject(
            worldcover,
            reproj_worldcover,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear)
        reproj_esa_bounds = array_bounds(height, width, transform)
    else:
        reproj_worldcover = worldcover
        transform = src_transform
        height, width = src_height, src_width
        reproj_esa_bounds = src_bounds
    del worldcover
    gc.collect()

    # If the shape of the ESA mask does not match the shape of the merged data, resize the latter
    if agb_merged.shape != reproj_worldcover.shape:
        print('Resizing agb_merged and std_merged to match the ESA mask shape...')
        print(agb_merged.shape, reproj_worldcover.shape)
        resized_data, meta = resize_merged_data(agb_merged, std_merged, meta, reproj_worldcover.shape, reproj_esa_bounds)
        agb_merged, std_merged = resized_data[0, :, :], resized_data[1, :, :]
        del resized_data
    
    # Mask out permanent water bodies (80) or nodata (0) or built-up (50) or snow and ice (70)
    esa_mask = (reproj_worldcover == 80) | (reproj_worldcover == 0) | (reproj_worldcover == 50) | (reproj_worldcover == 70)
    if np.isnan(meta['nodata']) : nodata = np.nan
    else: nodata = meta['nodata']
    agb_merged[esa_mask] = nodata
    std_merged[esa_mask] = nodata
    del reproj_worldcover, esa_mask
    gc.collect()
    return agb_merged, std_merged, meta

def recasting(_dtype, meta, nodata, agb_merged, std_merged) :

    assert _dtype == 'uint16', f"Invalid dtype: {_dtype} (only 'uint16' and 'float32' are supported)."
    print(f'Casting to {_dtype}...')
    
    # Set values
    dtype = np.uint16
    nodataval = 65535
    meta.update({'dtype' : dtype, 'nodata' : nodataval})
    
    # Cast the NaN values to nodata
    if np.isnan(nodata) : 
        nodata_mask = np.isnan(agb_merged)
        std_nodata_mask = np.isnan(std_merged)
    else: 
        nodata_mask = (agb_merged == nodata)
        std_nodata_mask = (std_merged == nodata)
    agb_merged[nodata_mask] = nodataval
    std_merged[std_nodata_mask] = nodataval
    
    # Cast to uint16
    agb_merged[agb_merged > nodataval] = nodataval
    std_merged[std_merged > nodataval] = nodataval
    agb_merged[agb_merged < 0] = 0

    return agb_merged.astype(dtype), std_merged.astype(dtype), meta



def parser():
    """ 
    Main function. Returns an `ArgumentParser()` object containing the command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--ESA', type = str, required = True, help = 'Path to the file with the ESA geometries.')
    parser.add_argument('--mask', type = str, required = True, help = 'Path to the file with the ESA mask.')
    parser.add_argument('--GEDI', type = str, required = True, help = 'Path to the file with the GEDI geometries.')
    parser.add_argument('--S2', type = str, required = True, help = 'Path to the file with the S2 geometries.')
    parser.add_argument('--AOIs', type = str, required = True, help = 'Path to the file with the AOIs geometries.')
    parser.add_argument('--predictions', type = str, required = True, help = 'Path to the post-kriging predictions.')
    parser.add_argument('--inf_model', type = str, required = True, help = 'ID of the inference model.')
    parser.add_argument('--krig_model', type = str, required = True, help = 'ID of the kriging model.')
    parser.add_argument("--tile_name", required = True, type = str, help = 'ESA tile for which to run the merge.')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--buffer_size', type = int, required = True, help = 'Buffer size (in m) around the tile.')
    parser.add_argument('--year', type = int, required = True, help = 'Year for which to merge the predictions.')
    parser.add_argument('--crs', type = int, required = True, help = 'Which CRS to project to.')
    parser.add_argument('--border_crop', type = int, required = True, help = '# of pixels to crop from the border.')
    parser.add_argument('--method', type = str, required = True, help = 'How to merge the neighbors predictions.')
    parser.add_argument('--compute_metrics', type = str2bool, required = True, help = 'Whether to compute metrics or not.')
    parser.add_argument('--tmp_dir', type = str, required = True, help = 'Path to temporary directory for intermediate files.')
    parser.add_argument('--force', type = str2bool, required = True, help = 'Whether to force the merge or not.')
    parser.add_argument('--dtype', type = str, required = True, help = 'Data type of the output raster.')
    args = parser.parse_args()

    return args, args.ESA, args.mask, args.GEDI, args.S2, args.AOIs, args.predictions, args.tile_name, args.year, args.arch, \
            args.buffer_size, args.crs, args.border_crop, args.method, args.compute_metrics, args.inf_model, args.krig_model, \
            args.tmp_dir, args.force, args.dtype



#######################################################################################################################
# Code execution

"""
--ESA /path/to/data/EcosystemAnalysis/Models/Biomes/helper/3x3/esa_worldcover_tiles.geojson
--S2 /path/to/data/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp
--predictions /path/to/data/EcosystemAnalysis/Models/Biomes/kriging/predictions
--crs 8857 (for Equal Earth) 4326 (for WGS84, the ESA one)
"""

if __name__ == '__main__':

    args, path_ESA, path_mask, path_GEDI, path_S2, path_AOIs, path_predictions, tile_name, year, arch, \
        buffer_size, crs, border_crop, method, COMPUTE_METRICS, inf_model, krig_model, tmp_dir, FORCE, _dtype = parser()

    # Process the CRS
    target_crs = CRS.from_user_input(crs)

    # Get the ESA tile's geometry
    esa_gdf = gpd.read_file(path_ESA, engine = 'pyogrio')
    if esa_gdf.crs is None: raise ValueError("The ESA geometries do not have a CRS defined.")
    esa_gdf = esa_gdf[esa_gdf['ESA_names'] == tile_name]
    if esa_gdf.empty: raise ValueError(f"No ESA tile found for tile name: {tile_name}")
    # If we want to later on compute metrics, store the tile's geometry in EPSG:4326
    if COMPUTE_METRICS:
        assert esa_gdf.crs.to_string() == 'EPSG:4326', "The ESA geometries must be in EPSG:4326 for metrics computation."
        epsg4326_esa_geom = esa_gdf.geometry.values[0]
    
    # Check if the ESA tile has already been generated
    save_path = join(path_predictions, 'ESA', inf_model, krig_model, _dtype)
    if not isdir(save_path): makedirs(save_path, exist_ok = True)
    save_path = f'{save_path}/{tile_name}_{year}.tif'
    _exists = False if FORCE else isfile(save_path)

    if _exists :

        # Compute metrics (if needed)
        if COMPUTE_METRICS: compute_metrics(path_GEDI, epsg4326_esa_geom, year, target_crs, save_path)    
        else: print(f"ESA tile {tile_name} already exists. Skipping merge.")

    else: 

        # Check if it's been generated in another dtype
        # that only works if float32 has already been generated, and if we want uint16)
        if _dtype == 'uint16' and isfile(join(path_predictions, 'ESA', inf_model, krig_model, 'float32', f'{tile_name}_{year}.tif')):

            print(f'Found existing file for tile {tile_name} with dtype float32. Recasting to {_dtype}...')

            # Load the existing file
            with rs.open(join(path_predictions, 'ESA', inf_model, krig_model, 'float32', f'{tile_name}_{year}.tif'), 'r') as src : 
                meta = src.meta.copy()
                nodata = src.nodata
                agb_merged = src.read(1)
                std_merged = src.read(2)
            agb_merged, std_merged, meta = recasting(_dtype, meta, nodata, agb_merged, std_merged)
        
            # Save the recasted file
            print('Saving the recasted product...')
            with rs.open(save_path, 'w', **meta) as dst:
                dst.write(agb_merged, 1)
                dst.write(std_merged, 2)
                dst.set_band_description(1, 'AGB')
                dst.set_band_description(2, 'STD')
            print('done!')
            del std_merged, agb_merged
            gc.collect()

        
        else:

            print(f'Starting merge for ESA tile {tile_name}...')

            # Reproject the ESA geometries to the target CRS
            esa_gdf = esa_gdf.to_crs(target_crs)
            esa_geom = esa_gdf.geometry.values[0]
            del esa_gdf
            gc.collect()

            # Intersect with the AOI's geometry
            countries = gpd.read_file(path_AOIs).to_crs(target_crs)
            AOI = ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana']
            countries = countries[countries['name'].isin(AOI)]
            countries = countries[countries.geometry.intersects(esa_geom)]
            if countries.empty: raise ValueError(f"The ESA tile {tile_name} does not intersect with any of the AOIs' geometries.")
            if len(countries) > 1 : raise ValueError(f"The ESA tile {tile_name} intersects with multiple AOIs.")
            aoi_geom = countries.geometry.values[0]
            esa_geom = esa_geom.intersection(aoi_geom)

            # Buffer the ESA tile's geometry
            if buffer_size != 0 :
                unit_name = target_crs.axis_info[0].unit_name
                if unit_name == 'metre' : buff_esa_tile_geom = esa_geom.buffer(buffer_size)
                elif unit_name == 'degree' : buff_esa_tile_geom = buffered_geometry(esa_geom, buffer_size)
                else: raise ValueError(f"Unsupported CRS unit: {unit_name}. Only 'metre' and 'degree' are supported.")
            else: buff_esa_tile_geom = esa_geom

            # Get the names of Sentinel-2 tiles that intersect with the ESA tile
            s2_gdf = gpd.read_file(path_S2, engine = 'pyogrio').drop_duplicates(subset = ['Name'])
            if s2_gdf.crs != target_crs : s2_gdf = s2_gdf.to_crs(target_crs)
            s2_gdf = s2_gdf[s2_gdf.geometry.intersects(esa_geom)]
            if s2_gdf.empty: raise ValueError(f"No Sentinel-2 tiles found intersecting with ESA tile: {tile_name}")
            
            # Check that the union of the available tiles fully covers the ESA tile
            union_geom = union_all(s2_gdf.geometry)
            if not union_geom.covers(esa_geom) :
                raise ValueError(f"The union of the Sentinel-2 tiles does not cover the ESA tile: {tile_name}.")
            if not union_geom.intersects(esa_geom) :
                raise ValueError(f"The union of the Sentinel-2 tiles does not intersect with the ESA tile: {tile_name}.")
            s2_tilenames = s2_gdf['Name'].values.tolist()
            print(f"Found {len(s2_tilenames)} Sentinel-2 tiles intersecting with ESA tile {tile_name}: {s2_tilenames}")

            # Only keep valid S2 tiles
            path_valid = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_2020.txt'
            with open(path_valid, 'r') as f: og_valid_tiles = [t.strip() for t in f.readlines()]
            to_skip="58FEJ 58FEK 59FLB 01GEM 60FXL 58FGG 60FXK 10SDG 59GNQ 58GFN 11SKS 49SET 11SMR 31NCG 35MQN 45RWM 22NCM 30PVS 49SEC 17QQC 17QQF 11SQV 59HQU 37MCT"
            to_skip = to_skip.split(' ')
            valid_tiles = [t for t in og_valid_tiles if t not in to_skip]
            s2_tilenames = [t for t in s2_tilenames if t in valid_tiles]
            print(f"After filtering, {len(s2_tilenames)} Sentinel-2 tiles will be used for merging: {s2_tilenames}")

            # Iterate over the tiles, load the corresponding post-Kriging predictions and reproject them
            memfiles, datasets = [], []
            for s2_tile in s2_tilenames :

                print(f"Processing tile {s2_tile}...")

                # Check if we have data for this tile
                files = glob.glob(join(path_predictions, f'{arch}/{s2_tile}/{year}/{inf_model}/kriging-{krig_model}-*.tif'))
                if not files :
                    print(f">> No predictions found for tile {s2_tile}. Skipping tile.")
                    continue
                s2_path = files[0]

                memfile = join(tmp_dir, f'temp_{basename(s2_path)}')
                if isfile(memfile) :
                    print(f">> Temporary file found for tile {s2_tile}. Using it instead of processing again.")
                    memfiles.append(memfile)
                    datasets.append(rs.open(memfile))
                    continue
                
                # Process the data
                memfile = process_memfile(s2_path, border_crop, target_crs, buff_esa_tile_geom.bounds, method, tmp_dir)
                if memfile is not None:
                    memfiles.append(memfile)
                    datasets.append(rs.open(memfile))
            
            # Merge the datasets, and crop to the ESA tile's geometry (without the buffer)
            print('Merging...')
            nodata = datasets[0].nodata
            if np.isnan(nodata) : nodata = np.nan
            merged, agb_transform = merge(datasets, bounds = buff_esa_tile_geom.bounds, nodata = nodata, method = "sum", target_aligned_pixels = True)
            sum_agb_merged, sum_std_merged, sum_weights = merged[0], merged[1], merged[2]
            del merged
            mask = equals(sum_agb_merged, nodata) | equals(sum_std_merged, nodata) | equals(sum_weights, nodata) | (sum_weights == 0)
            agb_merged = np.divide(sum_agb_merged, sum_weights, where = ~mask)
            agb_merged[mask] = nodata
            agb_merged[np.isnan(agb_merged)] = nodata
            agb_merged[agb_merged < 0] = 0
            std_merged = np.divide(sum_std_merged, sum_weights, where = ~mask)
            std_merged[mask] = nodata
            for ds in datasets: ds.close()
            for memfile in memfiles: os.remove(memfile)
            del datasets, memfiles, mask, sum_agb_merged, sum_std_merged, sum_weights
            gc.collect()
            print('done!')
            
            # Get the ESA mask
            meta = {'transform' : agb_transform, 'height' : agb_merged.shape[0], 'width' : agb_merged.shape[1],
                'count' : 2, 'dtype' : 'float32', 'crs' : target_crs, 'nodata' : nodata, 'driver' : 'GTiff'}
            print('Applying the ESA mask...')
            agb_merged, std_merged, meta = apply_ESA_mask(path_mask, year, tile_name, target_crs, agb_merged, std_merged, meta)
            print('done!')

            # Crop to the geometry, not just the bounds
            print('Cropping to the ESA tile geometry...')
            mask_arr = geometry_mask(geometries = [esa_geom], transform = meta['transform'], invert = False, out_shape=(agb_merged.shape[0], agb_merged.shape[1]))
            nodata = meta['nodata']
            if np.isnan(nodata) : nodata = np.nan
            agb_merged[mask_arr] = nodata
            std_merged[mask_arr] = nodata
            del mask_arr
            gc.collect()
            print('done!')

            # Check if we need to cast to a different data type
            if _dtype != 'float32' : agb_merged, std_merged, meta = recasting(_dtype, meta, nodata, agb_merged, std_merged)

            # Save the merged predictions
            print('Saving the merged product...')
            with rs.open(save_path, 'w', **meta) as dst:
                dst.write(agb_merged, 1)
                dst.write(std_merged, 2)
                dst.set_band_description(1, 'AGB')
                dst.set_band_description(2, 'STD')
            print('done!')
            del std_merged, agb_merged
            gc.collect()
