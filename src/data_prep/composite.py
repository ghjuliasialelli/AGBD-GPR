"""

This script is used to generate yearly composites for Sentinel-2 products.
For a given Sentinel-2 tile, it lists all available products, unzips them if necessary,
and computes a composite using the specified method (e.g., median). The composite is saved
as a GeoTIFF file, and the products used for the composite are listed in a text file.

"""


#######################################################################################################################
# Imports

import glob
from os.path import join, isfile, exists, basename, dirname, isdir
from zipfile import ZipFile
import zipfile
from os import makedirs, remove
import numpy as np
from rasterio.windows import Window
import rasterio as rs
import argparse
import datetime as dt
import time
from datetime import timedelta
from scipy import stats
import shutil
import xml.etree.ElementTree as ET

#######################################################################################################################
# Helper functions 

# Sentinel-2 L2A bands that we want to use
S2_L2A_BANDS = {'10m' : ['B02', 'B03', 'B04', 'B08'],
                '20m' : ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'],
                '60m' : ['B01', 'B09']}

def str2bool(v):
    """ 
        Helper function to parse a string into a boolean.
        
        input: `v` (str), input string to be parsed
        output: bool
    """
    if v == 'true': return True
    elif v == 'false': return False
    else: raise argparse.ArgumentTypeError(f"Either 'true' or 'false' expected, got {v}.")


def _parser(argv=None) :
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile for which to run the composite.')
    parser.add_argument("--path_s2", required = True, type = str, help = 'Path to the Sentinel-2 products.')
    parser.add_argument("--method", required = True, type = str, choices = ['median'], help = 'Method to use for the composite.')
    parser.add_argument("--mode", required = True, type = str, choices = ['all', 'min'], help = 'Products to use for the composite. "all" uses all products, "min" uses only the minimum set of products.')
    parser.add_argument("--year", required = True, type = int, help = 'Year for which to run the composite.')
    parser.add_argument("--path_txt", required = True, type = str, help = 'Path to the text file with the list of products.')
    parser.add_argument("--window_size", type = int, default = None, help = 'Size of the window to use for processing.')
    parser.add_argument("--clouds", type = str2bool, required = True, help = 'Whether to mask out clouds for the composite.')
    parser.add_argument("--calculate_coverage", type = str2bool, required = True, help = 'Whether to calculate the coverage of the composite.')
    parser.add_argument("--force", type = str2bool, required = True, help = 'Force the computation of the composite, even if it already exists.')
    parser.add_argument("--skip_SCL", type = str2bool, default = False, help = 'Whether to skip the SCL band.')
    args = parser.parse_args(argv)
    return args.tile_name, args.path_s2, args.method, args.mode, args.year, args.path_txt, args.window_size, args.clouds, args.calculate_coverage, args.force, args.skip_SCL


def unzip_l2a(path_s2, s2_prod):
    """
    This function unzips the Sentinel-2 L2A product at hand, extracting only .tif files.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - s2_prod: string, name of the Sentinel-2 L2A product. (ends in .zip)

    Returns:
    - None
    """

    zip_path = join(path_s2, s2_prod + '.zip')
    
    with ZipFile(zip_path, 'r') as zip_ref:
        # Find the index of the folder containing the SAFE files
        namelist = zip_ref.namelist()
        idx = namelist[0].split('/').index(s2_prod + '.SAFE')
        
        for file in namelist:
            if (file.endswith('.tif')) or (file.endswith('.jp2')) or (file.endswith('MTD_MSIL2A.xml')) :
                # Create a new path by slicing off unwanted parts of the path
                parts = file.split('/')
                new_path = join(*parts[idx:])
                
                # Full path to where the file will be extracted
                full_path = join(path_s2, new_path)
                
                # Extract the file to the new path
                makedirs(dirname(full_path), exist_ok=True)
                if not file.endswith('/'):
                    with zip_ref.open(file) as source, open(full_path, 'wb') as target:
                        shutil.copyfileobj(source, target)


def radiometric_offset_values(path_s2, product, offset) :
    """
    This function extracts the BOA_QUANTIFICATION_VALUE and BOA_ADD_OFFSET_VALUES from the
    Sentinel-2 L2A product at hand, and returns them.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - product: string, name of the Sentinel-2 L2A product.
    - offset: int, 1 if the product was acquired after January 25th, 2022; 0 otherwise.

    Returns:
    - None
    """

    # There is a mismatch between the names of the physical bands in the metadata file, and the
    # names of the bands in the IMG_DATA/ folder. This dictionary defines the mapping
    bands_mapping = {'B1': 'B01', 'B2': 'B02', 'B3': 'B03', 'B4': 'B04', 'B5': 'B05', 'B6': 'B06', 'B7': 'B07', \
                    'B8': 'B08', 'B8A': 'B8A', 'B9': 'B09', 'B10': 'B10', 'B11': 'B11', 'B12': 'B12'}

    # Parse the XML file
    tree = ET.parse(f'{join(path_s2, product)}.SAFE/MTD_MSIL2A.xml')
    root = tree.getroot()

    # Get the BOA_QUANTIFICATION_VALUE
    for elem in root.find('.//QUANTIFICATION_VALUES_LIST') :
        if elem.tag == 'BOA_QUANTIFICATION_VALUE' :
            boa_quantification_value = float(elem.text)
            assert boa_quantification_value == 10000, f'BOA_QUANTIFICATION_VALUE is {boa_quantification_value}, should be 10000'
        else: continue

    # Get the physical bands and their ids
    physical_bands = {elem.get('bandId'): elem.get('physicalBand') \
                      for elem in root.find('.//Spectral_Information_List')}

    if offset :
        
        # Check the BOA offset values (should be 1000)
        for elem in root.find('.//BOA_ADD_OFFSET_VALUES_LIST') :
            physical_band = physical_bands[elem.get('band_id')]
            actual_band = bands_mapping[physical_band]
            boa_add_offset_value = np.abs(int(elem.text))
            assert boa_add_offset_value == 1000, f'BOA_ADD_OFFSET_VALUE is {boa_add_offset_value}, should be 1000 | band {actual_band}'


def list_products(s2_tile, path_s2, mode, year, path_txt) :
    """
    This function lists the available Sentinel-2 products for a given tile and year.

    Args:
    - s2_tile (str): Sentinel-2 tile identifier.
    - path_s2 (str): Path to the Sentinel-2 products directory.
    - mode (str): Mode to use for listing products ('all' or 'min').
    - year (int): Year for which to list products.
    - path_txt (str): Path to the text file with the list of products (if mode is 'min').

    Returns:
    - products (list): List of product identifiers.
    """
    if mode == 'all' :
        products = glob.glob(join(path_s2, f'*_{year}*_*_T{s2_tile}_*.zip'))
        products = [basename(p).rstrip('.zip') for p in products]
        print(f'Found {len(products)} products.')
    elif mode == 'min' :
        path = join(path_txt, f'{s2_tile}_{year}.txt')
        if not isfile(path) :
            raise FileNotFoundError(f'File {s2_tile}_{year}.txt not found in {path_txt}.')
        with open(path, 'r') as f: products = [p.strip() for p in f.readlines()]
    else: raise ValueError(f"Mode {mode} not supported")
    assert len(products) > 0, f'No products found for tile {s2_tile} in year {year}.'
    return products


def unzip_all(products, path_s2) :
    """
    This function unzips all Sentinel-2 products if they are not already unzipped.

    Args:
    - products (list): List of product identifiers.
    - path_s2 (str): Path to the Sentinel-2 products directory.

    Returns:
    - None
    """
    
    # Unzip if necessary
    for product in products :
        if not exists(join(path_s2, product + '.SAFE')) :
            print(f'Unzipping {product}...')
            unzip_l2a(path_s2, product)


def get_products_info(products, path_s2) :
    """
    This function extracts relevant information from the Sentinel-2 products.

    Args:
    - products (list): List of product identifiers.
    - path_s2 (str): Path to the Sentinel-2 products directory.

    Returns:
    - products_info (dict): Dictionary containing information about each product.
    - meta_10m, meta_20m, meta_60m (tuple): Metadata for the 10m, 20m, and 60m bands.
    """

    products_info = {}
    products_defect = []
    for product in products :

        # Get the date and tile name from the L2A product name
        _, _, date, _, _, tname, _ = product.split('_')
        year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])

        # Get the path to the IMG_DATA/ folder of the Sentinel-2 product
        try: path_to_img_data = glob.glob(join(path_s2, product + '.SAFE', 'GRANULE', '*', 'IMG_DATA'))[0]
        except Exception as e:
            print(f'Error finding IMG_DATA folder for product {product}: {e}. Skipping...')
            products_defect.append(product)
            continue

        # Check that all files of interest are present
        all_bands = True
        for res, bands in S2_L2A_BANDS.items() :
            for band in bands :
                if not exists(join(path_to_img_data, f'R{res}', f'{tname}_{date}_{band}_{res}.tif')) and \
                   not exists(join(path_to_img_data, f'R{res}', f'{tname}_{date}_{band}_{res}.jp2')) :
                    all_bands = False
        if not exists(join(path_s2, product + '.SAFE', 'MTD_MSIL2A.xml')) : all_bands = False
        if not all_bands :
            print(f'Not all bands found for product {product}. Skipping...')
            products_defect.append(product)
            continue

        # Get the file extension
        if not exists(join(path_to_img_data, 'R10m', f'{tname}_{date}_B02_10m.tif')) :
            if not exists(join(path_to_img_data, 'R10m', f'{tname}_{date}_B02_10m.jp2')) :
                raise FileNotFoundError(f'No 10m resolution bands found for {product}')
            else: file_extension = 'jp2'
        else: file_extension = 'tif'

        # Check the BOA quantification value (and BOA offsets if applicable)
        if dt.date(2022, 1, 25) <= dt.date(year, month, day) : boa_offset = 1 
        else: boa_offset = 0
        radiometric_offset_values(path_s2, product, boa_offset)

        products_info[product] = {'path_to_img_data': path_to_img_data, 'tname': tname, 'date': date, 'boa_offset': boa_offset, 'file_extension': file_extension}
    
    # If no valid products were found, raise an error
    if products_info == {} : raise ValueError(f'No valid products found.')
    
    # Get the metadata for the 10m, 20m, and 60m bands; from one of the valid products
    p_info = list(products_info.values())[0]
    path_to_img_data, tname, date, file_extension = [p_info[key] for key in ['path_to_img_data', 'tname', 'date', 'file_extension']]
    with rs.open(join(path_to_img_data, f'R10m', f'{tname}_{date}_B02_10m.{file_extension}')) as src :
        meta_10m = src.meta.copy()
    with rs.open(join(path_to_img_data, f'R20m', f'{tname}_{date}_B05_20m.{file_extension}')) as src :
        meta_20m = src.meta.copy()
    with rs.open(join(path_to_img_data, f'R60m', f'{tname}_{date}_B01_60m.{file_extension}')) as src :
        meta_60m = src.meta.copy()
    metadata = {'10m': meta_10m, '20m': meta_20m, '60m': meta_60m}
    
    return products_info, metadata, products_defect


def get_sizes(products, products_info) :
    """
    This function finds the height and width of the tile based on the Sentinel-2 products and calculates the window size
    based on the number of products and bands.

    Args:
    - products (list): List of product identifiers.
    - products_info (dict): Dictionary containing information about each product.

    Returns:
    - sizes (dict): Dictionary containing the height and width of the tile for each resolution.
    """

    # Get the heights and widths of the tile from the first product
    sizes = {}
    first_product = products[0]
    path_to_img_data, tname, date, file_extension = [products_info[first_product][key] for key in ['path_to_img_data', 'tname', 'date', 'file_extension']]
    with rs.open(join(path_to_img_data, f'R10m', f'{tname}_{date}_B02_10m.{file_extension}')) as src :
        tile_height, tile_width = src.height, src.width
        sizes['10m'] = (tile_height, tile_width)
    with rs.open(join(path_to_img_data, f'R20m', f'{tname}_{date}_B05_20m.{file_extension}')) as src :
        sizes['20m'] = (src.height, src.width)
    with rs.open(join(path_to_img_data, f'R60m', f'{tname}_{date}_B01_60m.{file_extension}')) as src :
        sizes['60m'] = (src.height, src.width)
    return sizes


def check_if_exists(save_path, res_band_tuples, calculate_coverage = False, force = False) :
    """
    This function checks if the composite already exists at the specified save path.

    Args:
    - save_path (str): Path where the composite will be saved.
    - res_band_tuples (list): List of tuples containing resolution and band names.
    - calculate_coverage (bool): Whether to check for the coverage mask.

    Returns:
    - all_good (bool): True if all required files exist, False otherwise.
    - missing_bands (list): List of missing bands or files.
    """

    if force :
        return False, [band for _, band in res_band_tuples] + ['products'] + (['count'] if calculate_coverage else [])
    
    # If the .zip file exists
    if isfile(f'{save_path}.zip') :
        try: 
            zip_ref = ZipFile(f'{save_path}.zip', 'r')
            namelist = zip_ref.namelist()
            namelist = [basename(n) for n in namelist]
            namelist.remove('')
        except zipfile.BadZipFile:
            print(f'BadZipFile error reading {save_path}.zip. It might be corrupted. Re-running the composite...')
            remove(f'{save_path}.zip')
            return False, [band for _, band in res_band_tuples] + ['products'] + (['count'] if calculate_coverage else [])
        except Exception as e:
            print(f'Error reading {save_path}.zip: {e}. Re-running the composite...')
            remove(f'{save_path}.zip')
            return False, [band for _, band in res_band_tuples] + ['products'] + (['count'] if calculate_coverage else [])
    
    # If the .zip file doesn't exist, but the folder does
    elif isdir(save_path) :
        namelist = glob.glob(join(save_path, '*'))
        namelist = [basename(n) for n in namelist]
        namelist.remove('')

    # Otherwise, there is no such data
    else: 
        all_bands = ['products'] + [band for _, band in res_band_tuples] + (['count'] if calculate_coverage else [])
        return False, all_bands

    # Check which files already exist    
    missing_bands = []
    for _, band in res_band_tuples :
        if not f'{band}.tif' in namelist : missing_bands.append(band)
    if not 'products.txt' in namelist : missing_bands.append('products')
    if calculate_coverage and (not 'count.tif' in namelist) : 
        missing_bands.append('count')
        if 'B02' not in missing_bands : missing_bands.append('B02')
    all_good = (len(missing_bands) == 0)

    return all_good, missing_bands

#######################################################################################################################
# Code execution

def composite(argv=None) :

    s2_tile, path_s2, method, mode, year, path_txt, window_size, clouds, calculate_coverage, force, skip_SCL = _parser(argv)

    # List the products and unzip them if necessary
    products = list_products(s2_tile, path_s2, mode, year, path_txt)
    unzip_all(products, path_s2)

    # Get the products information and metadata, and the sizes of the tile
    products_info, metadata, products_defect = get_products_info(products, path_s2)
    products = [p for p in products if p not in products_defect] # Filter out defective products
    sizes = get_sizes(products, products_info)

    # Prepare the keys and band tuples for processing
    keys = ['path_to_img_data', 'tname', 'date', 'file_extension', 'boa_offset']
    res_band_tuples = [(res, band) for res, bands in S2_L2A_BANDS.items() for band in bands]
    if skip_SCL : res_band_tuples = [t for t in res_band_tuples if t[1] != 'SCL']
    
    # Prepare for the composite to be saved
    save_path = join(path_s2, f'{s2_tile}_{year}_composite_{method}')
    all_good, missing_bands = check_if_exists(save_path, res_band_tuples, calculate_coverage, force)
    if all_good :
        print(f'Composite for tile {s2_tile} in year {year} already exists at {save_path}.')
        return
    if not exists(save_path) : makedirs(save_path, exist_ok = True)

    previous_res = None
    for i, (res, band) in enumerate(res_band_tuples) :

        if band not in missing_bands :
            print(f'Band {band} already exists, skipping...')
            continue
        CC = (band == 'B02' and calculate_coverage and 'count' in missing_bands)

        print(f'>> Processing band {i+1}/{len(res_band_tuples)}: {band} ({res})...')

        if res != previous_res :
            tile_height, tile_width = sizes[res]
            if window_size is None: window_size = max(tile_height, tile_width)
            n_height = int(np.ceil(tile_height / window_size))
            n_width = int(np.ceil(tile_width / window_size))
            print(f'Will use a window size of {window_size} pixels, which amounts to {n_height} x {n_width} window(s).')

        band_composite = np.zeros((tile_height, tile_width), dtype = np.float32)
        # If calculating coverage, initialize a mask to count valid pixels
        if CC : num_valid = np.zeros((tile_height, tile_width), dtype = np.uint8)
        # Iterate over the windows
        for h in range(n_height) :
            for v in range(n_width) :
                row_start, row_stop = h * window_size, (h + 1) * window_size
                col_start, col_stop = v * window_size, (v + 1) * window_size
                row_stop = min(row_stop, tile_height)
                col_stop = min(col_stop, tile_width)
                w_width = col_stop - col_start
                w_height = row_stop - row_start
                window = Window(col_start, row_start, w_width, w_height)

                products_data = np.zeros((len(products), w_height, w_width), dtype = np.float32)
                if clouds : cloud_masks = np.full((len(products), w_height, w_width), False, dtype = bool) # False = no cloud
                for i, product in enumerate(products) :
                    print(f'    . processing product {i+1}/{len(products)}: {product}...')
                    path_to_img_data, tname, date, file_extension, boa_offset = [products_info[product][key] for key in keys]
                    with rs.open(join(path_to_img_data, f'R{res}', f'{tname}_{date}_{band}_{res}.{file_extension}')) as src :
                        data = src.read(window = window).astype(np.float32)
                    if clouds :
                        if isfile(join(path_s2, f'{product}_CLOUDS.tif')) :
                            with rs.open(join(path_s2, f'{product}_CLOUDS.tif')) as src :
                                cloud_mask = src.read(1, window = window)
                        else:
                            print('      . no cloud mask found, proceeding without it...')
                            cloud_mask = np.zeros(data.shape, dtype = np.uint8)
                        cloud_masks[i, :, :] = (cloud_mask != 0) | (data == 0) # cloudy or nodata
                    valid_mask = (data != 0)
                    
                    # Calculate the surface reflectance values
                    if band != 'SCL' : 
                        data[valid_mask] = (data[valid_mask] - boa_offset * 1000) / 10000
                        data = (data - boa_offset * 1000) / 10000
                    
                    data[~valid_mask] = np.nan  # Set invalid pixels to NaN
                    products_data[i, :, :] = data

                # Calculate the reduction across the products
                if band == 'SCL' :
                    start_ = time.time()
                    print(f'    . calculating mode composite...')
                    data = stats.mode(products_data, axis = 0, nan_policy = 'omit')[0].astype(np.uint8)
                    print(f'      mode composite calculated in {time.time() - start_:.2f} seconds.')
                    data[np.isnan(data)] = 0  # Set NaN values to 0 for SCL
                else:
                    print(f'    . calculating {method} composite...')
                    if method == 'median' :

                        if not clouds: 
                            data = np.nanmedian(products_data, axis = 0)
                            data[np.isnan(data)] = 0  # Set NaN values to 0
                        else:

                            # Where there is at least one valid pixel, calculate the median
                            at_least_one_valid = np.any(~cloud_masks, axis = 0)
                            
                            # For pixels that have at least one valid observation, mask out the cloudy observations
                            products_data[:, at_least_one_valid] = np.where(cloud_masks[:, at_least_one_valid], np.nan, products_data[:, at_least_one_valid])
                            
                            # Calculate the median
                            data = np.nanmedian(products_data, axis = 0)
                            data[np.isnan(data)] = 0
                    
                    else: raise ValueError(f"Method {method} not supported")
                    if CC :
                        print(f'    . calculating coverage mask...')
                        window_valid = np.sum(~np.isnan(products_data), axis = 0).astype(np.uint8)
                band_composite[row_start : row_stop, col_start : col_stop] = data
                if CC : num_valid[row_start : row_stop, col_start : col_stop] = window_valid
        
        # Save the band composite
        print(f'    . saving {band} composite...')
        meta = metadata[res].copy()
        meta.update({'dtype': 'float32' if band != 'SCL' else 'uint8', 'count': 1, 'compress': 'lzw', 'driver': 'GTiff'})
        with rs.open(join(save_path, f'{band}.tif'), 'w', **meta) as dst :
            dst.write(band_composite, 1)
        print(f'      done :)')

        if CC :
            # Save the coverage mask
            print(f'    . saving coverage count...')
            meta.update({'dtype': 'uint8'})
            with rs.open(join(save_path, f'count.tif'), 'w', **meta) as dst :
                dst.write(num_valid, 1)
            print(f'      done :)')
    
    # Save the list of products
    if 'products' in missing_bands :
        print(f'    . saving list of products...')
        with open(join(save_path, 'products.txt'), 'w') as f :
            for product in products : f.write(f'{product}\n')
    
    # Compress the composite into a zip file and delete the uncompressed folder
    shutil.make_archive(save_path, 'zip', path_s2, f'{s2_tile}_{year}_composite_{method}')
    shutil.rmtree(save_path)

    # Remove the unzipped products
    for product in products :
        path_to_product = join(path_s2, product + '.SAFE')
        if exists(path_to_product) :
            print(f'Removing unzipped product {path_to_product}...')
            shutil.rmtree(path_to_product)
        else:
            print(f'Product {path_to_product} not found, skipping removal.')
    
    # Delete the defective products
    for product in products_defect :
        if exists(join(path_s2, product + '.SAFE')) :
            print(f'Removing defective product {path_to_product}...')
            shutil.rmtree(join(path_s2, product + '.SAFE'))
        #    remove(join(path_s2, product + '.zip'))


if __name__ == '__main__':
    t0 = time.time()
    composite()
    ttotal = time.time() - t0
    print(f'The process has finished. In: {str(timedelta(seconds=ttotal))}.')