"""

This script generates a cloud mask for all products of a given Sentinel-2 tile in a given year, using the cloudSEN12 model.

Information about this model is available at: https://github.com/IPL-UV/cloudsen12_models/tree/main?tab=readme-ov-file
Use the `cloudmask` conda environment to run this script.


"""

#######################################################################################################################
# Imports

from cloudsen12_models import cloudsen12
from os.path import join, exists, dirname, basename
import glob
import datetime as dt
import rasterio as rs
from zipfile import ZipFile
from os import makedirs
import shutil
import xml.etree.ElementTree as ET
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize
import numpy as np
from georeader.geotensor import GeoTensor
import argparse
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import time

S2_L2A_BANDS = {'10m' : ['B02', 'B03', 'B04', 'B08'],
                '20m' : ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL'],
                '60m' : ['B01', 'B09']}

NODATAVALS = {'S2' : 0}

S2_attrs = {'bands' : {'B01': np.uint16, 'B02': np.uint16, 'B03': np.uint16, 'B04': np.uint16, 'B05': np.uint16, 'B06': np.uint16, 
                        'B07': np.uint16, 'B08': np.uint16, 'B8A': np.uint16, 'B09': np.uint16, 'B11': np.uint16, 'B12': np.uint16, 
                        'SCL': np.uint8},
            'metadata' : {'vegetation_score': np.uint8, 'date' : np.int16, 'pbn' : np.uint16, 'ron' : np.uint8, 'boa_offset': np.uint8}
            }



#######################################################################################################################
# Helper functions

def str2bool(v):
    """ 
        Helper function to parse a string into a boolean.
        
        input: `v` (str), input string to be parsed
        output: bool
    """
    if v == 'true': return True
    elif v == 'false': return False
    else: raise argparse.ArgumentTypeError(f"Either 'true' or 'false' expected, got {v}.")


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


def process_S2_tile(product, path_s2) :
    """
    This function iterates over the bands of the Sentinel-2 L2A product at hand; reprojects them to
    EPSG 4326; upsamples them to 10m resolution (when needed) using cubic interpolation (nearest
    neighbor for the scene classification mask); and returns them.
    
    Args:
    - product: string, name of the Sentinel-2 L2A product.
    - path_s2: string, path to the Sentinel-2 data directory.

    Returns:
    - processed_bands: dictionary, with the band names as keys, and the corresponding 2d arrays as values.
    """

    # Unzip if necessary
    if not exists(join(path_s2, product + '.SAFE')) :
        print(f'Unzipping {product}...')
        unzip_l2a(path_s2, product)

    # Get the path to the IMG_DATA/ folder of the Sentinel-2 product
    path_to_img_data = glob.glob(join(path_s2, product + '.SAFE', 'GRANULE', '*', 'IMG_DATA'))[0]

    # Get the date and tile name from the L2A product name
    _, _, date, _, _, tname, _ = product.split('_')
    year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])

    # Check the BOA quantification value (and BOA offsets if applicable)
    if dt.date(2022, 1, 25) <= dt.date(year, month, day) : boa_offset = 1 
    else: boa_offset = 0
    radiometric_offset_values(path_s2, product, boa_offset)

    # Get the file extension
    if not exists(join(path_to_img_data, 'R10m', f'{tname}_{date}_B02_10m.tif')) :
        if not exists(join(path_to_img_data, 'R10m', f'{tname}_{date}_B02_10m.jp2')) :
            raise FileNotFoundError(f'No 10m resolution bands found for {product}')
        else: file_extension = 'jp2'
    else: file_extension = 'tif'

    # Iterate over the bands
    processed_bands = {}
    for res, bands in S2_L2A_BANDS.items() :
        for band in bands :

            # Read the band data
            with rs.open(join(path_to_img_data, f'R{res}', f'{tname}_{date}_{band}_{res}.{file_extension}')) as src :
                band_data = src.read(1)
                
                # Use the 10m resolution B02 band as reference
                if res == '10m' : 
                    if band == 'B02' :
                        upsampling_shape = band_data.shape
                        _transform = src.transform
                        crs = src.crs
                        bounds = src.bounds
                        meta = src.meta

                # Upsample the band to a 10m resolution if necessary
                else :
                    # Order 0 indicates nearest interpolation, and order 3 indicates bi-cubic interpolation
                    if band == 'SCL' :
                        band_data = upsampling_with_nans(band_data, upsampling_shape, NODATAVALS['S2'], 0).astype(S2_attrs['bands'][band])
                    else:
                        band_data = upsampling_with_nans(band_data.astype(np.float32), upsampling_shape, NODATAVALS['S2'], 1).astype(S2_attrs['bands'][band])

            # Turn the band into a 2d array
            if len(band_data.shape) == 3 : band_data = band_data[0, :, :]
            processed_bands[band] = band_data
    
    # Delete the unzipped folder to save space
    shutil.rmtree(join(path_s2, product + '.SAFE'))

    return _transform, upsampling_shape, processed_bands, crs, bounds, boa_offset, meta


def get_products(s2_tile, path_s2, year) :
    """
    This function lists all Sentinel-2 products for the tile and year at hand.

    Args:
    - s2_tile: string, name of the Sentinel-2 tile.
    - path_s2: string, path to the Sentinel-2 data directory.
    - year: int, year for which to list the products.

    Returns:
    - products: list of strings, names of the Sentinel-2 products (without the .zip extension).
    """
    matches = glob.glob(join(path_s2, f'*_{year}*_*_T{s2_tile}_*.zip'))
    if len(matches) == 0 : raise FileNotFoundError(f'No products found for tile {s2_tile} in year {year} at path {path_s2}.')
    else: print(f'Found {len(matches)} products.')
    products = [basename(m).rstrip('.zip') for m in matches]
    return products

def compute_mask(window_size, modelv2, _transform, processed_bands, crs, bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']) :
    """
    This function computes the cloud mask for the Sentinel-2 L2A product at hand, using the cloudSEN12 model.

    Args:
    - window_size: int, size of the window to use for processing.
    - modelv2: cloudSEN12 model, pre-loaded cloudSEN12 model.
    - _transform: affine.Affine, affine transform of the Sentinel-2 product.
    - processed_bands: dictionary, with the band names as keys, and the corresponding 2d arrays as values.
    - crs: rasterio.crs.CRS, coordinate reference system of the Sentinel-2 product.
    - bands: list of strings, names of the bands to use for the cloud mask computation.

    Returns:
    - mask: 2d array, cloud mask (1 for cloud, 0 for no cloud).
    """

    # Initialize the cloud mask
    tile_height, tile_width = processed_bands['B02'].shape
    mask = np.zeros((tile_height, tile_width), dtype = np.uint8)

    # Iterate over windows of the product (cause the model cannot handle the full image at once)
    n_height = int(np.ceil(tile_height / window_size))
    n_width = int(np.ceil(tile_width / window_size))
    for h in range(n_height) :
        for v in range(n_width) :
            
            # Calculate the window boundaries
            row_start, row_stop = h * window_size, (h + 1) * window_size
            col_start, col_stop = v * window_size, (v + 1) * window_size
            row_stop = min(row_stop, tile_height)
            col_stop = min(col_stop, tile_width)
            window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
            w_transform = window_transform(window, _transform)
            
            # Create the GeoTensor for the window
            img = np.stack([processed_bands[band][row_start : row_stop, col_start : col_stop] for band in bands]).astype(np.float32)
            img = GeoTensor(img, transform = w_transform, crs = crs, fill_value_default = NODATAVALS['S2'])
            
            # Predict the cloud mask for the window
            cloud = modelv2.predict(img / 10_000)
            mask[row_start : row_stop, col_start : col_stop] = cloud.values.astype(np.uint8)
    
    return mask


def _parser() :
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile for which to run the composite.')
    parser.add_argument("--path_s2", required = True, type = str, help = 'Path to the Sentinel-2 products.')
    parser.add_argument("--year", required = True, type = int, help = 'Year for which to run the composite.')
    parser.add_argument("--window_size", type = int, default = None, help = 'Size of the window to use for processing.')
    args = parser.parse_args()
    return args.tile_name, args.path_s2, args.year, args.window_size


#######################################################################################################################
# Code execution

def cloud_mask() :

    s2_tile, path_s2, year, window_size = _parser()

    # Note that this goes against what is specified in the .sh file. This is because the environment cannot be installed
    # on the cluster, so I am running the script iteratively for each tile on my local machine, instead of in parallel.
    # TODO Remove, and manage to install the environment on the cluster to run in parallel.
    #with open('/path/to/data/EcosystemAnalysis/Models/Biomes/helper/nodata/txt_files/french_guiana.txt','r') as f:
    #    tiles = [t.strip() for t in f.readlines()]
    #print(f'Found {len(tiles)} tiles to process.')

    tiles = [s2_tile]
    
    for s2_tile in tiles :
        print()
        print(f'Processing tile {s2_tile}...')

        # List all Sentinel-2 products for the tile and year at hand
        products = get_products(s2_tile, path_s2, year)

        # Load the cloudSEN12 model
        modelv2 = cloudsen12.load_model_by_name(name = "cloudsen12l2a", weights_folder = "cloudsen12_models")

        # Iterate over the products
        num_products = len(products)
        for i, product in enumerate(products) :

            print(f'    Processing product {i+1}/{num_products}...')

            try:

                out_path = join(path_s2, f'{product}_CLOUDS.tif')
                if exists(out_path) : continue
                else:

                    # Process the Sentinel-2 tile
                    _transform, _, processed_bands, crs, _, _, _ = process_S2_tile(product, path_s2)

                    # Compute the cloud mask
                    mask = compute_mask(window_size, modelv2, _transform, processed_bands, crs)

                    # Save it
                    out_meta = {"driver": "GTiff", "height": mask.shape[0], "width": mask.shape[1], "count": 1, "dtype": rs.uint8, 
                                "crs": crs, "transform": _transform, "compress": "lzw"}
                    with rs.open(out_path, "w", **out_meta) as dest: dest.write(mask, 1)
                    print(f'    Cloud mask saved.')

            except Exception as e:
                print(f'    Error processing product {product}: {e}')
                continue

if __name__ == '__main__':
    t0 = time.time()
    cloud_mask()
    ttotal = time.time() - t0
    print(f'The process has finished. In: {str(dt.timedelta(seconds=ttotal))}.')