"""


"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
import os
from zipfile import ZipFile
import shutil
import glob
from os.path import join, exists
import numpy as np
import xml.etree.ElementTree as ET
import pickle

#######################################################################################################################
# Helper functions 

def unzip_l2a(path_s2, s2_prod):
    """
    This function unzips the Sentinel-2 L2A product at hand, extracting only .tif files.

    Args:
    - path_s2: string, path to the Sentinel-2 data directory.
    - s2_prod: string, name of the Sentinel-2 L2A product. (ends in .zip)

    Returns:
    - None
    """

    zip_path = os.path.join(path_s2, s2_prod)
    
    with ZipFile(zip_path, 'r') as zip_ref:
        # Find the index of the folder containing the SAFE files
        namelist = zip_ref.namelist()
        idx = namelist[0].split('/').index(s2_prod.replace('.zip', '.SAFE'))
        
        for file in namelist:
            if (file.endswith('.tif')) or (file.endswith('MTD_MSIL2A.xml')) :
                # Create a new path by slicing off unwanted parts of the path
                parts = file.split('/')
                new_path = os.path.join(*parts[idx:])
                
                # Full path to where the file will be extracted
                full_path = os.path.join(path_s2, new_path)
                
                # Extract the file to the new path
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                if not file.endswith('/'):
                    with zip_ref.open(file) as source, open(full_path, 'wb') as target:
                        shutil.copyfileobj(source, target)


def get_cloud_coverage(path_s2, product) :

    tree = ET.parse(join(path_s2, product.replace('.zip', '.SAFE'), 'MTD_MSIL2A.xml'))
    root = tree.getroot()

    total_cloud_pct = 0
    for elem in root.find('.//Image_Content_QI') :
        if elem.tag in ['HIGH_PROBA_CLOUDS_PERCENTAGE', 'MEDIUM_PROBA_CLOUDS_PERCENTAGE'] : 
            total_cloud_pct += float(elem.text)
    
    return total_cloud_pct


def create_mapping(path_s2, tiles) :
    """
    TODO
    """

    # Initialize the mapping
    mapping = {}

    # Iterate over the tiles
    for tile in tiles :
        print()
        print('Tile:', tile)

        # For each tile, iterate over the available products
        match = None
        for year in [2020, 2019, 2018] :
            print('> Year:', year)
            
            products = [p.split('/')[-1] for p in glob.glob(join(path_s2, f'*_*_{year}*_*_*_T{tile}_*.zip'))]
            if products == [] :
                continue

            percentages = []
            for product in products :

                # If the product is in .SAFE, don't unzip, otherwise unzip but only the SCL and the BANDS
                if not exists(join(path_s2, product.replace('.zip', '.SAFE'))) :
                    print('> Unzipping product:', product)
                    try:
                        unzip_l2a(path_s2, product)
                    
                    except Exception as e:
                        print(f'>> Could not unzip: {e}')
                        percentages.append(100)
                        continue

                # Check that the product has all the bands we want
                bands = glob.glob(join(path_s2, product.replace('.zip', '.SAFE'), 'GRANULE', '*', 'IMG_DATA', 'R*m', '*.tif'))
                if len(bands) != 14 :
                    print('Missing bands.')
                    percentages.append(100)
                    continue

                # Now get the SCL cloud coverage
                try: 
                    scl = get_cloud_coverage(path_s2, product)
                    print(f'> Cloud coverage: {scl:.2f}%')
                    percentages.append(scl)
                except Exception as e:
                    print(f'>> Could not get cloud coverage: {e}')
                    percentages.append(100)
                    continue
            
            # Get the product with the least cloud coverage
            valid_percentages = [p for p in percentages if p < 100]

            if valid_percentages == [] :
                print('> No product found.')
                for product in products :
                    shutil.rmtree(join(path_s2, product.replace('.zip', '.SAFE')))

            else:
                match = products[np.argmin(percentages)]

                # Delete the .SAFE folders for the other products
                for product in products :
                    if product != match :
                        shutil.rmtree(join(path_s2, product.replace('.zip', '.SAFE')))

            # If we have a match, no need to iterate over the other years
            if match is not None :
                break

        if match is None :
            print('> No product found.')
            continue

        # Save tile: product (without extension) to the mapping
        mapping[tile] = match.replace('.zip', '')

    return mapping

#######################################################################################################################
# Code execution


if __name__ == '__main__': 
    
    # Set the paths
    s2_path = f'{DATA_ROOT}/S2_L2A'
    txt_path = f'{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/Sentinel_Clem_California_Cuba_Paraguay_UnitedRepublicofTanzania_Ghana_Austria_Greece_Nepal_ShaanxiProvince_NewZealand_FrenchGuiana.txt'

    # Read the tiles
    #with open(txt_path, 'r') as f:
    #   s2_tiles = [t.strip() for t in f.readlines()]

    # Get the mapping
    s2_tiles = ['10SGE', '10TCL', '11TKG', '16QHK', '17QQE', '21JWN', '21KVQ', '21KVT', '21KXP', '21NYF', '22NCH', '22NCK', '30NYP', '31NCG', '35MQP', '36LVR', '36LYQ', '36MUB', '36MVC', '36MXB', '37LDL', '37MBT', '37MDN', '45RTN', '45RVK', '59GNM', '59GPQ', '59HPA', '59HPB']
    mapping = create_mapping(s2_path, s2_tiles)

    # Save the mapping
    with open(join(s2_path, 'mapping_v2.pkl'), 'wb') as f:
       pickle.dump(mapping, f)