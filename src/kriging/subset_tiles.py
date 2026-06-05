"""

This script identifies a subset of all of the tiles, such that the subset is representative of the entire dataset.

"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
import numpy as np
from os.path import join
import numpy as np
import geopandas as gpd
from kriging.find_problematic_indices import get_s2_tiles

#######################################################################################################################
# Helper functions



#######################################################################################################################
# Code execution


if __name__ == "__main__" :

    seed = 42
    np.random.seed(seed)

    path_kriging = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/'
    path_valid = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_2020.txt'
    path_shp=f"{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
    path_geojson = join(DATA_ROOT, 'BiomassDatasetCreation', 'Data', 'countrySelection', 'AOIs.geojson')
    # Iterate over the regions
    regions = ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana']

    # Load all tiles to consider
    with open(path_valid, 'r') as f:
        og_valid_tiles = [t.strip() for t in f.readlines()]
    to_skip="58FEJ 58FEK 59FLB 01GEM 60FXL 58FGG 60FXK 10SDG 58GGR 59GNQ 58GFN 11SKS 49SET 11SMR 31NCG 35MQN 45RWM 22NCM 30PVS 49SEC 17QQC 17QQF 11SQV 59HQU 37MCT"
    to_skip = to_skip.split(' ')
    valid_tiles = [t for t in og_valid_tiles if t not in to_skip]

    # Load geometries of S2 tiles
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio').drop_duplicates(subset = ['Name'])
    grid_df = grid_df[grid_df['Name'].isin(valid_tiles)]

    # Load geometries of regions
    countries_df = gpd.read_file(path_geojson)
    countries_df = countries_df[countries_df['name'].isin(regions)]

    # For each region, get the corresponding tiles and keep a random 5% of them
    tiles_to_keep = []
    for region in regions :
        print(f"Processing region {region}...")
        s2_tiles = get_s2_tiles(region, countries_df, grid_df)
        # Pick at random 5% of the tiles
        n_tiles = len(s2_tiles)
        n_subset = max(1, int(0.05 * n_tiles))
        subset = np.random.choice(s2_tiles, n_subset, replace = False)
        tiles_to_keep.extend(subset)
    
    # Save the tiles to keep
    with open(join(path_kriging, 'txt_files', 'subset_tiles.txt'), 'w') as f :
        for t in tiles_to_keep :
            f.write(f"{t}\n")