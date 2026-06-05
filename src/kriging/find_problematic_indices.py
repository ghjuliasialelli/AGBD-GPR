"""

This script identifies GEDI footprint indices that are present across multiple tiles. It assigns each problematic index to the last tile it was found in.
This is relevant for computing metrics without double accounting these footprints.

Run with the `krige` environment, on the cluster, with the following command:
    sbatch --wrap="python find_problematic_indices.py" --time=4:00:00 --mem-per-cpu=8G --cpus-per-task=8 --output=problematic.out --error=problematic.out

"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
import pickle
import time
from collections import Counter
from os.path import join
import geopandas as gpd

#######################################################################################################################
# Helper functions

def get_s2_tiles(region, countries_df, grid_df) :
    """
    Returns the list of S2 tiles that intersect with the given region.

    Args:
    - region (str): The name of the region.
    - countries_df (GeoDataFrame): A GeoDataFrame containing the geometries of regions.
    - grid_df (GeoDataFrame): A GeoDataFrame containing the geometries of S2 tiles.

    Returns:
    - s2_tiles (list): A list of S2 tile names that intersect with the region.
    """
    country_geom = countries_df[countries_df['name'] == region].geometry.values[0]
    region_tiles = grid_df[grid_df.intersects(country_geom)]
    s2_tiles = region_tiles['Name'].tolist()
    return s2_tiles

#######################################################################################################################
# Code execution

def main() :

    # Arguments #######################################################################################################
    
    path_kriging = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/'
    regions = ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana']

    # Load necessary data #############################################################################################

    # Load all tiles to consider
    with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_2020.txt', 'r') as f:
        og_valid_tiles = [t.strip() for t in f.readlines()]
    to_skip="58FEJ 58FEK 59FLB 01GEM 60FXL 58FGG 60FXK 10SDG 58GGR 59GNQ 58GFN 11SKS 49SET 11SMR 31NCG 35MQN 45RWM 22NCM 30PVS 49SEC 17QQC 17QQF 11SQV 59HQU 37MCT"
    to_skip = to_skip.split(' ')
    valid_tiles = [t for t in og_valid_tiles if t not in to_skip]

    # Load geometries of S2 tiles
    path_shp=f"{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio').drop_duplicates(subset = ['Name'])
    grid_df = grid_df[grid_df['Name'].isin(valid_tiles)]

    # Load geometries of regions
    path_geojson = join(DATA_ROOT, 'BiomassDatasetCreation', 'Data', 'countrySelection', 'AOIs.geojson')
    countries_df = gpd.read_file(path_geojson)
    countries_df = countries_df[countries_df['name'].isin(regions)]

    # Find problematic indices ########################################################################################

    problematic_indices = {}
    for region in regions :
        print(f"Processing region {region}...")
        region_counter = Counter()
        region_mapping = {} # mapping from index to tiles
        s2_tiles = get_s2_tiles(region, countries_df, grid_df)
        print(f'    found {len(s2_tiles)} tiles in the region.')
        for tile in s2_tiles :
            try:
                print(f'    > processing tile {tile}...')
                # Find the index of the tile in valid_tiles
                tile_idx = og_valid_tiles.index(tile) + 1
                pkl_path = join(path_kriging, 'predictions', 'indices')
                with open(join(pkl_path, f"idx-{tile_idx}.pkl"), 'rb') as f :
                    tile_indices = pickle.load(f)['indices']
                region_counter.update(tile_indices)
                region_mapping.update({idx: tile for idx in tile_indices})
            except Exception as e :
                print(f"        !!! could not process tile {tile}, skipping it. Error: {e}")
                continue
        region_problematic_indices = [idx for idx, count in region_counter.items() if count > 1]
        indices_to_tile = [region_mapping[idx] for idx in region_problematic_indices]
        problematic_indices[region] = {'indices': region_problematic_indices, 'tiles': indices_to_tile}

    # Save results
    with open('problematic_indices.pkl', 'wb') as f :
        pickle.dump(problematic_indices, f)

if __name__ == "__main__" :
    start_time = time.time()
    main()
    print(f"Script finished in {time.time() - start_time} seconds.")