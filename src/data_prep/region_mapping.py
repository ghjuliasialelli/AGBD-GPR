"""

This script creates a mapping between provided Sentinel-2 tiles and their corresponding GEDI region.

The format of the region_mapping.pkl file is as follows:
                tile: [region1]
                or tile: [region1, region2] if the tile belongs to multiple regions.

This was previously done in a naive manner, where countries were assigned region(s), and all tiles in
that country (as they were grouped by country in the .txt files) were assigned to that region. But this
is not scalable.


GEDI04_A world region includes the geologically defined continents of Africa and Europe. The South America
world region is the continent of South America, Central America and the Caribbean islands, and geological
North America south of southern Mexico. The Australia and Oceania world region is geological Australia and
the island regions north of Australia on the east side of the Wallace line, which defines the floral and 
faunal boundary between Australia and Asia during the Pleistocene (Mayr, 1944). The islands of Micronesia, 
Melanesia, and Polynesia are associated with the Australia and Oceania world region regardless of political
affiliation. The North American world region includes geological North America north of southern Mexico. The
continent of Asia was divided into north and south regions that approximately correspond to temperate and 
tropical forests.


History of versions:
    - v1: initial version. Hand-crafted, s.t. each tile in our subset was in the right GEDI region.
    - v2: extended the mapping to include Cecilia's tiles, based on the mapping in the world_regions.geojson file.
    - v3: extended the mapping to all S2 tiles, based on the mapping in the world_regions.geojson file.

"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
import geopandas as gpd
import pickle
import argparse

#######################################################################################################################
# Code execution

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles', type = str, help = 'Tiles for which to download products (filename).')
    parser.add_argument('--all', action = 'store_true', help = 'If set, all tiles will be processed. Otherwise, only the tiles in the provided file will be processed.')
    parser.add_argument('--version', type = str, default = 'v3', help = 'Version of the mapping to create (v1, v2, v3).')
    args = parser.parse_args()

    assert args.tiles or args.all, "Please provide a tiles file or set --all to True."

    path_shp = f"{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
    path_geojson = f"{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/helper/region_mapping/world_regions.geojson"
    path_tiles = args.tiles

    # Load the Sentinel-2 tiles and their geometries
    grid_df = gpd.read_file(path_shp, engine = 'pyogrio').drop_duplicates(subset = ['Name'])
    # TODO add a filter that removes the ones that cover water
    
    if path_tiles :
        with open(path_tiles, 'r') as f : tiles = [t.strip() for t in f.readlines()]
        grid_df = grid_df[grid_df['Name'].isin(tiles)]
    else:
        # If --all is set, process all tiles
        tiles = grid_df['Name'].tolist()

    # Load the world regions and their geometries
    world_regions = gpd.read_file(path_geojson, engine = 'pyogrio')

    # GEDI str region to integer
    # (0=Water, 1=Europe, 2=North Asia, 3=Australasia, 4=Africa, 5=South Asia, 6=South America, 7=North America)
    str_to_int = {'Af': 4, 'Au': 3, 'Eu': 1, 'N-Am': 7, 'S-Am': 6, 'N-As': 2, 'S-As': 5}

    # Iterate over the tiles and find the corresponding world region(s)
    mapping = {}
    num_tiles = len(tiles)
    for i, tile in enumerate(tiles) :
        print(f"({i}/{num_tiles}) Processing tile: {tile}")
        tile_geom = grid_df[grid_df['Name'] == tile].geometry.values[0]
        world_region = world_regions[world_regions.geometry.intersects(tile_geom)]['world_region'].values.tolist()
        if world_region == [] : 
            # Edge cases
            if tile in ['10SDG', '11SKS'] : world_region = ['N-Am']
            else: print(f">> Warning: {tile} does not intersect with any world region.")

        # Get the GEDI region class (0=Water, 1=Europe, 2=North Asia, 3=Australasia, 4=Africa, 5=South Asia, 6=South America, 7=North America)
        if world_region != [] :
            mapping[tile] = [str_to_int[w] for w in world_region]
    
    # Save the mapping to a pickle file
    path_mapping = f"{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/s2_tile_to_region-{args.version}.pkl"
    with open(path_mapping, 'wb') as f :
        pickle.dump(mapping, f)
    print(f"Mapping saved to {path_mapping}")