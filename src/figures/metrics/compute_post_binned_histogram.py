import os
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
"""

This script computes and saves the binned histogram of post-kriging predictions.

Run with the `krige` environment, on the cluster, with the following command:
    sbatch --wrap="python compute_post_binned_histogram.py --model_name <model_name> --force --test" --time=4:00:00 --mem-per-cpu=4G --cpus-per-task=1 --output=post_histogram.out --error=post_histogram.out

"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
from os.path import exists
from os import makedirs
import argparse
import numpy as np
import pickle
import time
from os.path import join
from os.path import isfile
import numpy as np
import geopandas as gpd

#######################################################################################################################
# Helper functions

def _parser() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, required = True, help = 'Model name identifier.')
    parser.add_argument('--force', action='store_true', help = 'Whether to force recomputation even if the file already exists.')
    parser.add_argument('--subset', action='store_true', help = 'Whether to use a subset of the data.')
    parser.add_argument('--test', action='store_true', help = 'Whether to compute only on the test data.')
    parser.add_argument('--ref_model', type = str, required = False, help = 'Model to compare for the test set.')
    parser.add_argument('--regional', action='store_true', help = 'Aggregate on a regional-level.')
    parser.add_argument('--min', type = int, default = 500, help = 'Minimum residuals values to consider (negative).')
    parser.add_argument('--max', type = int, default = 400, help = 'Maximum residuals values to consider (positive).')
    parser.add_argument('--CCI', action='store_true', help = 'Whether to eval CCI preds.')
    args = parser.parse_args()
    return args.model_name, args.force, args.subset, args.test, args.ref_model, args.regional, args.min, args.max, args.CCI

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

def compute_metrics(post, ref, lbs, ubs, bin_hists, residual_bins) :
    """
    This function updates the histograms of residuals per label bin.

    Args:
    - post (np.ndarray): Post-kriging predictions.
    - ref (np.ndarray): Reference values.
    - lbs (list): Lower bounds of the label bins.
    - ubs (list): Upper bounds of the label bins.
    - bin_hists (dict): Dictionary to store histograms of residuals per label bin.
    - residual_bins (np.ndarray): Bins for the residual histograms.

    Returns:
    - bin_hists (dict): Updated dictionary with histograms of residuals per label bin.
    """

    # Remove NaNs
    valid_mask = ~np.isnan(post) & ~np.isnan(ref)
    if np.sum(valid_mask) == 0 : return bin_hists
    post = post[valid_mask]
    ref = ref[valid_mask]

    # Compute the residuals and update the histograms
    residuals = post - ref
    for lb, ub in zip(lbs, ubs):
        mask = (ref >= lb) & (ref < ub)
        if np.any(mask):
            counts, _ = np.histogram(residuals[mask], bins = residual_bins)
            bin_hists[f"{lb}-{ub}"] += counts
    return bin_hists


#######################################################################################################################
# Code execution

def main() :

    # Arguments #######################################################################################################

    model_name, force, subset, test_set, ref_model, regional, _min, _max, CCI = _parser()
    if 'baseline' in model_name :
        baseline = True
        if test_set : 
            assert ref_model is not None, "When using a baseline model on the test set, need to provide a reference model with --ref_model."
    else: 
        baseline = False
        if test_set : ref_model = model_name

    path_kriging = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/'
    path_valid = f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/inference/per_tile/valid_2020.txt'
    path_shp=f"{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp"
    path_geojson = join(DATA_ROOT, 'BiomassDatasetCreation', 'Data', 'countrySelection', 'AOIs.geojson')
    year = 2020
    arch = 'nico_film'
    inf_model = '17997535-1_17997535-2_17997535-3'
    regions = ['California', 'Cuba', 'Paraguay', 'UnitedRepublicofTanzania', 'Ghana', 'Austria', 'Greece', 'Nepal', 'ShaanxiProvince', 'NewZealand', 'FrenchGuiana']

    # Per-label-bin histograms
    residual_bins = np.arange(-_min,_max,1)

    # If test_set, need to load the config from wandb
    if test_set : import wandb; api = wandb.Api()  # W&B account required only for this lookup

    if not CCI:
        save_path = join(path_kriging, 'predictions', arch, str(year), inf_model)
        file_name = f"{model_name}-hist-post{'_subset' if subset else ''}{'_test' if test_set else ''}{'_regional' if regional else ''}{('_' + str(_min) + '-' + str(_max)) if _min != 400 else ''}.pkl"
    else:
        save_path = join(path_kriging, 'predictions', 'CCI')
        file_name = f"CCI-hist-post{'_subset' if subset else ''}{'_test' if test_set else ''}{'_regional' if regional else ''}{('_' + str(_min) + '-' + str(_max)) if _min != 400 else ''}.pkl"
    if not isfile(join(save_path, file_name)) or force:


        # Load necessary data #############################################################################################

        # Load all tiles to consider
        with open(path_valid, 'r') as f:
            og_valid_tiles = [t.strip() for t in f.readlines()]
        to_skip="58FEJ 58FEK 59FLB 01GEM 60FXL 58FGG 60FXK 10SDG 58GGR 59GNQ 58GFN 11SKS 49SET 11SMR 31NCG 35MQN 45RWM 22NCM 30PVS 49SEC 17QQC 17QQF 11SQV 59HQU 37MCT"
        to_skip = to_skip.split(' ')
        valid_tiles = [t for t in og_valid_tiles if t not in to_skip]

        # Subset tiles
        if subset:
            with open(join(path_kriging, 'txt_files', 'subset_tiles.txt'), 'r') as f:
                subset_tiles = [t.strip() for t in f.readlines()]

        # Load geometries of S2 tiles
        grid_df = gpd.read_file(path_shp, engine = 'pyogrio').drop_duplicates(subset = ['Name'])
        grid_df = grid_df[grid_df['Name'].isin(valid_tiles)]

        # Load geometries of regions
        countries_df = gpd.read_file(path_geojson)
        countries_df = countries_df[countries_df['name'].isin(regions)]

        # Load the problematic indices, pre-computed by find_problematic_indices.py
        path_pkl = join(path_kriging, 'helper')
        with open(join(path_pkl, 'problematic_indices.pkl'), 'rb') as f :
            all_problematic_indices = pickle.load(f)

        # Compute histogram ####################################################################################################

        # Define the bins
        bins = np.arange(0, 501, 50)
        lbs, ubs = bins[:-1], bins[1:]

        # Initialize the histograms
        if regional : bin_hists = {region: {f"{lb}-{ub}": np.zeros(len(residual_bins) - 1, dtype = np.int64) for lb, ub in zip(lbs, ubs)} for region in regions}
        else: bin_hists = {f"{lb}-{ub}": np.zeros(len(residual_bins)-1, dtype=np.int64) for lb, ub in zip(lbs, ubs)}

        # Iterate over the regions
        for region in regions :
            print(f"Processing region {region}...")
            problematic_indices = np.array(all_problematic_indices[region]['indices'])
            problematic_tiles = all_problematic_indices[region]['tiles']
            
            # Iterate over the tiles in the region
            s2_tiles = get_s2_tiles(region, countries_df, grid_df)
            if subset : s2_tiles = [t for t in s2_tiles if t in subset_tiles]
            for tile in s2_tiles :
                try:
                    print(f'    > processing tile {tile}...')
                    tile_idx = og_valid_tiles.index(tile) + 1

                    # Read the .pkl file of the tile
                    if not CCI : # normal case
                        tif_path = join(path_kriging, 'predictions', arch, tile, str(year), inf_model)
                        if baseline : fname = f"results-{model_name}.pkl"
                        else: fname = f"results-{model_name}-{tile_idx}.pkl"
                        with open(join(tif_path, fname), 'rb') as f : tile_data = pickle.load(f)
                        post, ref = np.array(tile_data['post']), np.array(tile_data['ref'])
                        # Load the tile's indices
                        with open(join(path_kriging, 'predictions', 'indices', f"idx-{tile_idx}.pkl"), 'rb') as f :
                            idx = pickle.load(f)['indices']

                    else: # evaluating CCI predictions
                        tif_path = join(path_kriging, 'predictions', 'CCI')
                        fname = f"results-{tile}.pkl"
                        with open(join(tif_path, fname), 'rb') as f : tile_data = pickle.load(f)
                        post, ref, idx = np.array(tile_data['preds']), np.array(tile_data['refs']), np.array(tile_data['idxs'])

                    # Compute either on the test set
                    if test_set :
                        # Load the indices of the test set
                        save_dir = join(path_kriging, 'predictions', 'splits')
                        runs = api.runs(f"{WANDB_ENTITY}/kriging", {"display_name": ref_model + f'-{tile_idx}'})
                        run = runs[len(runs) - 1]
                        config = run.config
                        test_holdout, val_holdout, max_train_footprints, max_split_diff, stripe_size, ood = config['test_holdout'], config['val_holdout'], config['max_train_footprints'], float(config['max_split_diff']), config['stripe_size'], config.get('ood', False)
                        num_footprints = run.summary['num_footprints']
                        ood = ood and (num_footprints > 500)
                        if stripe_size == 200 : ood = False  # Temporary fix for old runs
                        fname = f"splits-{tile_idx}_{test_holdout}_{val_holdout}_{max_train_footprints}_{max_split_diff}_{stripe_size}{'_ood' if ood else ''}.pkl"
                        with open(join(save_dir, fname), 'rb') as f: 
                            test_indices = pickle.load(f)['test']
                        ok_mask = np.isin(idx, test_indices)
                    # Or on the full set, but removing the problematic indices
                    else:
                        tile_problematic_indices = problematic_indices[np.isin(problematic_tiles, tile, invert = True)]
                        ok_mask = np.isin(idx, tile_problematic_indices, invert = True)

                    # Compute the metrics
                    if regional : bin_hists[region] = compute_metrics(post[ok_mask], ref[ok_mask], lbs, ubs, bin_hists[region], residual_bins)
                    else: bin_hists = compute_metrics(post[ok_mask], ref[ok_mask], lbs, ubs, bin_hists, residual_bins)
                    
                except Exception as e :
                    print(f"        !!! could not process tile {tile}, skipping it. Error: {e}")
                    continue

        # Save to file
        if not exists(save_path) : makedirs(save_path)
        with open(join(save_path, file_name), 'wb') as f:
            pickle.dump({'binned_histogram': bin_hists, 'residual_bins': residual_bins}, f)
        print(f"File {join(save_path, file_name)} saved.")
        
    else:
        print(f"File {join(save_path, file_name)} already exists, skipping computation.")

if __name__ == "__main__" :
    start_time = time.time()
    main()
    print(f"Script finished in {(time.time() - start_time)/60:.2f} minutes.")