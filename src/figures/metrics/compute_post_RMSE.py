import os
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
"""

This script computes the post-kriging RMSE on all of the tiles, taking care of problematic footprints.

Run with the `krige` environment, on the cluster, with the following command:
    sbatch --wrap="python compute_post_RMSE.py --model_name <model_name> --force --subset --test" --time=4:00:00 --mem-per-cpu=4G --cpus-per-task=1 --output=post_RMSE-%A.out --error=post_RMSE-%A.out

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
import geopandas as gpd
from os.path import isfile

REF_BIOMES = {
    '20': 'Shrubs', 
    '30': 'HV', 
    '40': 'Crops', 
    '90': 'HW', 
    '111': 'C-ENL', 
    '112': 'C-EBL', 
    '114': 'C-DBL', 
    '115': 'C-M', 
    '116': 'C-O', 
    '121': 'O-ENL', 
    '122': 'O-EBL', 
    '124': 'O-DBL', 
    '125': 'O-M', 
    '126': 'O-O'
}

biomes = list([int(b) for b in REF_BIOMES.keys()])

#######################################################################################################################
# Helper functions

def _parser() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, required = True, help = 'Model name identifier.')
    parser.add_argument('--force', action='store_true', help = 'Whether to force recomputation even if the file already exists.')
    parser.add_argument('--subset', action='store_true', help = 'Whether to use a subset of the data.')
    parser.add_argument('--test', action='store_true', help = 'Whether to compute only on the test data.')
    parser.add_argument('--ref_model', type = str, required = False, help = 'Model to compare for the test set.')
    parser.add_argument('--CCI', action='store_true', help = 'Whether to eval CCI preds.')
    args = parser.parse_args()
    return args.model_name, args.force, args.subset, args.test, args.ref_model, args.CCI

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

def compute_metrics(post, ref) :
    """
    Computes the RMSE between the post-kriging predictions and the reference values.

    Args:
    - post (np.ndarray): The post-kriging predictions.
    - ref (np.ndarray): The reference values.

    Returns:
    - res (dict): A dictionary containing the overall RMSE and number of footprints.
    - binned_res (dict): A dictionary containing the binned RMSE and number of footprints.
    """
    res = {}
    binned_res = {}

    # Remove NaNs
    valid_mask = ~np.isnan(post) & ~np.isnan(ref)
    if np.sum(valid_mask) == 0 :
        print("No valid footprints found!")
        return {}, {}
    post = post[valid_mask]
    ref = ref[valid_mask]

    # Overall metrics
    me = np.mean(post - ref)
    mae = np.mean(np.abs(post - ref))
    rmse = np.sqrt(np.mean(np.power(post - ref, 2)))
    num_footprints = len(post)
    res = {'rmse': rmse, 'num_footprints': num_footprints, 'me': me, 'mae': mae}

    # Binned metrics

    bins = np.arange(0, 501, 50)
    lbs, ubs = bins[:-1], bins[1:]
    for lb, ub in zip(lbs, ubs) :
        bin_mask = (ref >= lb) & (ref < ub)
        if np.sum(bin_mask) > 0 :
            bin_rmse = np.sqrt(np.mean((post[bin_mask] - ref[bin_mask]) ** 2))
            bin_num_footprints = np.sum(bin_mask)
            binned_res[f'{lb}-{ub}'] = {'rmse': bin_rmse, 'num_footprints': bin_num_footprints}

    return res, binned_res

#######################################################################################################################
# Code execution

def main() :

    # Arguments #######################################################################################################

    model_name, force, subset, test_set, ref_model, CCI = _parser()
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

    # If test_set, need to load the config from wandb
    if test_set : import wandb; api = wandb.Api()  # W&B account required only for this lookup

    if not CCI :
        save_path = join(path_kriging, 'predictions', arch, str(year), inf_model)
        file_name = f"{model_name}-post{'_subset' if subset else ''}{'_test' if test_set else ''}.pkl"
    else:
        save_path = join(path_kriging, 'predictions', 'CCI')
        file_name = f"CCI-post{'_subset' if subset else ''}{'_test' if test_set else ''}.pkl"

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

        # Compute RMSE ####################################################################################################

        # Define the bins
        bins = np.arange(0, 501, 50)
        lbs, ubs = bins[:-1], bins[1:]
        _bins = [f'{lb}-{ub}' for lb, ub in zip(lbs, ubs)]
        
        # Iterate over the regions
        all_results = {}
        for region in regions :
            print(f"Processing region {region}...")
            results = {'overall': {'rmse': [], 'num_footprints': [], 'me': [], 'mae': []}, 
                       'binned' : {bin: {'rmse': [], 'num_footprints': []} for bin in _bins},
                       'biome': {biome: {'rmse': [], 'num_footprints': []} for biome in biomes}}
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
                        with open(join(tif_path, f"results-{model_name}-{tile_idx}.pkl"), 'rb') as f :
                            tile_data = pickle.load(f)
                        post, ref = np.array(tile_data['post']), np.array(tile_data['ref'])
                        # Load the indices of the tile
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
                        fname = f"splits-{tile_idx}_{test_holdout}_{val_holdout}_{max_train_footprints}_{max_split_diff}_{stripe_size}{'_ood' if ood else ''}.pkl"
                        with open(join(save_dir, fname), 'rb') as f:
                            test_indices = pickle.load(f)['test']
                        ok_mask = np.isin(idx, test_indices)
                    # Or on the full set, but removing the problematic indices
                    else:
                        tile_problematic_indices = problematic_indices[np.isin(problematic_tiles, tile, invert = True)]
                        ok_mask = np.isin(idx, tile_problematic_indices, invert = True)

                    # Load the biomes.pkl file of the tile
                    with open(join(path_kriging, 'predictions', 'biomes', f"biomes-{tile_idx}.pkl"), 'rb') as f :
                        tile_biomes = pickle.load(f)['biomes']

                    # Compute the metrics
                    res, binned_res = compute_metrics(post[ok_mask], ref[ok_mask])
                    if res == {} : continue
                    results['overall']['rmse'].append(res['rmse'])
                    results['overall']['num_footprints'].append(res['num_footprints'])
                    results['overall']['me'].append(res['me'])
                    results['overall']['mae'].append(res['mae'])
                    for bin, vals in binned_res.items() :
                        results['binned'][bin]['rmse'].append(vals['rmse'])
                        results['binned'][bin]['num_footprints'].append(vals['num_footprints'])
                    
                    # Compute biome-specific metrics
                    non_prob_post, non_prob_ref = post[ok_mask], ref[ok_mask]
                    non_prob_biomes = tile_biomes[ok_mask]
                    for biome in biomes :
                        biome_mask = (non_prob_biomes == biome)
                        if np.sum(biome_mask) > 0 :
                            biome_res, _ = compute_metrics(non_prob_post[biome_mask], non_prob_ref[biome_mask])
                            if biome_res == {} : continue
                            results['biome'][biome]['rmse'].append(biome_res['rmse'])
                            results['biome'][biome]['num_footprints'].append(biome_res['num_footprints'])
                    
                except Exception as e :
                    print(f"        !!! could not process tile {tile}, skipping it. Error: {e}")
                    continue

            # Save the results for the region
            all_results[region] = results


        # Compute the overall RMSE ##############################################################################################
        
        overall_rmse = {'overall': {'rmse': np.nan, 'num_footprints': 0, 'me': np.nan, 'mae': np.nan}, 
                        'binned' : {bin: {'rmse': np.nan, 'num_footprints': 0} for bin in _bins},
                        'biome': {biome: {'rmse': np.nan, 'num_footprints': 0} for biome in biomes}}
        
        total_mse = 0
        total_me = 0
        total_mae = 0
        total_num_footprints = sum([sum(all_results[region]['overall']['num_footprints']) for region in regions])
        
        binned_total_mse = {bin: 0 for bin in _bins}
        binned_total_num_footprints = {bin: sum([sum(all_results[region]['binned'][bin]['num_footprints']) for region in all_results]) for bin in _bins}

        biome_total_mse = {biome: 0 for biome in biomes}
        biome_total_num_footprints = {biome: sum([sum(all_results[region]['biome'][biome]['num_footprints']) for region in all_results]) for biome in biomes}
        
        for results in all_results.values() :
            
            # Overall
            rmses, num_footprints = np.array(results['overall']['rmse']), np.array(results['overall']['num_footprints'])
            mes, maes = np.array(results['overall']['me']), np.array(results['overall']['mae'])
            if len(rmses) == 0 : continue
            
            rmses = np.power(rmses, 2)
            num_footprints = num_footprints / total_num_footprints
            total_mse += np.sum(rmses * num_footprints)

            total_me += np.sum(mes * num_footprints)
            total_mae += np.sum(maes * num_footprints)
            
            # Binned
            binned_results = results['binned']
            for bin in binned_results.keys() :
                rmses, num_footprints = np.array(binned_results[bin]['rmse']), np.array(binned_results[bin]['num_footprints'])
                if len(rmses) > 0 :
                    rmses = np.power(rmses, 2)
                    num_footprints = num_footprints / binned_total_num_footprints[bin]
                    binned_total_mse[bin] += np.sum(rmses * num_footprints)

            # Biomes
            biome_results = results['biome']
            for biome in biome_results.keys() :
                rmses, num_footprints = np.array(biome_results[biome]['rmse']), np.array(biome_results[biome]['num_footprints'])
                if len(rmses) > 0 :
                    rmses = np.power(rmses, 2)
                    num_footprints = num_footprints / biome_total_num_footprints[biome]
                    biome_total_mse[biome] += np.sum(rmses * num_footprints)
        
        for bin in _bins :
            if binned_total_num_footprints[bin] > 0 :
                overall_rmse['binned'][bin]['rmse'] = np.sqrt(binned_total_mse[bin])
                overall_rmse['binned'][bin]['num_footprints'] = binned_total_num_footprints[bin]
            else :
                overall_rmse['binned'][bin]['rmse'] = None
                overall_rmse['binned'][bin]['num_footprints'] = 0
        
        for biome in biomes :
            if biome_total_num_footprints[biome] > 0 :
                overall_rmse['biome'][biome]['rmse'] = np.sqrt(biome_total_mse[biome])
                overall_rmse['biome'][biome]['num_footprints'] = biome_total_num_footprints[biome]
            else :
                overall_rmse['biome'][biome]['rmse'] = None
                overall_rmse['biome'][biome]['num_footprints'] = 0
        
        overall_rmse['overall']['rmse'] = np.sqrt(total_mse)
        overall_rmse['overall']['me'] = total_me
        overall_rmse['overall']['mae'] = total_mae
        print(f"Overall RMSE: {overall_rmse['overall']['rmse']:.2f}t/ha on {total_num_footprints} footprints.")
        print(f"Overall ME: {overall_rmse['overall']['me']:.2f}t/ha.")
        print(f"Overall MAE: {overall_rmse['overall']['mae']:.2f}t/ha.")
        overall_rmse['overall']['num_footprints'] = total_num_footprints

        all_results['overall'] = overall_rmse
        
        # Save the results #################################################################################################
        if not exists(save_path) : makedirs(save_path, exist_ok = True)
        with open(join(save_path, file_name), 'wb') as f :
            pickle.dump(all_results, f)
    
    else: print(f"File {join(save_path, file_name)} already exists, skipping computation.")
        

if __name__ == "__main__" :
    start_time = time.time()
    main()
    print(f"Script finished in {(time.time() - start_time)/60:.2f} minutes.")