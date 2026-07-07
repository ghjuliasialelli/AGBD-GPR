"""
Kriging (GP residual correction) of OUR full-resolution prediction, tile by tile, evaluated
against GEDI. Launched by kriging.sh when pred="ours" and downsampled="false".

Thin entry point: it loops over the Sumatra S2 tiles and runs the shared pipeline
(sumatra/pipeline.py) once per tile. With --ood true, the pipeline additionally removes
out-of-distribution GEDI footprints before fitting (see run_kriging_for_map / get_train_val_test_split).

Run: configure and launch via kriging.sh (do not call the args by hand).
"""

from os.path import join

from config import DATA_ROOT
from sumatra.common import parser
from sumatra.pipeline import run_kriging_for_map, set_seeds

# Sumatra S2 tiles covered by our full-resolution prediction.
TILES = ['47MRV', '48MTE', '48MUE', '47MRU', '48MTD', '48MUD', '47MRT', '48MTC', '48MUC']

# Directory holding the per-tile full-res predictions to correct (S2 acquisition year 2021).
FULLRES_PRED_DIR = join(DATA_ROOT, 'EcosystemAnalysis', 'Models', 'Biomes', 'predictions', 'nico_film')


if __name__ == '__main__':
    args = parser()
    set_seeds(args.seed)
    ens = '_'.join(args.ens_models)
    for s2_tile in TILES:
        print('-' * 80)
        print(f'Processing tile {s2_tile}...\n')
        path_predictions = join(FULLRES_PRED_DIR, ens, f"{s2_tile}_2021_composite.tif")
        model_name = f'{args.model_name}-{s2_tile}'
        run_kriging_for_map(args, path_predictions, model_name, s2_tile=s2_tile)
