"""

This script loads an already trained Kriging model, and runs predictions on the specified Sentinel-2 tile.

"""

#######################################################################################################################
# Imports

from kriging.kriging import *

def parser():
    """ 
    Returns an `ArgumentParser()` object containing the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--kriging_model', type = str, required = True, help = 'Kriging model name.')
    parser.add_argument('--tile_id', type = str, required = True, help = 'Tile ID.')
    parser.add_argument('--s2_tile', type = str, required = True, help = 'S2 tile.')
    parser.add_argument('--year', type = int, required = True, help = 'Year of the prediction.')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--path_predictions', type = str, required = True, help = 'Directory with the predictions.')
    parser.add_argument('--path_dem', type = str, required = True, help = 'Directory with the DEM.')
    parser.add_argument('--path_kriging', type = str, required = True, help = 'Directory for Kriging.')
    parser.add_argument('--seed', type = int, default = 10, help = 'Random seed.')
    args = parser.parse_args()

    return args.kriging_model, args.tile_id, args.s2_tile, args.year, args.arch, args.path_predictions, args.path_dem, args.path_kriging, args.seed

#######################################################################################################################
# Code execution

if __name__ == '__main__':


    # Initialize everything #######################################################################
    program_start = time()

    # Parse the arguments
    kriging_model, tile_id, s2_tile, year, arch, path_predictions, path_dem, path_kriging, _seed = parser()
    model_name = f'{kriging_model}-{tile_id}'

    # Load the trained kriging checkpoint (model tensors + the config saved at fit time)
    ckpt_path = join(path_kriging, 'checkpoints', kriging_model)
    if not isdir(ckpt_path) : makedirs(ckpt_path, exist_ok=True)
    data = torch.load(join(ckpt_path, f'{model_name}_{s2_tile}.pt'), weights_only=False, map_location='cpu')
    train_x, train_y, likelihood_state, ndims, norm_values = data['train_x'], data['train_y'], data['likelihood_state'], data['ndims'], data['norm_values']

    # Recover the configuration used at fit time. Prefer the copy embedded in the
    # checkpoint (saved by kriging.py); for older checkpoints that predate it, fall
    # back to Weights & Biases (needs WANDB_ENTITY and a logged-in account).
    config = data.get('config')
    if config is None :
        import os, wandb
        entity = os.environ.get('WANDB_ENTITY')
        runs = wandb.Api().runs(f'{entity}/kriging', {'display_name': model_name})
        if not runs :
            raise RuntimeError(f"Checkpoint {model_name}_{s2_tile}.pt has no embedded config and no "
                               f"W&B run named '{model_name}' was found. Re-run kriging with the current "
                               f"code, or set WANDB_ENTITY to the entity that holds that run.")
        config = runs[0].config
    # arch/year fall back to the CLI args if an older run did not log them
    arch = config.get('arch') if config.get('arch') is not None else arch
    year = config.get('year') if config.get('year') is not None else year
    ens_models, matern_nu, COMPUTE_VAR, norm_res, coords, pred_vals, aux, extra_features, composites, agb, norm_coords, norm_aux = \
        config.get('ens_models'), config.get('matern_nu'), config.get('COMPUTE_VAR'), config.get('norm_res'), config.get('coords'), config.get('pred_vals'), config.get('aux'), config.get('extra_features'), config.get('composites'), config.get('agb'), config.get('norm_coords'), config.get('norm_aux')

    # Set the random seeds for reproducibility
    random.seed(_seed), np.random.seed(_seed), torch.manual_seed(_seed), torch.cuda.manual_seed_all(_seed)

    # Define the path to the dense predictions
    if composites : s2_path = join(path_predictions, arch, ens_models, f'{s2_tile}_{year}_composite.tif')
    else : s2_path = join(path_predictions, arch, s2_tile, str(year), ens_models, 'AGB_merged.tif')

    # Load necessary data
    pred_agb, pred_std, pred_mask, meta, transform, upsampling_shape, nodataval = load_dense_preds(s2_path = s2_path, aux = aux)
    dem = None
    extra_ft = get_extra_features(pred_agb, features = extra_features)
    valid_mask = (pred_mask == 0)
    
    # Define the likelihood
    # homoskedastic noise model (i.e. all inputs have the same observational noise)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.load_state_dict(likelihood_state)
    
    # Define the model
    model = ExactGPModel(train_x, train_y, likelihood, ndims = train_x.shape[1], matern_nu = matern_nu)

    # Load the model weights
    state_dict = torch.load(join(ckpt_path, f'{model_name}_{s2_tile}.ckpt'), map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval(), likelihood.eval()
    model.to('cuda')

    # Apply Kriging to the entire tile ############################################################################

    # Now, get the predictions on the entire tile
    start_time = time()
    print('    Predicting...')
    tile_x = get_tile_data(pred_agb, dem, pred_std, extra_ft, norm_values, coords, norm_coords, aux, norm_aux, pred_vals, extra_features)
    del dem, pred_std, extra_features
    pred_y = np.zeros_like(tile_x[:, 0], dtype='float')
    if COMPUTE_VAR : pred_y_var = np.zeros_like(tile_x[:, 0], dtype='float')
    model = model.to(torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        N = 500 # 1_000
        for i in range(0, pred_y.size, N):
            tile_x_crop = tile_x[i : min(i + N, pred_y.size)].cuda()
            observed_pred = likelihood(model(tile_x_crop))
            pred_y[i : min(i + N, pred_y.size)] = observed_pred.mean.cpu().numpy()
            if COMPUTE_VAR : pred_y_var[i : min(i + N, pred_y.size)] = observed_pred.variance.cpu().numpy()
    tile_x_crop = tile_x_crop.cpu().detach()
    tile_x = tile_x.cpu().detach()
    del(tile_x_crop, tile_x)
    torch.cuda.empty_cache()
    gc.collect()
    print(f'    Done! In {time() - start_time} seconds.')

    # Calculate the residuals on the entire tile
    residuals = pred_y.reshape(pred_agb.shape)
    if norm_res : residuals = residuals * norm_values['res']['std'] + norm_values['res']['mean']
    del(pred_y)
    if COMPUTE_VAR : 
        residuals_var = pred_y_var.reshape(pred_agb.shape)
        if norm_res : residuals_var *= np.pow(norm_values['res']['std'], 2)
        residuals_var = np.clip(residuals_var, 0, None) # Ensure non-negativity
        del(pred_y_var)

    # Apply the residuals to the predictions
    if agb : corrected = np.clip(residuals, 0, None)
    else: corrected = np.clip(pred_agb - residuals, 0, None)

    # Where the input data was undefined, mask the output
    corrected[~valid_mask] = np.nan
    if COMPUTE_VAR : 
        residuals_var[~valid_mask] = np.nan
    del valid_mask

    # Cleanup
    model, likelihood = model.to('cpu'), likelihood.to('cpu')
    del(model, likelihood)
    torch.cuda.empty_cache()
    gc.collect()

    if not agb: del(pred_agb)

    ###################################################################################################################
    # Save the dense predictions
    start_time = time()
    print('Saving the corrected prediction...')
    count = 3 if COMPUTE_VAR else 2
    meta.update(count = count, dtype = 'float32', nodata = np.nan)
    tif_path = join(path_kriging, 'predictions', arch, s2_tile, str(year), ens_models)
    with rs.open(join(tif_path, f"kriging-{model_name}.tif"), 'w', **meta) as dst:
        # Write the corrected AGB predictions
        dst.write(corrected, 1)
        dst.set_band_description(1, 'AGB')
        # Write the residuals
        if agb : residuals = pred_agb - residuals
        dst.write(residuals, 2)
        dst.set_band_description(2, 'Residuals')
        # Write the STD
        if COMPUTE_VAR :
            dst.write(np.sqrt(residuals_var), 3)
            dst.set_band_description(3, 'STD')
    print(f'Done! In {time() - start_time} seconds.')

    ###################################################################################################################
    # Measure total time taken
    total_time = time() - program_start
    print(f'Total time taken: {total_time} seconds.')
