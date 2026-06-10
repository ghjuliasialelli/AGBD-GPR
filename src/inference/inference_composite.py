"""

This script performs inference, as the inference.py file, except that it is tailored to run inference on a composite
of Sentinel-2 products. Therefore, some pre-processing steps differ.

The script can be executed by running: bash inference/inference_composite.sh.

"""

#######################################################################################################################
# Imports

from config import PATHS
import time
from os.path import join, isdir, isfile
import os, pickle, argparse
import torch
import numpy as np
import rasterio as rs
from torch import set_float32_matmul_precision
from model.models import Net
from model.wrapper import Model
from torch import set_float32_matmul_precision
from inference.inference_helper import *
from model.dataset import normalize_bands, normalize_data, encode_lc, one_hot_encode, embed_lc, get_doy, get_topology
import warnings
from model.parser import str2bool
from datetime import timedelta, datetime
import pandas as pd
from inference.inference_ds import InferenceDataset_v3
from torch.utils.data import DataLoader

# Silencing specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

#######################################################################################################################
# Helper functions 

def inf_parser(argv=None):
    """ 
    Main function. Returns an `ArgumentParser()` object containing the command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type = int, required = True, help = 'Year to do inference on.')
    parser.add_argument('--models', type = str, nargs = '+', required = True, help = 'Model names')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--saving_dir', type = str, help = 'Directory in which to save the plots.')
    parser.add_argument("--tile_name", required = True, type = str, help = 'Tile on which to run the prediction.')
    parser.add_argument("--method", required = True, type = str, help = 'Method used for the composites.')
    parser.add_argument("--batch_size", type = int, default = 2, help = 'Batch size for the dataloader.')
    parser.add_argument("--patch_size", nargs = 2, type = int, default = [200,200], help = 'Size (height,width) of the patches.')
    parser.add_argument("--pred_crop", nargs = 4, type = int, default = [0, 0, 0, 0], help = 'Pixels to crop off the predictions (off_ht, off_wl, off_hb, off_wr).')
    parser.add_argument("--masking", type = str2bool, default = 'false', help = 'Whether to mask the input.')
    parser.add_argument('--dtype', type = str, default = 'float32', help = 'Data type to save the predictions.')
    parser.add_argument("--mode", type = str2bool, default = 'false', help = 'Whether to use mode for biome embedding.')
    parser.add_argument("--std", type = str2bool, default = 'true', help = 'Whether to compute and save the STDs in case of ensembling.')
    parser.add_argument("--factor", type = float, default = 5, help = 'Factor for the Gaussian weights.')
    args = parser.parse_args(argv)

    return args, args.year, args.models, args.arch, args.saving_dir, args.tile_name, args.method, args.patch_size, args.pred_crop, args.masking, args.dtype, args.mode, args.std, args.batch_size, args.factor


def process_composite(composite, path_tiles, masking):
    """
    This function processes the composite Sentinel-2 data for a given tile, extracting the bands and upsampling them to a 10m resolution.

    Args:
    - composite: str, name of the composite (e.g., 'T31TCH_2020_composite_median')
    - path_tiles: str, path to the directory containing the Sentinel-2 tiles

    Returns:
    - _transform: rasterio.transform.Affine, the transform of the 10m resolution B02 band
    - upsampling_shape: tuple, the shape of the upsampled bands
    - processed_bands: dict, a dictionary containing the upsampled bands
    - crs: rasterio.crs.CRS, the CRS of the Sentinel-2 tile
    - bounds: tuple, the bounds of the Sentinel-2 tile
    - lat_cos: np.ndarray, cosine of the latitude coordinates
    - lat_sin: np.ndarray, sine of the latitude coordinates
    - lon_cos: np.ndarray, cosine of the longitude coordinates
    - lon_sin: np.ndarray, sine of the longitude coordinates
    - meta: dict, metadata of the Sentinel-2 tile
    """

    # Unzip the composite folder if necessary
    if not isdir(join(path_tiles, composite)) :
        if isfile(join(path_tiles, f'{composite}.zip')) :
            print(f'Unzipping {composite}.zip...')
            shutil.unpack_archive(join(path_tiles, f'{composite}.zip'), extract_dir = path_tiles)
        else: raise FileNotFoundError(f'No composite found for {composite}.')
    else: print(f'Composite {composite} already unzipped.')

    # Iterate over the bands
    processed_bands = {}
    for res, bands in S2_L2A_BANDS.items() :
        for band in bands :

            if not masking and band == 'SCL' : continue

            # Read the band data
            with rs.open(join(path_tiles, composite, f'{band}.tif')) as src :
                band_data = src.read(1)
                transform = src.transform
                crs = src.crs
                bounds = src.bounds

                # Extract the lat/lon for one of the bands
                if band == 'B02' :
                    transformer = Transformer.from_crs(crs, 'EPSG:4326')
                    lat_cos, lat_sin, lon_cos, lon_sin = encode_tile(src, transformer)
                    meta = src.meta

            # Turn the band into a 2d array
            if len(band_data.shape) == 3 : band_data = band_data[0, :, :]

            # Use the 10m resolution B02 band as reference
            if res == '10m' :
                if band == 'B02' :
                    # Base the other bands' upsampling on this band's
                    upsampling_shape = band_data.shape
                    # Save the transform of this band
                    _transform = transform
            
            # Upsample the band to a 10m resolution if necessary
            else :
                # Order 0 indicates nearest interpolation, and order 3 indicates bi-cubic interpolation
                if band == 'SCL' :
                    band_data = upsampling_with_nans(band_data, upsampling_shape, NODATAVALS['S2'], 0).astype(S2_attrs['bands'][band])
                else:
                    band_data = upsampling_with_nans(band_data.astype(np.float32), upsampling_shape, NODATAVALS['S2'], 1).astype(S2_attrs['bands'][band])

            # Store the transformed band
            assert band_data.shape == (upsampling_shape[0], upsampling_shape[1])
            processed_bands[band] = band_data
    
    # Delete the unzipped folder
    shutil.rmtree(join(path_tiles, composite))

    return _transform, upsampling_shape, processed_bands, crs, bounds, None, lat_cos, lat_sin, lon_cos, lon_sin, meta


def load_input(year, alos_year, paths, tile_name, method, norm_values, cfg, alos_order = ['HH', 'HV'], embeddings = None, masking = False):    
    """ 
    Reads the input tile specified in tile_name, as well as the corresponding encoded geographical coordinates,
    and normalize the input.

    Args:
    - year: int, the year of the composite
    - paths: dict, paths to the dataset
    - tile_name: str, the name of the Sentinel-2 tile (e.g., '58GGR')
    - method: str, the method used for the composite (e.g., 'median')
    - norm_values: dict, normalization values for the bands
    - cfg: dict, configuration dictionary containing the model settings
    - alos_order: list, the order of the ALOS bands to use
    - embeddings: dict, embeddings for the land cover data

    Returns:
    - data: list, the input data for the model
    - mask: np.ndarray, the mask for the input data
    - meta: dict, metadata for the input data
    """
    
    start_time = time.time()
    print('Loading input...')

    # Initialize the data
    data = []

    # Sentinel 2 bands -------------------------------------------------------------------------------------------    
    
    # 2. Process the composite
    composite = f'{tile_name}_{year}_composite_{method}'
    transform, upsampling_shape, s2_bands, crs, bounds, _, lat_cos, lat_sin, lon_cos, lon_sin, meta = process_composite(composite, paths['tiles'], masking)
    patch_size = (upsampling_shape[0], upsampling_shape[1])
    if masking: scl_band = s2_bands.pop('SCL')

    if cfg['bands'] != [] : 

        # Normalize the data
        s2_order = cfg['bands']
        s2_bands = np.moveaxis(np.array([s2_bands[band] for band in s2_order]), 0, -1)
        s2_bands = normalize_bands(s2_bands, norm_values['S2_bands'], s2_order, cfg['norm_strat'], NODATAVALS['S2'])
        data.extend([s2_bands])
    
        # Sentinel-2 dates
        if cfg.get('s2_dates', False) :
            # Create a date that's the middle of the year
            s2_date = datetime.strptime(f'{year}-07-01', '%Y-%m-%d')
            s2_num_days = (s2_date - datetime.strptime('2019-04-17', '%Y-%m-%d')).days
            s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, patch_size)
            s2_num_days = np.full((patch_size[0], patch_size[1]), s2_num_days).astype(np.float32)
            s2_num_days = normalize_data(s2_num_days, norm_values['Sentinel_metadata']['S2_date'], 'min_max' if cfg['norm_strat'] == 'pct' else cfg['norm_strat'])
            if cfg.get('s2_day', False) : 
                data.extend([s2_num_days[..., np.newaxis]])
            if cfg.get('s2_doy', False) :
                data.extend([s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
    
    # Get the geographical coordinates ----------------------------------------------------------------------------
    if cfg['latlon']: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
    # else: data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis]])

    # Get the GEDI dates -----------------------------------------------------------------------------------------
    if cfg.get('gedi_dates', False) :
        data.extend([s2_num_days[..., np.newaxis], s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])

    # Get the ALOS data ------------------------------------------------------------------------------------------
    if cfg.get('alos', False) :
        # 1. Get the data
        alos_raw = load_ALOS_data(tile_name, paths['alos'], alos_year)
        alos_tile = get_tile(alos_raw, transform, upsampling_shape, 'ALOS', ALOS_attrs)
        alos_bands = np.moveaxis(np.array([alos_tile['HH'], alos_tile['HV']]), 0, -1)
        # 2. Get the gamma naught values
        alos_bands = np.where(alos_bands == NODATAVALS['ALOS'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)
        # 3. Normalize the data
        alos_bands = normalize_bands(alos_bands, norm_values['ALOS_bands'], alos_order, cfg['norm_strat'], -9999.0)
        data.extend([alos_bands])

    # Get the CH data --------------------------------------------------------------------------------------------
    if cfg.get('ch', False) :
        # 1. Get the data
        ch_bands = load_CH_data(paths['ch'], tile_name, year)
        ch, ch_std = ch_bands['ch'], ch_bands['std']
        # 2. Normalize the data
        ch = normalize_data(ch, norm_values['CH']['ch'], cfg['norm_strat'], NODATAVALS['CH'])
        ch_std = normalize_data(ch_std, norm_values['CH']['std'], cfg['norm_strat'], NODATAVALS['CH'])
        data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])
    
    # Get the LC data --------------------------------------------------------------------------------------------
    # 1. Get the data
    lc_raw = load_LC_data(paths['lc'], tile_name)
    lc_tile = get_tile(lc_raw, transform, upsampling_shape, 'LC', LC_attrs)
    lc = np.moveaxis(np.array([lc_tile['lc'], lc_tile['prob']]), 0, -1)
    biome = lc[..., 0]
    
    # 2. Transform the data
    if cfg.get('lc', False) :
        if cfg.get('ft_onehot', False) :
            _, lc_prob = embed_lc(lc, embeddings)
            lc = one_hot_encode(lc[:, :, 0], 'lc')
            data.extend([lc, lc_prob[..., np.newaxis]])
        elif cfg.get('ft_cat2vec', False) :
            lc, lc_prob = embed_lc(lc, embeddings)
            data.extend([lc, lc_prob[..., np.newaxis]])
        elif cfg.get('ft_sincos', False) :
            lc_cos, lc_sin, lc_prob = encode_lc(lc)
            data.extend([lc_cos[..., np.newaxis], lc_sin[..., np.newaxis], lc_prob[..., np.newaxis]])
        else: raise ValueError('Invalid encoding for land cover data.')

    # Get the DEM data -------------------------------------------------------------------------------------------
    if cfg.get('dem', False) :
        
        # 1. Get the data
        dem_raw = load_DEM_data(paths['dem'], tile_name)
        dem_tile = get_tile(dem_raw, transform, upsampling_shape, 'DEM', DEM_attrs)
        dem = dem_tile['dem']

        # 2. Get the slope and aspect
        if cfg.get('topo', False) :
            slope, aspect_cos, aspect_sin = get_topology(dem)
            if cfg.get('slope', False) : data.extend([slope[..., np.newaxis]])
            if cfg.get('aspect', False) : data.extend([aspect_cos[..., np.newaxis], aspect_sin[..., np.newaxis]])

        # 3. Normalize the data
        dem = normalize_data(dem, norm_values['DEM'], cfg['norm_strat'], NODATAVALS['DEM'])
        data.extend([dem[..., np.newaxis]])

    # Get the GEDI region class ----------------------------------------------------------------------------------
    if cfg.get('region', False) :
        with open(join(paths['region'], 's2_tile_to_region-v3.pkl'), 'rb') as f: region_mapping = pickle.load(f)
        region = region_mapping[tile_name]
        if isinstance(region, list) : region = region[0] # TODO, later, consider implementing multiple regions compatible code
        region = np.squeeze(one_hot_encode(np.full((1,1), region), 'region_cla').astype(np.float32))

    # Concatenate the data ---------------------------------------------------------------------------------------
    data = torch.from_numpy(np.concatenate(data, axis = -1)).to(torch.float)
    
    # Append the biome embedding if FiLM is enabled --------------------------------------------------------------
    if cfg.get('film', False) : 
        if cfg.get('region', False) : data = (data, biome, region)
        else: data = (data, biome)

    # Get the mask -----------------------------------------------------------------------------------------------
    # i.e. where it is Water (6) and Snow or ice (11) or No Data (0) or Saturated or defective pixel (1)
    if masking: mask = (scl_band == 6) | (scl_band == 11) | (scl_band == 0) | (scl_band == 1)
    else: mask = None

    print('done!')
    end_time = time.time()
    print(f'Loading input took {end_time - start_time} seconds.')

    return data, mask, meta


def predict_patch(model, patch, device, biome_emb = None):
    """
    Predict patch for AGBD.

    Args:
    - model: (torch.nn.Module) the model to use for prediction
    - patch: (np.ndarray) the patch to predict
    - device: (torch.device) the device on which to perform inference
    - biome_emb: (torch.Tensor) the biome embedding to use for prediction, if applicable

    Returns:
    - preds: (np.ndarray) the predicted AGBD patch
    """

    # Transform the input patch for prediction
    if len(patch.shape) == 3: # (features, height, width)
        patch = torch.unsqueeze(torch.permute(patch, [2,0,1]), 0).to(device)
        if biome_emb is not None: 
            biome_emb = torch.tensor(np.expand_dims(biome_emb, axis = 0)).to(device)
            preds = model.model((patch, biome_emb)).cpu().detach().numpy()[0, 0, :, :]
        else: preds = model.model(patch).cpu().detach().numpy()[0, 0, :, :]
    elif len(patch.shape) == 4: # (batch, features, height, width)
        patch = torch.permute(patch, [0, 3, 1, 2]).to(device)
        if biome_emb is not None: 
            biome_emb = biome_emb.to(device)
            preds = model.model((patch, biome_emb)).cpu().detach().numpy()[:, 0, :, :]
        else: preds = model.model(patch).cpu().detach().numpy()[:, 0, :, :]
    else: raise ValueError('The patch should have either 3 or 4 dimensions.')

    return preds


def efficient_predict_tile_v3(dataloader, models, device, pred_height, pred_width):
    """
    This function predicts the AGBD for a Sentinel-2 tile, using a list of models, and a dataloader.
    This approach takes the Gaussian weighted average of overlapping patches, while padding the borders
    of the tile with symmetric padding to avoid edge effects.
    
    Args:
    - dataloader: (torch.utils.data.DataLoader) the dataloader to use for prediction
    - models: (list) the models to use for prediction
    - device: (torch.device) the device on which to perform inference
    - pred_height: (int) the height of the predicted AGBD
    - pred_width: (int) the width of the predicted AGBD
    
    Returns:
    - predictions: (np.ndarray) the predicted AGBD for the Sentinel-2 tile
    """
    
    print('Starting prediction...')
    
    # Placeholder for the predictions
    summed_predictions = np.full(shape = (len(models), pred_height, pred_width), fill_value = np.nan)
    sum_weights = np.full(shape = (len(models), pred_height, pred_width), fill_value = 0.0)
    
    # Iterate over the batches
    for batch in dataloader :
        
        # Unpack the batch
        patch, biome_emb, pred_indices, patch_weights, crop_indices = batch
        x_indices, y_indices = pred_indices # indices to find the position of the patch in summed_predictions and sum_weights
        v1s, v2s, h1s, h2s = crop_indices # indices to crop the prediction to remove the padded data
        
        # Iterate over the models
        for model_dim, model in enumerate(models) :

            preds = predict_patch(model, patch, device, biome_emb)
            cropped_preds = preds[:, v1s : v2s, h1s : h2s] # crop the predictions to remove the padded data
            
            # Iterate over the predictions
            for i in range(len(preds)) :
                
                # Indices to find the position of the patch in summed_predictions and sum_weights
                indices = (x_indices[i].numpy(), y_indices[i].numpy())

                # Get the weighted prediction for the patch
                patch_weight = patch_weights[i].numpy()
                weighted_pred = cropped_preds * patch_weight

                # Update summed_predictions, taking care of NaN values
                pred_patch = summed_predictions[(model_dim,) + indices]
                summed_predictions[(model_dim,) + indices] = np.where(np.isnan(pred_patch), weighted_pred, pred_patch + weighted_pred)

                # Update sum_weights
                sum_weights[(model_dim,) + indices] += patch_weight
    
    # Reduce the predictions by the weights
    if np.any(sum_weights == 0): print("Warning: There are weights equal to 0. This may lead to NaN values in the predictions.")
    predictions = np.where(sum_weights > 0, summed_predictions / sum_weights, np.nan)
    print('done!')
    
    return predictions

#######################################################################################################################
# Inference class definition

class Inference:
    """ 
    An `Inference` object loads a PyTorch model and performs AGBD inference at the Sentinel-2 tile level.
    """

    def __init__(self, arch, model_name, paths, tile_name, args, device):
        """
        Initialization method.

        Args:
        - arch (str) : the architecture of the model
        - model_name (str) : the name of the model
        - paths (dict) : the paths to the dataset
        - tile_name (str) : the name of the Sentinel-2 tile
        - args (argparse.Namespace) : the command-line arguments
        - device (torch.device) : the device on which to perform

        Returns:
        - None
        """

        self.arch = arch
        self.model_name = model_name
        self.paths = paths
        self.tile_name = tile_name
        self.args = args     
        self.device = device
        self.load_model()
    
    def load_model(self):
        """ 
        Loads the model, setting self.model.
        """

        # Initialize the model
        model = Net(model_name = self.arch, in_features = self.args.in_features, num_outputs = self.args.num_outputs, 
                    channel_dims = self.args.channel_dims, max_pool = self.args.max_pool, downsample = None,
                    leaky_relu = self.args.leaky_relu, patch_size = self.args.patch_size, local = False, device = self.device, biome_dim = self.args.biome_dim, emb_dim = self.args.emb_dim,
                    debug_film = self.args.debug_film, bn = self.args.bn, num_sepconv_blocks = self.args.num_sepconv_blocks, 
                    num_sepconv_filters = self.args.num_sepconv_filters, long_skip = self.args.long_skip, only_entry = self.args.only_entry, 
                    linear_emb = self.args.linear_emb, padding_mode = self.args.padding_mode, returns = self.args.returns)

        model = Model(model, lr = self.args.lr, step_size = self.args.step_size, gamma = self.args.gamma, 
                        patch_size = self.args.patch_size, downsample = self.args.downsample, 
                        loss_fn = self.args.loss_fn, film = self.args.film, debug_film = self.args.debug_film)
    
        state_dict = torch.load(join(self.paths['ckpt'], self.arch, f'{self.model_name}_best.ckpt'), map_location = torch.device(self.device), weights_only = True)['state_dict']
        state_dict = {k:v for k,v in state_dict.items() if 'teacher' not in k}
        model.load_state_dict(state_dict) 
        # add the following line if the nico model is not compiled : state_dict = {k.replace('_orig_mod.',''):v for k,v in state_dict.items()}
        
        model.to(self.device)
        model.eval()
        model.model.eval()
        self.model = model.model

#######################################################################################################################
# Code execution

def run_inference(argv=None):
    
    # Get the command line arguments and set the global variables
    args, year, models, arch, saving_dir, tile_name, method, patch_size, pred_crop, masking, dtype, mode, std, batch_size, factor = inf_parser(argv)

    if tile_name in ['10SFG', '10SFH', '10TEK', '10TEL', '11SLT', '11SLU', '11SLV', '11SMT', '11SMU', '11SMV', '48SXG', '48SYE', '48SYF', '49SDS', '49SDT', '49SDU'] : alos_year = 2019
    else: alos_year = year

    # Settings
    set_float32_matmul_precision('high')
    cpus_per_task = 8
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths (sourced from configs/<env>.yaml via config.PATHS).
    dataset_path = dict(PATHS)
    dataset_path['saving_dir'] = saving_dir

    # We get the config for one of the models
    with open(join(dataset_path['ckpt'], f'{models[0]}.pkl'), 'rb') as f: cfg = pickle.load(f)
    for key, value in cfg.items(): setattr(args, key, value)
    args = init_args_dataset(args) # add the missing arguments to the config

    # Load the models
    inference_objects = [Inference(arch = arch, model_name = model_name, paths = dataset_path, tile_name = tile_name, args = args, device = device) for model_name in models]
    inf_models = [inference_object.model for inference_object in inference_objects]

    # Load the cat2vec embeddings if needed
    if cfg.get('ft_cat2vec', False) or cfg.get('emb_cat2vec', False) :
        embeddings = pd.read_csv(join(dataset_path['embeddings'], 'embeddings_train.csv'))
        embeddings = dict([(v,np.array([a,b,c,d,e])) for v, a,b,c,d,e in zip(embeddings.mapping, embeddings.dim0, embeddings.dim1, embeddings.dim2, embeddings.dim3, embeddings.dim4)])
    else: embeddings = None

    # Load the input
    new_stats, prob_norm = cfg.get('new_stats', False), cfg.get('prob_norm', False)
    with open(os.path.join(dataset_path['norm'], f"statistics_subset_2019-2020-v4{'-1' if (new_stats or prob_norm) else ''}.pkl"), mode = 'rb') as f: norm_values = pickle.load(f)
    img, pred_mask, meta = load_input(year, alos_year, dataset_path, tile_name, method, norm_values, cfg, embeddings = embeddings, masking = masking)

    # Get the ensemble predictions
    dataset = InferenceDataset_v3(img, patch_size, pred_crop, cfg, embeddings = embeddings, mode = mode, factor = factor)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = cpus_per_task)
    predictions = efficient_predict_tile_v3(dataloader, inf_models, device, dataset.pred_height, dataset.pred_width)

    # Get the average predictions
    avg_preds_variables = np.nanmean(predictions, axis = 0)
    if len(models) > 1 : avg_preds_std = np.nanstd(predictions, axis = 0)

    # Take care of the data type
    if dtype == 'uint16' :
        dtype, nodata = np.uint16, 65535
        avg_preds_variables[avg_preds_variables > 65535] = 65535
        if len(models) > 1 : avg_preds_std[avg_preds_std > 65535] = 65535
    elif dtype == 'float32' : 
        dtype, nodata = np.float32, -9999.0
    else: raise Exception('Invalid dtype.')

    # Cast the data to the appropriate range/data type
    avg_preds_variables[avg_preds_variables < 0] = 0
    avg_preds_variables[np.isinf(avg_preds_variables)] = nodata
    avg_preds_variables[np.isnan(avg_preds_variables)] = nodata
    avg_preds_variables = avg_preds_variables.astype(dtype)
    if len(models) > 1 :
        avg_preds_std[np.isinf(avg_preds_std)] = nodata
        avg_preds_std[np.isnan(avg_preds_std)] = nodata
        avg_preds_std = avg_preds_std.astype(dtype)

    # Mask the predictions if needed
    if masking :
        avg_preds_std[pred_mask] = nodata
        avg_preds_variables[pred_mask] = nodata

    # Save the AGB predictions to GeoTIFF, with dtype uint16
    meta.update(driver = 'GTiff', dtype = dtype, count = 2 if len(models) > 1 else 1, compress = 'lzw', nodata = nodata)
    output_path = join(dataset_path['saving_dir'], arch, '_'.join(models))
    if not os.path.exists(output_path): os.makedirs(output_path)
    with rs.open(os.path.join(output_path, f'{tile_name}_{year}_composite.tif'), 'w', **meta) as f:
        f.write(avg_preds_variables, 1)
        f.set_band_description(1, 'AGB')
        if len(models) > 1 and std:
            f.write(avg_preds_std, 2)
            f.set_band_description(2, 'STD')

if __name__ == '__main__':
    t0 = time.time()
    run_inference()
    ttotal = time.time() - t0
    print(f'Inference done! in: {str(timedelta(seconds=ttotal))}.')