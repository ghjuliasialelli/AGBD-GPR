"""

DESCRIPTION - This script performs inference on a Sentinel-2 tile, using a trained model. The script loads the input
data, splits the Sentinel-2 tile into patches, and predicts the AGBD for each patch. The predictions are then mosaiced
to obtain the final AGBD map for the Sentinel-2 tile. The script saves the AGBD map as a GeoTIFF file. This script is
different from the normal inference.py script in that it takes as input multiple products from the same S2 tile, and
performs inference on each product separately, then merges the predictions. It is used in the inference_merge.sh script.

"""

#######################################################################################################################
# Imports

from config import DATA_ROOT
import time
from os.path import join, isfile
import os, pickle, argparse
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
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
from inference.inference_residuals import init_args_dataset
from inference.inference_ds import InferenceDataset_v3
from torch.utils.data import DataLoader
import random
import gc
import psutil
import cv2

# Silencing specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")

import torch._dynamo
torch._dynamo.config.suppress_errors = True

random.seed(10)

#######################################################################################################################
# Helper functions 

def inf_parser():
    """ 
    Main function. Returns an `ArgumentParser()` object containing the command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type = str, required = True, help = 'Path to the dataset')
    parser.add_argument('--models', type = str, nargs = '+', required = True, help = 'Model names')
    parser.add_argument('--arch', type = str, required = True, help = 'Architecture of the model')
    parser.add_argument('--entity', type = str, default = f'{WANDB_ENTITY}', help = 'wandb entity for the model.')
    parser.add_argument('--saving_dir', type = str, help = 'Directory in which to save the predictions.')
    parser.add_argument("--products", required = True, type = str, nargs = '+', help = 'Products on which to run the prediction.')
    parser.add_argument("--batch_size", type = int, default = 2, help = 'Batch size for the dataloader.')
    parser.add_argument("--patch_size", nargs = 2, type = int, default = [200,200], help = 'Size (height,width) of the patches.')
    parser.add_argument("--pred_crop", nargs = 4, type = int, default = [0, 0, 0, 0], help = 'Pixels to crop off the predictions (off_ht, off_wl, off_hb, off_wr).')
    parser.add_argument("--mask_pred", type = str2bool, default = 'false', help = 'Whether to mask the final prediction.')
    parser.add_argument("--save_scl_mask", type = str2bool, default = 'true', help = 'Whether to save the SCL mask of individual products.')
    parser.add_argument('--dtype', type = str, default = 'float32', help = 'Data type to save the predictions.')
    parser.add_argument("--mode", type = str2bool, default = 'false', help = 'Whether to use mode for biome embedding.')
    parser.add_argument("--factor", type = float, default = 5, help = 'Factor for the Gaussian weights.')
    parser.add_argument("--reduction", required = True, type = str, help = 'How to merge.')
    parser.add_argument("--force", type = str2bool, default = 'false', help = 'Whether to force re-computation of existing outputs.')
    args = parser.parse_args()

    return args, args.dataset_path, args.models, args.arch, args.entity, args.saving_dir, args.products, \
        args.batch_size, args.patch_size, args.pred_crop, args.dtype, args.mode, args.factor, args.reduction, \
        args.mask_pred, args.save_scl_mask, args.force

process = psutil.Process(os.getpid())
def print_current_RAM() :
    #total, used, _ = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    mem = process.memory_info().rss  # in bytes
    print(f"RAM usage: {mem / (1024**3):.2f} GB")


def load_s2_bands(s2_bands, boa_offset, norm_values, cfg, s2_prod, patch_size) :
    """
    This function loads the Sentinel-2 bands from the product, normalizes them, and returns them as a list.

    Args:
    - boa_offset (float): the offset to apply to the bands
    - norm_values (dict): a dictionary containing the normalization values for the bands
    - cfg (dict): a dictionary containing the configuration for the model
    - s2_prod (str): the name of the Sentinel-2 product
    - patch_size (tuple): the size of the patch to extract from the Sentinel-2 product

    Returns:
    - s2_data (list): a list containing the normalized Sentinel-2 bands
    """

    # Initialize the data
    s2_dim = len(s2_bands) + (1 if cfg.get('s2_day', False) else 0) + (2 if cfg.get('s2_doy', False) else 0)
    s2_data = np.empty(shape = (patch_size[0], patch_size[1], s2_dim), dtype = np.float32)
    count_bands = 0

    # 3. Get the SR values for the optical bands
    for band, band_value in s2_bands.items() :
        s2_bands[band] = (band_value - boa_offset * 1000) / 10000

    # 4. Normalize the data
    s2_order = cfg['bands']
    s2_bands = np.moveaxis(np.array([s2_bands[band] for band in s2_order]), 0, -1)
    s2_bands = normalize_bands(s2_bands, norm_values['S2_bands'], s2_order, cfg['norm_strat'], NODATAVALS['S2'])
    
    #s2_data.extend([s2_bands])
    s2_data[:, :, : s2_bands.shape[-1]] = s2_bands
    count_bands += s2_bands.shape[-1]
    del s2_bands, s2_order
    gc.collect()

    # Sentinel-2 dates
    if cfg.get('s2_dates', False) :
        s2_date = datetime.strptime(s2_prod.split('_')[2][:8], '%Y%m%d')
        s2_num_days = (s2_date - datetime.strptime('2019-04-17', '%Y-%m-%d')).days
        s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, patch_size)
        s2_num_days = np.full((patch_size[0], patch_size[1]), s2_num_days).astype(np.float32)
        s2_num_days = normalize_data(s2_num_days, norm_values['Sentinel_metadata']['S2_date'], 'min_max' if cfg['norm_strat'] == 'pct' else cfg['norm_strat'])
        if cfg.get('s2_day', False) : 
            #s2_data.extend([s2_num_days[..., np.newaxis]])
            s2_data[:, :, count_bands : count_bands + 1] = s2_num_days[..., np.newaxis]
            count_bands += 1
        if cfg.get('s2_doy', False) :
            #s2_data.extend([s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
            s2_data[:, :, count_bands : count_bands + 1] = s2_doy_cos[..., np.newaxis]
            s2_data[:, :, count_bands + 1 : count_bands + 2] = s2_doy_sin[..., np.newaxis]
            count_bands += 2
        del s2_date, s2_num_days, s2_doy_cos, s2_doy_sin
        gc.collect()
    
    #s2_dim = sum([d.shape[-1] for d in s2_data])
    
    return s2_data, s2_dim


def load_input(year, paths, tile_name, product_name, norm_values, cfg, alos_order = ['HH', 'HV'], embeddings = None, debug = False, s2_only = False):    
    """ 
    Reads the input tile specified in tile_name, as well as the corresponding encoded geographical coordinates,
    and normalize the input.

    Args:
    - year (int): the year of the product
    - paths (dict): a dictionary containing the paths to the dataset
    - tile_name (str): the name of the Sentinel-2 tile
    - product_name (str): the name of the Sentinel-2 product
    - norm_values (dict): a dictionary containing the normalization values for the bands
    - cfg (dict): a dictionary containing the configuration for the model
    - alos_order (list): the order of the ALOS bands to use
    - embeddings (dict): a dictionary containing the cat2vec embeddings for the land cover classes
    - debug (bool): whether to run in debug mode (default: False)

    Returns:
    - data (tuple): a tuple containing the tile's data
    - scl_band (np.ndarray): the SCL band of the tile
    - meta (dict): a dictionary containing the metadata of the tile
    """
    
    start_time = time.time()
    print('Loading input...')

    # Initialize the data

    # Sentinel 2 bands -------------------------------------------------------------------------------------------

    # 1. Get the product
    s2_prod = product_name

    # 2. Process the product
    print('pre process_s2_tile')
    transform, upsampling_shape, s2_bands, crs, bounds, boa_offset, lat_cos, lat_sin, lon_cos, lon_sin, meta = process_S2_tile(s2_prod, paths['tiles'])
    print('done!')
    patch_size = (upsampling_shape[0], upsampling_shape[1])
    data = np.empty(shape = (upsampling_shape[0], upsampling_shape[1], cfg['in_features']), dtype = np.float32)
    bands_count = 0
    scl_band = s2_bands.pop('SCL')

    if debug: 
        data = np.zeros((patch_size[0], patch_size[1], 31)).astype(np.float32)

        if cfg.get('region', False) :
            with open(join(paths['region'], 's2_tile_to_region-v3.pkl'), 'rb') as f: region_mapping = pickle.load(f)
            region = region_mapping[tile_name]
            if isinstance(region, list) : region = region[0] # TODO, later, consider implementing multiple regions compatible code
            region = np.squeeze(one_hot_encode(np.full((1,1), region), 'region_cla').astype(np.float32))

        lc_raw = load_LC_data(paths['lc'], tile_name)
        lc_tile = get_tile(lc_raw, transform, upsampling_shape, 'LC', LC_attrs)
        lc = np.moveaxis(np.array([lc_tile['lc'], lc_tile['prob']]), 0, -1)
        biome = lc[..., 0]

        if cfg.get('film', False) : 
            if cfg.get('region', False) : data = (data, biome, region)
            else: data = (data, biome)
        print('done!')

        if s2_only: return data, scl_band, meta, 12
        else: return data, scl_band, meta 

    if cfg['bands'] != [] :
        print('bands--')

        print('pre load_s2_bands')
        s2_data, s2_dim = load_s2_bands(s2_bands, boa_offset, norm_values, cfg, s2_prod, patch_size)
        print('done!')
        #data.extend(s2_data)
        data[:, :, : s2_dim] = s2_data
        bands_count += s2_dim
        print('post extend')
        del s2_data, s2_bands, boa_offset
        gc.collect()

        # Return only the Sentinel-2 features if specified
        if s2_only :
            del lat_cos, lat_sin, lon_cos, lon_sin
            gc.collect()
            return torch.from_numpy(data[:, :, : s2_dim]), scl_band, meta
    else: s2_dim = None

    # Get the geographical coordinates ----------------------------------------------------------------------------
    print('latlon--')
    if cfg['latlon']: 
        # data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis], lon_cos[..., np.newaxis], lon_sin[..., np.newaxis]])
        data[:, :, bands_count : bands_count + 1] = lat_cos[..., np.newaxis]
        data[:, :, bands_count + 1 : bands_count + 2] = lat_sin[..., np.newaxis]
        data[:, :, bands_count + 2 : bands_count + 3] = lon_cos[..., np.newaxis]
        data[:, :, bands_count + 3 : bands_count + 4] = lon_sin[..., np.newaxis]
        bands_count += 4
    else: 
        # data.extend([lat_cos[..., np.newaxis], lat_sin[..., np.newaxis]])
        data[:, :, bands_count : bands_count + 1] = lat_cos[..., np.newaxis]
        data[:, :, bands_count + 1 : bands_count + 2] = lat_sin[..., np.newaxis]
        bands_count += 2
    del lat_cos, lat_sin, lon_cos, lon_sin
    gc.collect()

    # Get the GEDI dates -----------------------------------------------------------------------------------------
    if cfg.get('gedi_dates', False) :
        print('gedi_dates--')
        # At inference time, set the GEDI date to be the same as the Sentinel-2 date
        s2_date = datetime.strptime(s2_prod.split('_')[2][:8], '%Y%m%d')
        s2_num_days = (s2_date - datetime.strptime('2019-04-17', '%Y-%m-%d')).days
        s2_doy_cos, s2_doy_sin = get_doy(s2_num_days, patch_size)
        s2_num_days = np.full((patch_size[0], patch_size[1]), s2_num_days).astype(np.float32)
        s2_num_days = normalize_data(s2_num_days, norm_values['Sentinel_metadata']['S2_date'], 'min_max' if cfg['norm_strat'] == 'pct' else cfg['norm_strat'])
        # data.extend([s2_num_days[..., np.newaxis], s2_doy_cos[..., np.newaxis], s2_doy_sin[..., np.newaxis]])
        data[:, :, bands_count : bands_count + 1] = s2_num_days[..., np.newaxis]
        data[:, :, bands_count + 1 : bands_count + 2] = s2_doy_cos[..., np.newaxis]
        data[:, :, bands_count + 2 : bands_count + 3] = s2_doy_sin[..., np.newaxis]
        bands_count += 3
        del s2_date, s2_num_days, s2_doy_cos, s2_doy_sin
        gc.collect()

    # Get the ALOS data ------------------------------------------------------------------------------------------
    if cfg.get('alos', False) :
        print('alos--')
        # 1. Get the data
        alos_raw = load_ALOS_data(tile_name, paths['alos'], year)
        alos_tile = get_tile(alos_raw, transform, upsampling_shape, 'ALOS', ALOS_attrs)
        alos_bands = np.moveaxis(np.array([alos_tile['HH'], alos_tile['HV']]), 0, -1)
        # 2. Get the gamma naught values
        alos_bands = np.where(alos_bands == NODATAVALS['ALOS'], -9999.0, 10 * np.log10(np.power(alos_bands.astype(np.float32), 2)) - 83.0)
        # 3. Normalize the data
        alos_bands = normalize_bands(alos_bands, norm_values['ALOS_bands'], alos_order, cfg['norm_strat'], -9999.0)
        #data.extend([alos_bands])
        data[:, :, bands_count : bands_count + alos_bands.shape[-1]] = alos_bands
        del alos_bands, alos_raw, alos_tile
        gc.collect()

    # Get the CH data --------------------------------------------------------------------------------------------
    if cfg.get('ch', False) :
        print('ch--')
        # 1. Get the data
        ch_bands = load_CH_data(paths['ch'], tile_name, year)
        ch, ch_std = ch_bands['ch'], ch_bands['std']
        # 2. Normalize the data
        ch = normalize_data(ch, norm_values['CH']['ch'], cfg['norm_strat'], NODATAVALS['CH'])
        ch_std = normalize_data(ch_std, norm_values['CH']['std'], cfg['norm_strat'], NODATAVALS['CH'])
        #data.extend([ch[..., np.newaxis], ch_std[..., np.newaxis]])
        data[:, :, bands_count : bands_count + 1] = ch[..., np.newaxis]
        data[:, :, bands_count + 1 : bands_count + 2] = ch_std[..., np.newaxis]
        bands_count += 2
        del ch, ch_std, ch_bands
        gc.collect()
    
    # Get the LC data --------------------------------------------------------------------------------------------
    # 1. Get the data
    print('lc--')
    lc_raw = load_LC_data(paths['lc'], tile_name)
    lc_tile = get_tile(lc_raw, transform, upsampling_shape, 'LC', LC_attrs)
    lc = np.moveaxis(np.array([lc_tile['lc'], lc_tile['prob']]), 0, -1)
    biome = lc[..., 0]
    del lc_raw, lc_tile
    gc.collect()
    
    # 2. Transform the data
    if cfg.get('lc', False) :
        print('lc--')
        # TODO check if this can be done more efficiently
        if cfg.get('ft_onehot', False) :
            _, lc_prob = embed_lc(lc, embeddings)
            lc = one_hot_encode(lc[:, :, 0], 'lc')
            #data.extend([lc, lc_prob[..., np.newaxis]])
            data[:, :, bands_count : bands_count + lc.shape[-1]] = lc
            data[:, :, bands_count + lc.shape[-1] : bands_count + lc.shape[-1] + 1] = lc_prob[..., np.newaxis]
            bands_count += lc.shape[-1] + 1
            del lc, lc_prob
            gc.collect()
        elif cfg.get('ft_cat2vec', False) :
            lc, lc_prob = embed_lc(lc, embeddings)
            #data.extend([lc, lc_prob[..., np.newaxis]])
            data[:, :, bands_count : bands_count + lc.shape[-1]] = lc
            data[:, :, bands_count + lc.shape[-1] : bands_count + lc.shape[-1] + 1] = lc_prob[..., np.newaxis]
            bands_count += lc.shape[-1] + 1
            del lc, lc_prob
            gc.collect()
        elif cfg.get('ft_sincos', False) :
            lc_cos, lc_sin, lc_prob = encode_lc(lc)
            #data.extend([lc_cos[..., np.newaxis], lc_sin[..., np.newaxis], lc_prob[..., np.newaxis]])
            data[:, :, bands_count : bands_count + 1] = lc_cos[..., np.newaxis]
            data[:, :, bands_count + 1 : bands_count + 2] = lc_sin[..., np.newaxis]
            data[:, :, bands_count + 2 : bands_count + 3] = lc_prob[..., np.newaxis]
            bands_count += 3
            del lc_cos, lc_sin, lc_prob
            gc.collect()
        else: raise ValueError('Invalid encoding for land cover data.')

    # Get the DEM data -------------------------------------------------------------------------------------------
    if cfg.get('dem', False) :
        print('dem--')
        
        # 1. Get the data
        dem_raw = load_DEM_data(paths['dem'], tile_name)
        dem_tile = get_tile(dem_raw, transform, upsampling_shape, 'DEM', DEM_attrs)
        dem = dem_tile['dem']

        # 2. Get the slope and aspect
        if cfg.get('topo', False) :
            print('topo--')
            slope, aspect_cos, aspect_sin = get_topology(dem)
            if cfg.get('slope', False) : 
                #data.extend([slope[..., np.newaxis]])
                data[:, :, bands_count : bands_count + 1] = slope[..., np.newaxis]
                bands_count += 1
            if cfg.get('aspect', False) : 
                #data.extend([aspect_cos[..., np.newaxis], aspect_sin[..., np.newaxis]])
                data[:, :, bands_count : bands_count + 1] = aspect_cos[..., np.newaxis]
                data[:, :, bands_count + 1 : bands_count + 2] = aspect_sin[..., np.newaxis]
                bands_count += 2
            del slope, aspect_cos, aspect_sin
            gc.collect()

        # 3. Normalize the data
        dem = normalize_data(dem, norm_values['DEM'], cfg['norm_strat'], NODATAVALS['DEM'])
        #data.extend([dem[..., np.newaxis]])
        data[:, :, bands_count : bands_count + 1] = dem[..., np.newaxis]
        bands_count += 1
        del dem, dem_raw, dem_tile
        gc.collect()

    # Get the GEDI region class ----------------------------------------------------------------------------------
    if cfg.get('region', False) :
        print('region--')
        with open(join(paths['region'], 's2_tile_to_region-v3.pkl'), 'rb') as f: region_mapping = pickle.load(f)
        region = region_mapping[tile_name]
        if isinstance(region, list) : region = region[0] # TODO, later, consider implementing multiple regions compatible code
        region = np.squeeze(one_hot_encode(np.full((1,1), region), 'region_cla').astype(np.float32))

    # Check that the model requires no residuals -----------------------------------------------------------------
    assert cfg.get('residuals', False) == False, 'Model uses residuals. Please refer to inference_residuals.py'

    # Concatenate the data ---------------------------------------------------------------------------------------
    # TODO at this point, it takes up almost all of the memory
    print('concatenate--')
    data = torch.from_numpy(data)
    
    # Append the biome embedding if FiLM is enabled --------------------------------------------------------------
    print('film--')
    if cfg.get('film', False) : 
        if cfg.get('region', False) : 
            data = (data, biome, region)
            del region, biome
            gc.collect()
        else: 
            data = (data, biome)
            del biome
            gc.collect()

    # ------------------------------------------------------------------------------------------------------------
        
    print('done!')
    end_time = time.time()
    print(f'Loading input took {end_time - start_time} seconds.')

    return data, scl_band, meta, s2_dim


def predict_patch(model, patch, device, biome_emb = None):
    """
    Predict patch for AGBD.

    Args:
    - model: (torch.nn.Module) the model to use for prediction
    - patch: (np.ndarray) the patch to predict
    - device: (torch.device) the device on which to perform inference
    - biome_emb: (torch.Tensor or None) the biome embedding to use for prediction, if applicable

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
                    leaky_relu = self.args.leaky_relu, patch_size = self.args.patch_size, local = (self.args.dataset_path == 'local'), device = self.device, biome_dim = self.args.biome_dim, emb_dim = self.args.emb_dim,
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

def run_inference():

    # Get the command line arguments and set the global variables
    args, dataset_path, models, arch, entity, saving_dir, products, \
        batch_size, patch_size, pred_crop, _dtype, mode, factor, reduction, \
        mask_pred, save_scl_mask, FORCE = inf_parser()


    # Settings
    set_float32_matmul_precision('high')
    if (dataset_path == 'local') : accelerator, cpus_per_task = 'auto', 8
    else: accelerator, cpus_per_task = 'gpu', int(os.environ.get('SLURM_CPUS_PER_TASK'))
    if cpus_per_task is None: cpus_per_task = 16
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths
    if dataset_path == 'local' : 
        dataset_path = {'norm': f'{DATA_ROOT}/patches',
                           'tiles': f'{DATA_ROOT}/S2_L2A',
                           'ch': f'{DATA_ROOT}/CH',
                           'ckpt': f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/weights',
                           'alos': f'{DATA_ROOT}/ALOS',
                           'dem': f'{DATA_ROOT}/ALOS',
                           'lc': f'{DATA_ROOT}/LC',
                           'region' : f'{DATA_ROOT}/BiomassDatasetCreation/Data/download_Sentinel',
                           'embeddings': f'{DATA_ROOT}/EcosystemAnalysis/Models/Baseline/cat2vec',
                           'esa': f'{DATA_ROOT}/WorldCover/S2'
                           }
    else:
        dataset_path = {'norm': f'{DATA_ROOT}/Data/patches',
                           'tiles': f'{DATA_ROOT}/Data/S2_L2A',
                           'ch': f'{DATA_ROOT}/EcosystemAnalysis/Models/Nico/global-canopy-height-model',
                           'ckpt': f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/weights',
                           'alos': f'{DATA_ROOT}/Data/ALOS',
                           'dem': f'{DATA_ROOT}/Data/ALOS',
                           'lc': f'{DATA_ROOT}/Data/LC',
                           'region': f'{DATA_ROOT}/Data',
                           'embeddings': f'{DATA_ROOT}/EcosystemAnalysis/Models/Baseline/cat2vec',
                           'esa': f'{DATA_ROOT}/data/ESA_WorldCover/data/ESA_WorldCover_10m_2020_v100_sentinel2_tiles',
                           'esa_backup': f'{DATA_ROOT}/Data/WorldCover'
                           }
    dataset_path['saving_dir'] = saving_dir

    # We get the config for one of the models
    cfg = load_train_config(dataset_path['ckpt'], arch, models[0], entity=entity)
    for key, value in cfg.items(): setattr(args, key, value)
    args = init_args_dataset(args) # add the missing arguments to the config

    # Get the year and tile name from the first product
    year = int(products[0].split('_')[2][:4])
    tile_name = products[0].split('_')[5].lstrip('T')

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

    # Iterate over the products and perform inference
    num_products = len(products)
    all_same_year = (len(set([int(product.split('_')[2][:4]) for product in products])) == 1)
    for i, product_name in enumerate(products):
        print(f'Processing product {i+1}/{num_products}...')

        # Check if it's already been processed
        output_path = join(dataset_path['saving_dir'], arch, tile_name, str(year), '_'.join(models))
        pred_exists = isfile(os.path.join(output_path, f'AGB_{product_name}.tif'))
        mask_exists = isfile(os.path.join(dataset_path['tiles'], f'{product_name}_MASK.tif'))
        if (not FORCE) and ((pred_exists and not save_scl_mask) or (pred_exists and save_scl_mask and mask_exists)):
            print(f'Predictions for {product_name} already exist. Skipping...')
            continue

        # Load the product
        print_current_RAM()
        print('pre-load input...')

        if all_same_year :
            if i == 0 : img, pred_mask, meta, s2_dim = load_input(year, dataset_path, tile_name, product_name, norm_values, cfg, embeddings = embeddings)
            else: img[0][..., :s2_dim], pred_mask, meta = load_input(year, dataset_path, tile_name, product_name, norm_values, cfg, embeddings = embeddings, s2_only = True)
        else: img, pred_mask, meta, _ = load_input(year, dataset_path, tile_name, product_name, norm_values, cfg, embeddings = embeddings)
        print('done.')
        print_current_RAM()

        # Take care of downsampling, if needed
        if cfg['downsample'] : raise Exception('Downsampling is not supported in this script.')

        # Get the ensemble predictions
        print_current_RAM()
        print('Creating dataset and dataloader...')
        
        # Get the ensemble predictions
        dataset = InferenceDataset_v3(img, patch_size, pred_crop, cfg, embeddings = embeddings, mode = mode, factor = factor)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = cpus_per_task)
        predictions = efficient_predict_tile_v3(dataloader, inf_models, device, dataset.pred_height, dataset.pred_width)

        # Get the average predictions
        avg_preds_variables = np.nanmean(predictions, axis = 0)
        # Where the SCL mask is NODATA, mask the prediction
        avg_preds_variables[pred_mask == 0] = np.nan
        if len(models) > 1 : avg_preds_std = np.nanstd(predictions, axis = 0)

        # Take care of the data type
        if _dtype == 'uint16' :
            dtype, nodata = np.uint16, 65535
            avg_preds_variables[avg_preds_variables > 65535] = 65535
            if len(models) > 1 : avg_preds_std[avg_preds_std > 65535] = 65535
        elif _dtype == 'float32' : 
            dtype, nodata = np.float32, -9999.0
        else: raise Exception(f'Invalid dtype: {_dtype}.')

        # Cast the data to the appropriate range/data type
        avg_preds_variables[avg_preds_variables < 0] = 0
        avg_preds_variables[np.isinf(avg_preds_variables)] = nodata
        avg_preds_variables[np.isnan(avg_preds_variables)] = nodata
        avg_preds_variables = avg_preds_variables.astype(dtype)
        if len(models) > 1 :
            avg_preds_std[np.isinf(avg_preds_std)] = nodata
            avg_preds_std[np.isnan(avg_preds_std)] = nodata
            avg_preds_std = avg_preds_std.astype(dtype)

        # Save the predictions
        print(f'Saving predictions...')
        meta.update(driver = 'GTiff', dtype = dtype, count = 2 if len(models) > 1 else 1, compress = 'lzw', nodata = nodata)
        if not os.path.exists(output_path): os.makedirs(output_path)
        with rs.open(os.path.join(output_path, f'AGB_{product_name}.tif'), 'w', **meta) as f:
            f.write(avg_preds_variables, 1)
            f.set_band_description(1, 'AGB')
            if len(models) > 1 :
                f.write(avg_preds_std, 2)
                f.set_band_description(2, 'STD')
        print('done!')

        # Save the mask if needed
        if save_scl_mask :
            print(f'Saving mask for {product_name}...')
            print_current_RAM()
            meta.update(driver = 'GTiff', dtype = 'uint8', count = 1, compress = 'lzw', nodata = 255)
            with rs.open(os.path.join(dataset_path['tiles'], f'{product_name}_MASK.tif'), 'w', **meta) as f:
                f.write(pred_mask, 1)
                f.set_band_description(1, 'SCL')
            print_current_RAM()
            print('done!')

        # Delete the input variables to free memory
        print('Deleting input variables to free memory...')
        print_current_RAM()
        del(pred_mask, dataset, dataloader, predictions, avg_preds_variables, avg_preds_std)
        gc.collect()
        print_current_RAM()
        print('done!')

    # Merge the predictions ###########################################################################################

    if not 'meta' in locals() : # Load the meta from one of the products
        output_path = join(dataset_path['saving_dir'], arch, tile_name, str(year), '_'.join(models))
        with rs.open(join(output_path, f'AGB_{products[0]}.tif'), 'r') as src:
            meta = src.meta

    print('Merging predictions...')
    print_current_RAM()

    # Load the ESA WorldCover mask (one per tile per year)
    if mask_pred:
        print('Loading ESA WorldCover mask...')
        if isfile(join(dataset_path['esa'], f'ESA_WorldCover_10m_2020_v100_{tile_name}.tif')):
            with rs.open(join(dataset_path['esa'], f'ESA_WorldCover_10m_2020_v100_{tile_name}.tif')) as src:
                worldcover = src.read(1)
        elif isfile(join(dataset_path['esa_backup'], f'ESA_WorldCover_10m_2020_v100_{tile_name}.tif')):
            with rs.open(join(dataset_path['esa_backup'], f'ESA_WorldCover_10m_2020_v100_{tile_name}.tif')) as src:
                worldcover = src.read(1)
        else: raise FileNotFoundError(f'ESA WorldCover mask not found for tile {tile_name}.')
        # Permanent water bodies (80) or nodata (0) or built-up (50) or snow and ice (70)
        esa_mask = (worldcover == 80) | (worldcover == 0) | (worldcover == 50) | (worldcover == 70)
        print_current_RAM()
    
    # Iterate over the products and read the predictions
    print('Reading predictions...')
    preds, product_masks = [], []
    if len(models) > 1 : stds = []
    for product_name in products:
        output_path = join(dataset_path['saving_dir'], arch, tile_name, str(year), '_'.join(models), f'AGB_{product_name}.tif')
        with rs.open(output_path) as src:
            pred = src.read(1)
            pred[pred == src.nodata] = np.nan
            if src.count > 1 :
                std = src.read(2)
                std[std == src.nodata] = np.nan

        # SCL band
        output_path = join(dataset_path['tiles'], f'{product_name}_MASK.tif')
        with rs.open(output_path) as src: scl = src.read(1)
        scl_mask = (scl == 0) | (scl == 1) # No Data (0) or Saturated or defective pixel (1)
        pred[scl_mask] = np.nan
        product_masks.append(scl_mask)
        if len(models) > 1 : std[scl_mask] = np.nan

        if mask_pred:            
            # ESA WorldCover mask
            pred[esa_mask] = np.nan
            if len(models) > 1 : std[esa_mask] = np.nan
        
        preds.append(pred)
        if len(models) > 1 : stds.append(std)

    # If there are NODATA values in one of the SCL masks, crop the predictions
    if any([mask.any() for mask in product_masks]) :
        for i, mask in enumerate(product_masks):
            # do it with a constant value of 5 pixels
            r = 5
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((2*r+1,2*r+1), dtype = np.uint8), iterations = 1).astype(bool)
            preds[i][mask] = np.nan

    # Merge the predictions, across products
    if reduction == 'median' : 
        merged_preds = np.nanmedian(np.array(preds), axis = 0)
        if len(models) > 1 : merged_preds_std = np.nanmedian(np.array(stds), axis = 0)
    
    elif reduction == 'mean' : 
        merged_preds = np.nanmean(np.array(preds), axis = 0)
        if len(models) > 1 : merged_preds_std = np.nanmean(np.array(stds), axis = 0)

    else: raise ValueError("Invalid reduction method. Use 'mean' or 'median'.")
    print('done!')

    # Take care of the data type
    if _dtype == 'uint16' :
        dtype, nodata = np.uint16, 65535
        merged_preds[merged_preds > 65535] = 65535
        if len(models) > 1 : merged_preds_std[merged_preds_std > 65535] = 65535
    elif _dtype == 'float32' : 
        dtype, nodata = np.float32, -9999.0
    else: raise Exception('Invalid dtype.')

    # Cast the data to the appropriate range/data type
    merged_preds[merged_preds < 0] = 0
    merged_preds[np.isinf(merged_preds)] = nodata
    merged_preds[np.isnan(merged_preds)] = nodata
    merged_preds = merged_preds.astype(dtype)
    if len(models) > 1 :
        merged_preds_std[np.isinf(merged_preds_std)] = nodata
        merged_preds_std[np.isnan(merged_preds_std)] = nodata
        merged_preds_std = merged_preds_std.astype(dtype)

    # Save the predictions
    print(f'Saving merged predictions...')
    meta.update(driver = 'GTiff', dtype = dtype, count = 2 if len(models) > 1 else 1, compress = 'lzw', nodata = nodata)
    output_path = join(dataset_path['saving_dir'], arch, tile_name, str(year), '_'.join(models))
    fname = f'AGB_merged_{reduction}.tif'
    with rs.open(join(output_path, fname), 'w', **meta) as f:
        f.write(merged_preds, 1)
        f.set_band_description(1, 'AGB')
        if len(models) > 1 :
            f.write(merged_preds_std, 2)
            f.set_band_description(2, 'STD')
    print('done!')


if __name__ == '__main__':
    t0 = time.time()
    run_inference()
    ttotal = time.time() - t0
    print(f'Inference done! in: {str(timedelta(seconds=ttotal))}.')