"""

Dataset for faster inference.

"""

#######################################################################################################################
# Imports

from torch.utils.data import Dataset
import numpy as np
from model.dataset import biome_distribution, one_hot_encode, NODATAVALS

#######################################################################################################################
# Helper functions 

def embed_patch(biome_patch, cfg, embeddings = None, mode = False) :
    """
    This function embeds the biome patch, using the specified encoding.

    Args:
    - biome_patch: (np.ndarray) the biome patch to embed
    - cfg: (dict) the configuration of the model
    - embeddings: (np.ndarray) the embeddings to use for the cat2vec model

    Returns:
    - biome_emb: (np.ndarray) the embedded biome patch
    """

    # Get the index of the biome to be considered
    if mode: # the one that is the most frequent
        unique_values, counts = np.unique(biome_patch.flatten(), return_counts=True)
        most_frequent_value = unique_values[np.argmax(counts)]
        b_x, b_y = np.argwhere(biome_patch == most_frequent_value)[0]
    else: # the central pixel one
        b_x = b_y = biome_patch.shape[0] // 2
    
    # Encode the biome
    if cfg.get('emb_dist', False) : # biome distribution
        biome_emb = biome_distribution(biome_patch)
    elif cfg.get('emb_onehot', False) : # one-hot encoding
        biome_emb = one_hot_encode(biome_patch, 'lc')
    elif cfg.get('emb_cat2vec', False) : # cat2vec encoding
        biome_emb = np.vectorize(lambda x: embeddings.get(x, embeddings.get(0)), signature = '()->(n)')(biome_patch)
    elif cfg.get('emb_sincos', False): # sine cosine encoding
        lc_cos = np.where(biome_patch == NODATAVALS['LC'], 0, (np.cos(2 * np.pi * biome_patch / 100) + 1) / 2)
        lc_sin = np.where(biome_patch == NODATAVALS['LC'], 0, (np.sin(2 * np.pi * biome_patch / 100) + 1) / 2)
        biome_emb = np.array([lc_cos, lc_sin]).swapaxes(0, -1)
    else: raise ValueError('Invalid encoding for land cover data.')

    return biome_emb[b_x, b_y, :].astype(np.float32)


def symmetric_index(idx, size):
    idx[idx < 0] = -idx[idx < 0] - 1
    idx[idx >= size] = 2*size - idx[idx >= size] - 1
    return np.clip(idx, 0, size - 1)


def gaus2d(x = 0, y = 0, mx = 0, my = 0, sx = 1, sy = 1): 
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    
def get_patch_weight(src_height, src_width, factor = 5):
    """
    This function returns a 2D Gaussian weight matrix for the patch.

    Args:
    - src_height: (int) the height of the patch
    - src_width: (int) the width of the patch

    Returns:
    - weights: (np.array) the 2D Gaussian weight matrix for the patch
    """
    xmin, xmax = - src_height // 2, src_height // 2
    ymin, ymax = - src_width // 2, src_width // 2
    x = np.linspace(xmin, xmax, src_width)
    y = np.linspace(ymin, ymax, src_height)
    x, y = np.meshgrid(x, y)
    sx = (x.max() - x.min()) / factor
    sy = (y.max() - y.min()) / factor
    weights = gaus2d(x, y, sx=sx, sy=sy)
    return weights.astype(np.float32)

def padded_patch_v3(img, x1, x2, y1, y2, pad):
    """
    This function returns a patch of the image, padded symmetrically if needed.

    Args:
    - img: (np.ndarray) the image to extract the patch from
    - x1, x2, y1, y2: (int) the coordinates of the patch to extract
    - pad: (int) the amount of padding to apply symmetrically

    Returns:
    - indices: (tuple) the indices to extract the patch from the image
    """
    
    H, W = img.shape[:2]
    
    # Compute full padded coordinates
    x1p, x2p = x1 - pad, x2 + pad
    pred_height = x2p - x1p
    y1p, y2p = y1 - pad, y2 + pad
    pred_width = y2p - y1p
    
    # Whether symmetric padding is needed
    pad_rows = not (0 <= x1p and x2p <= H)
    pad_cols = not (0 <= y1p and y2p <= W)
    
    # Process the indices
    row_idx = np.arange(x1p, x2p)
    col_idx = np.arange(y1p, y2p)
    if pad_rows: 
        img_row_idx = symmetric_index(row_idx, H)
        pred_row_idx = np.arange(max(0, x1p), min(H, x2p))
        vert_crop = (np.abs(min(0, x1p)), min(H, x2p) - x1p)
    else: 
        img_row_idx = pred_row_idx = row_idx
        vert_crop = (0, pred_height)
    if pad_cols: 
        img_col_idx = symmetric_index(col_idx, W)
        pred_col_idx = np.arange(max(0, y1p), min(W, y2p))
        hor_crop = (np.abs(min(0, y1p)), min(W, y2p) - y1p)
    else: 
        img_col_idx = pred_col_idx = col_idx
        hor_crop = (0, pred_width)
    img_indices = np.ix_(img_row_idx, img_col_idx)
    pred_indices = np.ix_(pred_row_idx, pred_col_idx)
    return img_indices, pred_indices, vert_crop + hor_crop

############################################################################################################################################
# Inference dataset class definition

class InferenceDataset_v3(Dataset):

    def __init__(self, img, patch_size, pred_crop, cfg, embeddings = None, mode = False, factor = 5) :

        # Process the image ###########################################################################################
        self.film, self.region = cfg.get('film', False), cfg.get('region', False)
        if self.film :
            if self.region : img, self.biome, self.region_cla = img
            else: img, self.biome = img

        # Define variables for the splitting of the Sentinel-2 tile into patches ######################################

        # Width and height of the input Sentinel-2 tile
        img_height, img_width, _ = img.shape
        # Width and height of the desired patches
        patch_height, patch_width = patch_size
        assert np.unique(pred_crop).size == 1, "pred_crop should be a single value for all sides."
        pred_crop = int(pred_crop[0])  # Assuming pred_crop is a single value
        patch_height_no_border, patch_width_no_border = patch_height - 2 * pred_crop, patch_width - 2 * pred_crop
        # Find the number of times the patch will fit entirely in the image
        n_height = int(np.ceil(img_height / patch_height_no_border))
        n_width = int(np.ceil(img_width / patch_width_no_border))

        # Define variables for the predictions mosaicing ##############################################################

        images_indices = []
        for y in range(0, n_height) :
            y_coord = y * patch_height_no_border
            if y_coord > img_height - patch_height_no_border :
                # move last patch up if it would exceed the image bottom
                y_coord = img_height - patch_height_no_border
            for x in range(0, n_width) :
                x_coord = x * patch_width_no_border
                if x_coord > img_width - patch_width_no_border:
                    # move last patch left if it would exceed the image right border
                    x_coord = img_width - patch_width_no_border
                images_indices.append((y_coord, y_coord + patch_height_no_border, x_coord, x_coord + patch_width_no_border)) 
                # has shape (patch_height_no_border x patch_width_no_border), will be padded later to patch_height x patch_width
        self.images_indices = images_indices
    
        # Calculate the number of patches, without using the indices
        self.num_patches = n_height * n_width
        assert len(images_indices) == self.num_patches, f"Expected {self.num_patches} patches, but got {len(images_indices)} indices."

        # To store
        self.img = img
        self.cfg, self.embeddings = cfg, embeddings
        self.mode = mode
        self.pred_crop = pred_crop
        self.pred_height, self.pred_width = img_height, img_width

        # Define the patch weight
        self.patch_weight = get_patch_weight(patch_height, patch_width, factor)


    def __len__(self):
        return self.num_patches
    

    def __getitem__(self, idx):

        # Process the image
        x1, x2, y1, y2 = self.images_indices[idx]
        img_indices, pred_indices, (v1, v2, h1, h2) = padded_patch_v3(self.img, x1, x2, y1, y2, pad = self.pred_crop)
        patch = self.img[img_indices]  # Use the indices to extract the patch

        # Take care of FiLM embeddings
        if self.film :
            b_indices, _, _ = padded_patch_v3(self.biome, x1, x2, y1, y2, pad = self.pred_crop)
            biome_patch = self.biome[b_indices]
            biome_emb = embed_patch(biome_patch, self.cfg, self.embeddings, self.mode)
            if self.region : biome_emb = np.concatenate([biome_emb, self.region_cla], axis = 0)
        else: biome_emb = 0.0

        # Get the patch_weights
        pred_height, pred_width = pred_indices[0].shape[0], pred_indices[1].shape[1]
        patch_weights = self.patch_weight[:pred_height, :pred_width]

        return patch, biome_emb, pred_indices, patch_weights, (v1, v2, h1, h2)