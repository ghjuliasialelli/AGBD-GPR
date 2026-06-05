"""
Model factory.

This release uses a single architecture: BioFiLM, a FiLM-conditioned, Xception-style
fully-convolutional encoder-decoder (`NicoNet_FiLM`). The other architectures explored
during development (plain NicoNet, U-Net variants, FCN, MLP, linear probe, Gaussian
heads) are not part of this paper and have been removed.

`Net` keeps its original constructor signature so existing checkpoints and call sites
(train.py, eval.py, inference*.py) continue to work; only the BioFiLM branch is built.
"""

from model.nico_net_film import NicoNet_FiLM
from model.biomes import REF_BIOMES
import torch.nn as nn


class Net(nn.Module):
    """Thin wrapper around the BioFiLM model (`NicoNet_FiLM`)."""

    def __init__(self, model_name, emb_dim, in_features=4, num_outputs=1, channel_dims=(16, 32, 64, 128),
                 max_pool=False, downsample=None, leaky_relu=False, patch_size=[15, 15],
                 local=False, device='cpu', biome_dim=128, debug_film=False, bn='yes',
                 num_sepconv_blocks=8, num_sepconv_filters=728, long_skip=False, only_entry=False,
                 linear_emb=False, padding_mode='zeros', returns="dense", sigreg_lambda=0.0, predict='agbd'):
        super(Net, self).__init__()

        self.model_name = model_name
        self.num_outputs = num_outputs
        self.biomes = list(REF_BIOMES.keys())
        self.returns = returns
        self.predict = predict

        # Classification head, only used for biome prediction
        if self.predict == 'biome':
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        else:
            self.pool = nn.Identity()

        # BioFiLM (the only supported architecture in this release)
        if self.model_name == "nico_film":
            self.model = NicoNet_FiLM(in_features=in_features, num_outputs=num_outputs, emb_dim=emb_dim, biome_dim=biome_dim,
                                      num_sepconv_blocks=num_sepconv_blocks, num_sepconv_filters=num_sepconv_filters,
                                      long_skip=long_skip, returns=returns, patch_size=patch_size[0], only_entry=only_entry,
                                      linear_emb=linear_emb, padding_mode=padding_mode, sigreg_lambda=sigreg_lambda)
        else:
            raise NotImplementedError(
                f"Architecture '{model_name}' is not part of this release; only 'nico_film' (BioFiLM) is supported."
            )

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, tuple):
            y, latents = y
        y = self.pool(y)
        return y if not isinstance(y, tuple) else (y, latents)
