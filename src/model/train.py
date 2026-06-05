"""

This script is the entry point for training the models. It is called by train.sh.

"""

###################################################################################################
# IMPORTS 

from config import DATA_ROOT
import os
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
from model.models import Net
from model.wrapper import Model
from model.parser import setup_parser, check_args
from model.dataset import GEDIDataset
from os.path import join
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch import set_float32_matmul_precision
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import logging
from wandb_utils import wandb_enabled
import numpy as np

# Debugging
torch._logging.set_logs(dynamo=logging.DEBUG)

try: 
    seed_everything(3 + int(os.environ.get('SLURM_ARRAY_TASK_ID')), workers = True)
except: 
    seed_everything(3, workers = True)
global_seed = torch.initial_seed()
os.environ["WANDB__SERVICE_WAIT"] = "300"

#####################################################################################################################################################
# Helper functions

def get_model_checkpoint_callback(dir, fname):
    return ModelCheckpoint(monitor = 'val/agbd_rmse', dirpath = dir, filename = f'{fname}_best', save_top_k = 1, mode = 'min', save_last = True)

def get_early_stopping_callback(patience, min_delta):
    return EarlyStopping(monitor = 'val/agbd_rmse', patience = patience, min_delta = min_delta, verbose = True)

def get_progress_bar():
    return TQDMProgressBar(refresh_rate = 1000)

#####################################################################################################################################################
# Code execution


def main():
    
    # Parse the arguments
    args, _ = setup_parser().parse_known_args()

    # Checking the arguments
    assert check_args(args) is True, 'Arguments are not valid.'

    # Settings
    set_float32_matmul_precision('high')
    cpus_per_task = args.num_cpus
    num_devices = args.num_gpus
    print(f"num_devices={num_devices} cpus_per_task={cpus_per_task}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve dataset paths. Use --dataset_path local for a local layout under DATA_ROOT
    # (configure DATA_ROOT in config.py or the environment), or pass an explicit root.
    if args.dataset_path == 'local' :
        cat2vec = f"{DATA_ROOT}/EcosystemAnalysis/Models/Baseline/cat2vec/{'AGBD' if not args.lite else 'AGBD-Lite'}"
        dataset_path = {'h5': f'{DATA_ROOT}/patches', 'norm': f'{DATA_ROOT}/patches',
                        'map': f'{DATA_ROOT}/patches', 'embeddings': cat2vec}
    else :
        dataset_path = {k: args.dataset_path for k in ['h5', 'norm', 'map', 'embeddings']}
    debug = False
    
    print('DEBUG :', debug)
    
    # Set up logging. Weights & Biases is optional (enable with USE_WANDB=true); by
    # default training runs without it.
    model_name = args.model_name.split('/')[-1]
    if wandb_enabled() :
        import os as _os
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(entity = _os.environ.get('WANDB_ENTITY'), project = args.arch,
                                   name = (None if model_name == 'local' else model_name), log_model = False)
        if model_name == 'local' : model_name = wandb_logger.experiment.name
    else :
        wandb_logger = False  # PyTorch Lightning disables logging when logger=False

    # Save the training configuration next to the checkpoint so eval/inference/predict can
    # run without Weights & Biases (they read this sidecar via load_train_config()).
    import os as _os, json as _json
    _os.makedirs(args.model_path, exist_ok=True)
    with open(_os.path.join(args.model_path, f'{model_name}_config.json'), 'w') as _f:
        _json.dump(vars(args), _f, default=str, indent=1)

    # Define the trainer
    trainer = Trainer(max_epochs = args.n_epochs,
                    accelerator = 'gpu', 
                    logger = wandb_logger,
                    num_sanity_val_steps = 1,
                    val_check_interval = 0.5,
                    callbacks = [get_early_stopping_callback(patience = args.patience, min_delta = args.min_delta), 
                                    get_model_checkpoint_callback(dir = args.model_path, fname = model_name),
                                    get_progress_bar()])

    # Log the run configuration (only when W&B logging is enabled)
    if wandb_logger :
        wandb_logger.experiment.config.update(args)
        wandb_logger.experiment.config.update({'num_devices': num_devices, 'cpus_per_task': cpus_per_task, 'debug': debug})

    # Build the network based on the architecture requested
    assert args.arch == 'nico_film', f"this release only supports the BioFiLM architecture ('nico_film'), got {args.arch}"

    # Define the model (PyTorch module)
    model = Net(model_name = args.arch, in_features = args.in_features, num_outputs = args.num_outputs,
                channel_dims = args.channel_dims, max_pool = args.max_pool, downsample = None,
                leaky_relu = args.leaky_relu, patch_size = args.patch_size,
                local = (args.dataset_path == 'local'), device = device,
                biome_dim = args.biome_dim, emb_dim = args.emb_dim,
                debug_film = args.debug_film, bn = args.bn, num_sepconv_blocks = args.num_sepconv_blocks,
                num_sepconv_filters = args.num_sepconv_filters, long_skip = args.long_skip, only_entry = args.only_entry,
                linear_emb = args.linear_emb, padding_mode = args.padding_mode, returns = args.returns, sigreg_lambda = args.sigreg_lambda,
                predict = args.predict)

    # Define the Model (pytorch lightning wrapper)
    model = Model(model, lr = args.lr, step_size = args.step_size, gamma = args.gamma, 
                    patch_size = args.patch_size, downsample = args.downsample,
                    loss_fn = args.loss_fn, film = args.film, debug_film = args.debug_film,
                    l2 = args.l2, crop = args.crop, sim_dist = args.sim_dist, 
                    similarity = args.similarity, similarity_weight = args.similarity_weight,
                    SCC_ws = args.SCC_ws, SCC_softmax = args.SCC_softmax,
                    log_transform = args.log_transform, predict = args.predict)
    
    # Define the datasets and dataloaders
    if args.subsample_2020 and 2020 in args.years:
        # Subsample the 2020 training set
        train_dataset_2020 = train_dataset = GEDIDataset(paths = dataset_path, years = [2020], chunk_size = args.chunk_size, mode = "train", args = args, debug = debug, film = args.film, mask_s2 = args.train_mask)
        rng = np.random.default_rng(seed = global_seed)
        indices = rng.choice(len(train_dataset_2020), size = 2509225, replace = False)
        train_dataset_2020_sub = Subset(train_dataset_2020, indices)
        if 2019 in args.years:
            train_dataset_2019 = GEDIDataset(paths = dataset_path, years = [2019], chunk_size = args.chunk_size, mode = "train", args = args, debug = debug, film = args.film, mask_s2 = args.train_mask)
            train_dataset = ConcatDataset([train_dataset_2019, train_dataset_2020_sub])
        else: train_dataset = train_dataset_2020_sub
    else: train_dataset = GEDIDataset(paths = dataset_path, years = args.years, chunk_size = args.chunk_size, mode = "train", args = args, debug = debug, film = args.film, mask_s2 = args.train_mask)

    val_dataset = GEDIDataset(paths = dataset_path, years = args.years, chunk_size = args.chunk_size, mode = "val", args = args, debug = debug, film = args.film, mask_s2 = args.val_mask)
    test_dataset = GEDIDataset(paths = dataset_path, years = args.years, chunk_size = args.chunk_size, mode = "test", args = args, debug = debug, film = args.film, mask_s2 = args.test_mask)

    bs = args.batch_size // args.lite_chunk_size if args.lite else args.batch_size
    train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True, num_workers = cpus_per_task, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = bs, shuffle = False, num_workers = cpus_per_task, pin_memory = True)

    bs = args.batch_size if ((not args.lite) or (args.lite and args.lite_eval_big)) else args.batch_size // args.lite_chunk_size
    test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = False, num_workers = cpus_per_task, pin_memory = True)

    # Train the model
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = [val_loader, test_loader])


if __name__ == '__main__':
    main()
