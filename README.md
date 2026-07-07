# BioFiLM + GPR for bias correction of satellite-based biomass estimates

Code for the paper **"Gaussian Process Regression for Bias Correction of Satellite-based
Biomass Estimates"** (Sialelli, Peters, Scheibenreif, Wegner, Schindler).

The repository contains two contributions:

1. **BioFiLM** — a FiLM-conditioned, Xception-style fully-convolutional encoder–decoder that
   adapts a shared base model to different biomes for dense (10 m) above-ground biomass (AGB)
   prediction.
2. **GPR / kriging calibration** — a post-hoc, per-tile Gaussian Process Regression scheme that
   corrects the spatially structured residuals of dense AGB maps, validated on an independent
   site in Sumatra and applied to the ESA CCI biomass product.

## Repository layout

```
src/
├── model/        BioFiLM model + training/eval        (§3.1, §4.2.1, Table 1)
├── inference/    dense AGB map generation             (§4.2.2)
├── kriging/      per-tile GPR calibration             (§3.2, §4.2.2, Table 2)
├── sumatra/      Sumatra use case + ESA CCI           (§4.3, Table 3, Fig 6)
└── data_prep/    Sentinel-2 compositing / cloud / orbit / region helpers (§4.2.2, §3.1.1)
configs/          per-environment data-path configs (<env>.yaml) + ensemble run-id lists
scripts/train/    training launchers (BioFiLM ablations, Table 1 / Appendix E.6–E.7)
config.py         loads configs/<AGBD_ENV>.yaml and exposes DATA_ROOT + all paths
DATA.md           manifest of the large inputs you must provide
```

## Paper → code map

| Paper item | Where |
|------------|-------|
| §3.1 BioFiLM architecture | `src/model/nico_net_film.py`, `src/model/models.py` |
| §3.1.2 / §4.2.1 training, Table 1 | `src/model/train.py`, `src/model/eval.py`, `scripts/train/` |
| §4.2.2 dense map generation | `src/inference/inference*.py`, `src/data_prep/` |
| §3.2 / §4.2.2 kriging, Table 2 | `src/kriging/kriging.py`, `predict.py`, `post_merge.py`, `threshold.py` |
| §4.3 Sumatra (Table 3) | `src/sumatra/kriging_ours_gedi.py` (our map), `kriging_gedi.py` (ESA CCI), `kriging_downsampled_gedi.py` |
| Figure 6 (Sumatra maps) | `src/sumatra/compose_figure.py`, `src/sumatra/get_results.ipynb` |

## Installation

The project uses [**uv**](https://docs.astral.sh/uv/) for environment management. It installs a
pinned, reproducible environment (from `uv.lock`) — including PyTorch and the full geospatial
stack — in one step, with no conda and no system GDAL:

```bash
uv sync --extra notebook        # core deps + Jupyter, into ./.venv
```

Then either prefix commands with `uv run` (e.g. `uv run python -m model.eval ...`,
`uv run jupyter lab`), or activate the venv (`source .venv/bin/activate`). `uv sync` also installs
this repo as an editable package, so `config`, `model`, `inference`, `kriging`, `sumatra`, … import
directly.

Optional extras: `--extra tracking` (Weights & Biases), `--extra cloudmask` (the IPL-UV cloudSEN12
model needed only by `data_prep/cloud_mask.py`), `--extra dev` (ipdb).

The env is Python 3.11 with PyTorch, PyTorch Lightning, GPyTorch and rasterio/geopandas/pyogrio.
On Linux the default `torch` wheel is CUDA-enabled; for a CPU-only machine install it from the CPU
index (`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## Quick start: single-tile pipeline

`examples/single_tile_pipeline.ipynb` runs the full pipeline for one Sentinel-2 tile —
composite -> inference -> kriging — by importing and calling the repository's functions
(`data_prep.composite.composite`, `inference.inference_composite.run_inference`,
`kriging.kriging.main`). Set `AGBD_ENV` (the first cell defaults it to `pf-pc28`), edit the
configuration cell, and run top to bottom.
Requires the full environment, a GPU, and the data described in `DATA.md`. See `examples/README.md`.

## Data

All large inputs (AGBD patches, Sentinel-2/ALOS rasters, GEDI footprints, checkpoints, dense
prediction rasters, ESA CCI and Sumatra reference products) live outside the repo. Their
locations are described per environment by a YAML file in **`configs/<env>.yaml`**, which sets
`data_root` and every data/output sub-path. You select the active environment with the
**`AGBD_ENV`** variable:

```bash
export AGBD_ENV=pf-pc28              # loads configs/pf-pc28.yaml (or `euler`, or your own)
```

`AGBD_ENV` is **required** — `import config` raises if it is unset, listing the available
configs. To add a machine, copy an existing file to `configs/<name>.yaml`, edit `data_root`
and the `paths:` block, and set `AGBD_ENV=<name>`. Two layouts ship by default: `pf-pc28`
(flat) and `euler` (a `Data/`-prefixed layout — confirm its `data_root` before use).

`config.py` reads the chosen YAML and exposes `DATA_ROOT`, a `PATHS` dict, and the
convenience `*_DIR` constants; Python modules do `from config import DATA_ROOT, PATHS, ...`.
You can still override the root at runtime without editing the YAML:

```bash
export DATA_ROOT=/mnt/agbd_data      # overrides data_root from the active config
```

Lay the directory out as described in **DATA.md**.

### Downloading the data

The large inputs are distributed as a bundle outside GitHub. Download them into `data/`
with the provided `download.sh` (it also fetches the AGBD HDF5 patches
`data_subset-2019-v4_*-*.h5`), then use the ready-made `local` config:

```bash
bash download.sh                 # downloads + extracts the bundle into ./data/
export AGBD_ENV=local            # loads configs/local.yaml
export DATA_ROOT="$(pwd)/data"   # or set `data_root` in configs/local.yaml
```

The bundle is organised as:

```
data/
├── AGB.tif  AGB-pre.tif  STD.tif  STD-pre.tif   # reference outputs (comparison only; not read by code)
├── example/                       # the single provided tile (30NXM) — composite + inference inputs
│   ├── S2*_T30NXM_*.zip           #   Sentinel-2 L2A products
│   ├── ALOS_30NXM_20.tif          #   ALOS PALSAR
│   ├── DEM_30NXM.tif              #   ALOS DSM
│   └── LC_30NXM_2019.tif          #   land cover
├── other/                         # shared model + normalisation inputs
│   ├── 17997535-1.pkl             #   model config (read by inference)
│   ├── 17997535-{1,2,3}_best.ckpt #   ensemble checkpoints (loaded flat, see note)
│   ├── statistics_subset_2019-2020-v4-1.pkl   # normalisation stats (inference)
│   ├── s2_tile_to_region-v3.pkl   #   tile -> FiLM region class
│   ├── embeddings_train.csv       #   cat2vec embeddings
│   ├── biomes_splits_to_name.pkl  #   train/val/test split (training/eval only)
│   └── sentinel_2_index_shapefile/  # Sentinel-2 MGRS index (.shp/.dbf/.prj/...)
├── kriging/                       # kriging inputs
│   ├── L4A_*-indexed.gpkg         #   GEDI L4A footprints (pass as --path_gedi)
│   └── valid_2020.txt             #   (also shipped in src/kriging/txt_files/)
└── Sumatra/                       # Sumatra use case (§4.3)
    ├── agbd_{100,500,1000}m.tif   #   field/ALS reference maps
    ├── L4A_Sumatra.gpkg           #   GEDI L4A (Sumatra AOI)
    ├── GEDI_L4B_AGBD_Sumatra.tif  #   GEDI L4B reference
    └── CCI_N00E100*.tif           #   ESA CCI biomass (CCI experiment)
```

`predictions/` and `kriging/` outputs are created by the pipeline; you don't download them.

**What runs with this bundle.** The full example pipeline (composite → inference → kriging)
for tile 30NXM runs out of the box. Note:

- **Weights are loaded flat from `other/`.** Inference loads checkpoints as
  `<ckpt>/<id>_best.ckpt` (no architecture subfolder), so the 3 `.ckpt` and the
  `17997535-1.pkl` config sit directly in `other/` — do not nest them in a subfolder.
- **Reference outputs** (`AGB.tif`, `AGB-pre.tif`, `STD.tif`, `STD-pre.tif`) ship at the
  bundle root for comparison only; no code reads them (`-pre` = pre-kriging, the others
  post-kriging).
- **Canopy height is not needed** — the provided checkpoints use `ch=False`.
- **`.h5` patches are only for training/eval from scratch.** Reproducing with the provided
  checkpoints does not need them. `train.py`/`eval.py` read them from a hardcoded
  `<DATA_ROOT>/patches`, so place them in `data/patches/` if you train.
- **Other tiles:** the `tiles/alos/dem/lc` paths in `configs/local.yaml` point at `example/`;
  to run on your own tiles, point them at your own S2/ALOS/DEM/LC data.
- **Sumatra analysis is self-contained.** `src/sumatra/get_results.ipynb` (Table 3 + Fig 6)
  reads a small bundled dataset in `src/sumatra/data/`, so it runs from a fresh clone with no
  external data and no `AGBD_ENV`. Only *re-running* the Sumatra kriging (`src/sumatra/kriging.sh`)
  needs the full external inputs (our S2/merged prediction, ESA CCI, GEDI L4A, ALOS DEM) under
  `DATA_ROOT`; see `src/sumatra/README.md`.
  - **Original data sources.** The files under `src/sumatra/data/` are redistributed copies
    provided for convenience. The originals come from: the **ESA CCI** biomass product
    ([CEDA catalogue](https://catalogue.ceda.ac.uk/uuid/6429d1aafe1e43b9b414e4a5a7f8b903/)), and
    the **Sumatra** reference dataset described in
    [May (2024), *Remote Sensing of Environment*](https://www.sciencedirect.com/science/article/pii/S0034425724004103)
    — the reference AGB prediction rasters themselves were obtained by contacting the author,
    Paul B. May.

> Note: because scripts import the top-level `config` module (and the `model`/`inference`/
> `kriging` packages), run them after `pip install -e .`, or with `PYTHONPATH=src`.

## Logging (Weights & Biases is optional)

W&B is **off by default** — `train.py`, the kriging pipeline (`kriging.py` → `predict.py`),
and the Sumatra scripts all run without it. To enable experiment tracking:

```bash
export USE_WANDB=true
export WANDB_ENTITY=<your-entity>
# optional: export WANDB_OFFLINE=true   # log to a local ./wandb/ dir
```

When disabled, a no-op run object is used (see `src/wandb_utils.py`), so logging calls are
ignored. `predict.py` reads the fit-time configuration from the kriging checkpoint, not W&B.

`eval.py` and the `inference/*.py` scripts read each model's training config from a JSON
sidecar that `train.py` saves next to the checkpoint (`{model}_config.json`), falling back to
W&B only for checkpoints trained before sidecars existed (see `load_train_config`).

> A few **analysis/tooling** scripts still query the W&B server to locate trained runs:
> `model/eval.py` and the full-scale `inference/inference*.py` when run with `--test_set`.
> These import `wandb` lazily (only at the lookup), so the modules load fine without it; you
> need a W&B account only to use those specific lookups.

## Typical workflow

1. **Train BioFiLM** (or use the provided checkpoints): `scripts/train/*.sh` → `src/model/train.py`.
2. **Generate dense AGB maps**: `src/data_prep/` (composite S2 tiles) → `src/inference/inference_composite.py`.
3. **Calibrate with kriging**: `src/kriging/kriging.py` → `predict.py` → `post_merge.py`.
4. **Sumatra use case**: `src/sumatra/`.

## Reproducing the paper

Set `AGBD_ENV` (see [Data](#data)) and provide the inputs in `DATA.md`, then:

| Paper artifact | How to produce |
|----------------|----------------|
| **Table 1** — BioFiLM vs. baseline RMSE | train: `scripts/train/*.sh` → `src/model/train.py`; evaluate: `python -m model.eval ...` |
| **Figs 3 / F.9–F.10** — dense AGB maps | `src/data_prep/` (composite) → `python -m inference.inference_composite ...` |
| **Table 2** — kriging configurations | `bash src/kriging/kriging.sh` (→ `kriging.py` → `predict.py`) |
| **Table 3 + Figure 6** — Sumatra (ours & ESA CCI) | open `src/sumatra/get_results.ipynb` (self-contained, reads `src/sumatra/data/`); to regenerate the corrected maps first, `bash src/sumatra/kriging.sh` → `kriging_*gedi.py`; figure panels: `compose_figure.py` |
| **End-to-end, one tile** | `examples/single_tile_pipeline.ipynb` |

`scripts/train/` covers the BioFiLM ablations (Table 1 and Appendix E.6/E.7); the kriging
hyper-parameters of Table 2's recommended configuration (config VII) are the defaults in
`src/kriging/kriging.sh`. For an exact software environment, generate a lockfile from your
install (`conda env export --no-builds > environment.lock.yml`).
