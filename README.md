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
├── figures/      metrics + plotting for the paper     (Table 2, Figs 4 & 5)
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
| Figure 4 (binned residuals) | `src/figures/plots/binned-histogram.py`, `binned-rmse.py` |
| Figure 5 (spider plots) | `src/figures/plots/biome-spiderplot.py`, `region-spiderplot.py` |
| Table 2 metrics (RMSE/MAE/ΔB) | `src/figures/metrics/compute_*_RMSE.py`, `compute_*_binned_histogram.py` |
| §4.3 Sumatra (Table 3) | `src/sumatra/kriging_ours_gedi.py` (our map), `kriging_gedi.py` (ESA CCI), `kriging_downsampled_gedi.py` |
| Figure 6 (Sumatra maps) | `src/sumatra/compose_figure.py`, `src/sumatra/get_results.ipynb` |

## Installation

```bash
conda env create -f environment.yml
conda activate biofilm-gpr-agb
pip install -e .          # makes the `model`, `inference`, `kriging`, ... packages importable
```

Python 3.10. The model/inference/kriging code uses PyTorch, PyTorch Lightning and GPyTorch;
`cloud_mask.py` additionally needs the external cloudSEN12 package (see DATA.md).

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

> Note: because scripts import the top-level `config` module (and the `model`/`inference`/
> `kriging` packages), run them after `pip install -e .`, or with `PYTHONPATH=src` (needed for
> the hyphen-named figure scripts, which run as files rather than importable modules).

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
> `model/eval.py`, the full-scale `inference/inference*.py`, and the `figures/metrics/*`
> scripts when run with `--test_set`. These import `wandb` lazily (only at the lookup), so the
> modules load fine without it; you need a W&B account only to use those specific lookups.

## Typical workflow

1. **Train BioFiLM** (or use the provided checkpoints): `scripts/train/*.sh` → `src/model/train.py`.
2. **Generate dense AGB maps**: `src/data_prep/` (composite S2 tiles) → `src/inference/inference_composite.py`.
3. **Calibrate with kriging**: `src/kriging/kriging.py` → `predict.py` → `post_merge.py`.
4. **Reproduce tables/figures**: `src/figures/metrics/` and `src/figures/plots/`.
5. **Sumatra use case**: `src/sumatra/`.

## Reproducing the paper

Set `AGBD_ENV` (see [Data](#data)) and provide the inputs in `DATA.md`, then:

| Paper artifact | How to produce |
|----------------|----------------|
| **Table 1** — BioFiLM vs. baseline RMSE | train: `scripts/train/*.sh` → `src/model/train.py`; evaluate: `python -m model.eval ...` |
| **Figs 3 / F.9–F.10** — dense AGB maps | `src/data_prep/` (composite) → `python -m inference.inference_composite ...` |
| **Table 2** — kriging configurations | `bash src/kriging/kriging.sh` (→ `kriging.py` → `predict.py`); metrics: `src/figures/metrics/compute_{pre,post}_RMSE.py`, `compute_{pre,post}_binned_histogram.py` |
| **Figure 4** — pre/post residuals per bin | `src/figures/plots/binned-histogram.py`, `binned-rmse.py` |
| **Figure 5** — spider plots (biome / region) | `src/figures/plots/biome-spiderplot.py`, `region-spiderplot.py` |
| **Table 3 + Figure 6** — Sumatra (ours & ESA CCI) | `bash src/sumatra/kriging.sh` → `kriging_*gedi.py`; figure: `compose_figure.py`, `get_results.ipynb` |
| **End-to-end, one tile** | `examples/single_tile_pipeline.ipynb` |

`scripts/train/` covers the BioFiLM ablations (Table 1 and Appendix E.6/E.7); the kriging
hyper-parameters of Table 2's recommended configuration (config VII) are the defaults in
`src/kriging/kriging.sh`. For an exact software environment, generate a lockfile from your
install (`conda env export --no-builds > environment.lock.yml`).
