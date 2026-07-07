# `src/sumatra` — GEDI-based kriging correction of AGB maps

This folder post-processes an above-ground biomass (AGB) map over Sumatra by **kriging**:
fitting a Gaussian Process (GP) to the *residuals* between the map and GEDI reference
footprints, then subtracting the predicted residual field from the map. The result is a
"corrected" AGB raster plus before/after error metrics.

If you are new to this folder, read this file first, then open `kriging_gedi.py` (the most
self-contained of the three pipeline scripts).

---

## The big picture

```
 dense AGB prediction (GeoTIFF)  ─┐
 GEDI L4A footprints (.gpkg)      ├─►  kriging_*.py  ─►  corrected AGB GeoTIFF + metrics .pkl
 (optional) STD / extra features ─┘        │
                                           ├─ 1. sample the map at each GEDI footprint
                                           ├─ 2. residual = map_AGB − GEDI_AGB
                                           ├─ 3. geographically-separated train/val/test split
                                           ├─ 4. fit an Exact GP to the residuals
                                           ├─ 5. predict residuals over the whole tile
                                           └─ 6. corrected = map − predicted_residual
```

The GP inputs are the footprint **(x, y) pixel coordinates**, and optionally: the map's own
predicted value, an auxiliary band (the ensemble STD), and "extra features" derived from the
map (local mean / std / coefficient-of-variation / range, plus Sobel and Laplacian edges).

---

## Files

| File | What it is |
|------|------------|
| **`pipeline.py`** | **The kriging pipeline itself**, in one place: the geographic split (`geographical_train_test_split`, `get_fold`), split assembly (`get_train_val_test_split`), and the top-level `run_kriging_for_map(cfg, path_predictions, model_name, s2_tile, reference)` that every entry script calls. Read this to understand what actually happens. |
| **`paths.py`** | **Single source of truth for every I/O path.** The `kriging_*.py` scripts write through it and `get_results.ipynb` reads through it, so output and input filenames can never drift apart. Reads prefer the bundled `data/` (below), falling back to the full `DATA_ROOT` tree; writes go to `DATA_ROOT` (or `SUMATRA_OUT`). |
| **`data/`** | **Self-contained bundle** (~320 MB): exactly the inputs + kriging outputs `get_results.ipynb` needs, so a fresh clone runs the notebook with no external data and no `AGBD_ENV`. See `data/README.md`. |
| **`kriging_gedi.py`** | Thin entry point for a **single-raster** prediction (e.g. the ESA **CCI** map): parse args → seed → `run_kriging_for_map`. Start here. |
| **`kriging_downsampled_gedi.py`** | Thin entry point, input = **our downsampled merged** prediction (a 2-band GeoTIFF: AGB + STD). |
| **`kriging_ours_gedi.py`** | Thin entry point, input = **our full-resolution** prediction; loops over the Sumatra tiles and additionally supports **OOD footprint removal** (`--ood`). |
| **`common.py`** | Lower-level helpers: `get_data` (returns the `SplitBundle` — see below), extra-feature computation, prediction loading, the CLI `parser` (returns an argparse `Namespace`), and the GP training loop with LR-retries. |
| **`kriging.sh`** | The launcher. Sets every hyper-parameter, then dispatches to one of the three entry scripts. **This is how the pipeline is meant to be run.** |
| **`compose_figure.py`** | Standalone figure builder: arranges pre-rendered map / residual / density panels into the paper's comparison figures. Not part of the kriging run. |
| **`get_results.ipynb`** | Post-hoc analysis of kriging outputs: agreement with GEDI L4B, binned residuals, density scatter plots, per-map metrics. Reads the `.tif` / `.pkl` written by the scripts above (all resolved through `paths.py`). |
| **`Resources/`** | Static inputs: `geometry.geojson`, `sumatra_tiles.txt`, `sumatra_products.txt`. |

The three `kriging_*.py` scripts are now **thin entry points**: they parse the CLI, set seeds,
and call `run_kriging_for_map` in `pipeline.py`. They differ only in how the input prediction is
loaded (and, for `ours`, the tile loop + OOD branch).

Shared code lives one level up: `src/kriging/core.py` (the `ExactGPModel`, geometry helpers,
tile/feature assembly) and `src/config.py` (`DATA_ROOT` and all data paths).

---

## How to run

Everything is driven by **`kriging.sh`** — edit the settings block at the top, then:

```bash
conda activate krige
export AGBD_ENV=pf-pc28            # picks configs/<env>.yaml -> sets DATA_ROOT
export PYTHONPATH=/path/to/AGBD-GPR/src
cd src/sumatra
bash kriging.sh
```

Outputs (checkpoints, splits, corrected rasters, metrics) go to fixed locations under
`DATA_ROOT/EcosystemAnalysis/Models/Biomes/Sumatra`, resolved by **`paths.py`**. To send them
elsewhere (e.g. a scratch dir for a dry run) without touching the inputs, set `SUMATRA_OUT`:

```bash
export SUMATRA_OUT=/tmp/my_kriging_run   # optional; inputs still read from DATA_ROOT
```

The naming scheme is defined at the top of `paths.py`; because both the scripts and the
notebook go through that module, they always agree on filenames.

### Then analyse the results

Open **`get_results.ipynb`** and run it top to bottom. It reproduces the paper's figures and
metric tables — agreement with GEDI L4B, pre/post maps, binned residuals, density scatters, and
the footprint-level comparison of our downsampled map vs. ESA CCI.

Because the data it needs is bundled under **`data/`** (and resolved via `paths.py`), the notebook
is **self-contained**: a fresh clone runs it end to end with no external data, no `AGBD_ENV`, from
any working directory. Running `kriging.sh` is only needed to *regenerate* those outputs.

### The main knobs in `kriging.sh`

| Setting | Meaning |
|---------|---------|
| `pred` | Which map to correct: `ours`, `CCI`, or `gdbt`. Selects the script + input path. |
| `downsampled` | When `pred=ours`: use the single downsampled `.tif` (`kriging_downsampled_gedi.py`) vs. the full-res path (`kriging_ours_gedi.py`). |
| `aux` | Auxiliary GP input: `STD` (ensemble std) or `none`. |
| `extra_features` | Map-derived features to add, e.g. `mean_25 std_25 cv_25 lr_25 sobel laplace`. |
| `coords` / `norm_coords` | Use / min-max-normalize the (x, y) coordinates. |
| `norm_res` | Z-score the residuals before fitting (rescales output-scale/noise priors). |
| `x_lengthscale` / `y_lengthscale` | Spatial lengthscales, or `dynamic` to derive them from footprint spacing. |
| `stripe_size` | Size (px) of the stripes used for the geographic train/val/test split. |
| `test_holdout` / `val_holdout` | Split fractions. |
| `max_train_footprints` | Cap on training footprints (subsampled if exceeded). |
| `num_iterations` | GP training iterations (with early stopping). |
| `SAVE` / `SAVE_preds` | Save the metrics `.pkl` / the corrected `.tif`. |

---

## The "split bundle"

Everything a fitted split needs — per-split residuals, coordinates, AGB values, predictions,
auxiliary data, extra features, and normalization stats — is carried by the **`SplitBundle`
dataclass** (`common.py`). `get_data()` builds it; `run_kriging_for_map()` unpacks its fields
once into locals. Each field is documented on the dataclass definition. (It used to be a bare
30-element tuple threaded through the code, which is why older commits look scary.)

---

## Cleanup notes / history

- **Deduplication (done):** the three `kriging_*.py` scripts used to be ~99%-identical copies of
  the whole pipeline. That shared body now lives once in `pipeline.py` (`run_kriging_for_map`),
  and the scripts are thin entry points. Verified to produce byte-identical outputs to the
  pre-refactor version on the downsampled run.
- **Split bundle → dataclass (done):** the 30-element tuple is now the `SplitBundle` dataclass
  in `common.py`.
- **Paths module (done):** all I/O now goes through `paths.py` (see "How to run"), replacing the
  old scattered `getcwd()`-relative paths and ad-hoc filename strings. This is what keeps the
  scripts and `get_results.ipynb` consistent.
- **Removed dead code:** two leftover lines in the CCI/downsampled scripts
  (`subsample = subsample and ...` / `ood = ood and ...`) were dead *and* raised `NameError`
  (`subsample` was never defined). Removed. OOD removal only exists in `kriging_ours_gedi.py`.
- `kriging_ours_gedi.py` still hardcodes the full-resolution input directory and year (2021) for
  the per-tile loop; the ensemble comes from `--ens_models`.
