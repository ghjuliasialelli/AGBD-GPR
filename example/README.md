# Examples

## `single_tile_pipeline.ipynb` — full pipeline for one Sentinel-2 tile

A notebook that runs the whole pipeline for a single tile by importing and calling the
repository's actual functions:

1. **Composite** — `data_prep/composite.py:composite()`
2. **Inference** — `inference/inference_composite.py:run_inference()`
3. **Kriging** — `kriging/kriging.py:main()`

Each step builds the same argument list the CLI scripts use and calls the function directly
(the script entry points were refactored to accept an optional `argv`, so they are usable both
from the shell and programmatically).

**Requirements**
- The project environment (`uv sync` — PyTorch, GPyTorch, rasterio, geopandas, …; see the top-level README) and a **GPU** for the inference and kriging steps.
- `DATA_ROOT` set (in `src/config.py` or the environment) to data laid out as in `DATA.md`:
  the S2 L2A products for the tile, the ALOS DEM, GEDI L4A footprints, the S2 tile-index
  shapefile, and the BioFiLM ensemble checkpoints.
- Weights & Biases is **not** required.

Open the notebook, edit the configuration cell (tile id, year, ensemble ids, paths), and run
top to bottom.

> The GP model used for calibration lives in `src/kriging/gp.py` (kept dependency-light:
> torch + gpytorch only), and is re-exported by `src/kriging/core.py`.
