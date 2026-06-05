"""
Central configuration for data and output paths.

All scripts in this repository read large inputs (the AGBD patches, Sentinel-2 / ALOS
rasters, GEDI footprints, model checkpoints, dense prediction rasters, the ESA CCI and
Sumatra reference products) from a single root directory, `DATA_ROOT`.

Set it either by:
  * exporting the environment variable, e.g. `export DATA_ROOT=/mnt/agbd_data`, or
  * editing the default below.

Every path in the codebase is built from `DATA_ROOT` (the scripts do
`from config import DATA_ROOT` and then `f"{DATA_ROOT}/..."`). See DATA.md for the exact
files expected under each sub-path. The convenience constants below are provided for the
most common locations; you can use them or build paths from `DATA_ROOT` directly.
"""

import os

# Root under which all data/checkpoints/predictions live. Override with the DATA_ROOT env var.
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/data")

# Common sub-paths (see DATA.md).
PATCHES_DIR     = os.path.join(DATA_ROOT, "Data", "patches")                              # AGBD HDF5 patches + norm stats
S2_DIR          = os.path.join(DATA_ROOT, "S2_L2A")                                       # Sentinel-2 L2A products
ALOS_DIR        = os.path.join(DATA_ROOT, "ALOS")                                         # ALOS PALSAR / DSM
LC_DIR          = os.path.join(DATA_ROOT, "LC")                                           # Copernicus land cover (biome)
CAT2VEC_DIR     = os.path.join(DATA_ROOT, "EcosystemAnalysis", "Models", "Baseline", "cat2vec")
WEIGHTS_DIR     = os.path.join(DATA_ROOT, "EcosystemAnalysis", "Models", "Biomes", "weights")        # BioFiLM checkpoints
PREDICTIONS_DIR = os.path.join(DATA_ROOT, "EcosystemAnalysis", "Models", "Biomes", "predictions")    # dense AGB rasters
KRIGING_DIR     = os.path.join(DATA_ROOT, "EcosystemAnalysis", "Models", "Biomes", "kriging")        # kriging intermediates/outputs
GEDI_DIR        = os.path.join(DATA_ROOT, "GEDI")                                         # GEDI L4A footprints
SUMATRA_DIR     = os.path.join(DATA_ROOT, "Sumatra-AGB")                                  # Sumatra reference + rasters
S2_INDEX_SHP    = os.path.join(DATA_ROOT, "BiomassDatasetCreation", "Data", "download_Sentinel", "sentinel_2_index_shapefile.shp")
