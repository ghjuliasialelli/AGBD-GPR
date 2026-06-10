"""
Central configuration for data and output paths.

All scripts in this repository read large inputs (the AGBD patches, Sentinel-2 / ALOS
rasters, GEDI footprints, model checkpoints, dense prediction rasters, the ESA CCI and
Sumatra reference products) from a single root directory, `DATA_ROOT`, laid out per
environment. The per-environment layout lives in `configs/<env>.yaml` at the repo root.

Environment selection: set the `AGBD_ENV` env var to the name of a config file in
`configs/`, e.g. `export AGBD_ENV=euler` -> loads `configs/euler.yaml`. Add your own
`configs/<name>.yaml` and point `AGBD_ENV` at it.

`DATA_ROOT` defaults to the `data_root` field of the chosen YAML, but can still be
overridden at runtime with `export DATA_ROOT=/somewhere/else`.

Usage:
  from config import DATA_ROOT, PATHS, S2_DIR
  # PATHS is the dict consumed by the inference scripts (keys: norm, tiles, ckpt, ...).
  # The *_DIR / *_SHP constants below are convenience aliases into PATHS.

See DATA.md for the exact files expected under each sub-path.
"""

import os

import yaml

# --- Environment selection -------------------------------------------------------------

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def _available_envs():
    if not os.path.isdir(_CONFIG_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(_CONFIG_DIR) if f.endswith(".yaml"))


def _load_env_config():
    """Load and validate the `configs/<AGBD_ENV>.yaml` config."""
    env = os.environ.get("AGBD_ENV")
    if not env:
        raise RuntimeError(
            "AGBD_ENV is not set. Set it to the name of a config in configs/, "
            f"e.g. `export AGBD_ENV=euler`. Available: {_available_envs()}."
        )
    cfg_path = os.path.join(_CONFIG_DIR, f"{env}.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"No environment config for AGBD_ENV='{env}' at {cfg_path}. "
            f"Available: {_available_envs()}. Set AGBD_ENV to one of these or add the file."
        )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    if "data_root" not in cfg or "paths" not in cfg:
        raise KeyError(f"{cfg_path} must define both 'data_root' and 'paths'.")
    return env, cfg


ENV, _cfg = _load_env_config()

# Root under which all data/checkpoints/predictions live.
# YAML `data_root` is the default; the DATA_ROOT env var still wins if set.
DATA_ROOT = os.environ.get("DATA_ROOT", _cfg["data_root"])

# Resolved path dictionary (templated on DATA_ROOT). This is the single source of truth;
# the inference/kriging scripts consume PATHS directly instead of rebuilding paths.
PATHS = {key: value.format(data_root=DATA_ROOT) for key, value in _cfg["paths"].items()}

# Per-environment run defaults (arch, ensemble members, cpus_per_task, ...). Optional.
DEFAULTS = _cfg.get("defaults", {})

# --- Convenience constants (aliases into PATHS, see DATA.md) ---------------------------
PATCHES_DIR     = PATHS["norm"]          # AGBD HDF5 patches + norm stats
S2_DIR          = PATHS["tiles"]         # Sentinel-2 L2A products / composites
ALOS_DIR        = PATHS["alos"]          # ALOS PALSAR / DSM
LC_DIR          = PATHS["lc"]            # Copernicus land cover (biome)
CAT2VEC_DIR     = PATHS["embeddings"]    # cat2vec embeddings
WEIGHTS_DIR     = PATHS["ckpt"]          # BioFiLM checkpoints
PREDICTIONS_DIR = PATHS["predictions"]   # dense AGB rasters
KRIGING_DIR     = PATHS["kriging"]       # kriging intermediates/outputs
GEDI_DIR        = PATHS["gedi"]          # GEDI L4A footprints
SUMATRA_DIR     = PATHS["sumatra"]       # Sumatra reference + rasters
S2_INDEX_SHP    = PATHS["s2_index_shp"]  # Sentinel-2 tile index shapefile
