"""
Single source of truth for every file path used by the Sumatra kriging pipeline.

The `kriging_*.py` scripts WRITE their outputs through these helpers, and `get_results.ipynb`
READS them back through the same helpers. Because both sides call the same functions, the
output filenames and the notebook's input filenames can never drift apart.

Two data locations, resolved automatically:
  * `data/` next to this file — a small (~320 MB) self-contained BUNDLE with exactly the inputs
    and outputs `get_results.ipynb` needs, so a fresh clone can run the notebook with no
    external data and no `AGBD_ENV`. Reads prefer this bundle when the file is present.
  * the full data tree under `config.DATA_ROOT` (processing machine, `AGBD_ENV` set) — used for
    all WRITES (a real kriging run) and as the read fallback when a file isn't in the bundle.
Set `SUMATRA_OUT` to redirect writes elsewhere (dry runs).

Bundle layout (flat, one folder to grab):
  data/inputs/   merged_downsampled-100m_composite.tif, CCI_N00E100.tif, GEDI_L4B_AGBD_Sumatra.tif,
                 L4A_Sumatra.gpkg, agbd_{100,500,1000}m.tif
  data/outputs/  kriging-<model>-<reference>_<stripe>.tif, splits-<...>.pkl

Full-tree naming scheme (unchanged):
  predictions/<arch>/<year>/<ens>/kriging-<model>-<reference>_<stripe>[_ood].tif   corrected AGB raster
  predictions/<arch>/<year>/<ens>/results-<model>-<reference>_<stripe>[_ood].pkl   footprint pre/post values
  predictions/splits/splits-<model>_<test>_<val>_<foot>_<diff>_<stripe>[_ood]-<reference>.pkl  split indices
  predictions/splits/split_map-<model>.pkl                                          split visualization array
  checkpoints/<model>/<model>.pt | <model>.ckpt                                     GP checkpoint
  figs/checkerboard_<stripe>_<model>.png                                            split figure
"""

import os
from os.path import join, dirname, abspath, isfile

# config.DATA_ROOT is OPTIONAL: on the processing machine (AGBD_ENV set) it locates the full data
# tree; in a fresh clone it may be absent, in which case everything falls back to the bundle.
try:
    from config import DATA_ROOT
except Exception:
    DATA_ROOT = None

# --- Repo-bundled data (self-contained; enough to run get_results.ipynb) ----------------
BUNDLE     = join(dirname(abspath(__file__)), "data")
BUNDLE_IN  = join(BUNDLE, "inputs")     # input maps + reference rasters + GEDI footprints
BUNDLE_OUT = join(BUNDLE, "outputs")    # corrected rasters + split indices

# --- Full data tree (processing machine); target of all WRITES + read fallback ----------
SUMATRA_DIR = os.environ.get("SUMATRA_OUT") or (
    join(DATA_ROOT, "EcosystemAnalysis", "Models", "Biomes", "Sumatra") if DATA_ROOT else BUNDLE)
PREDICTIONS = join(SUMATRA_DIR, "predictions")        # kriging outputs
SPLITS_DIR  = join(PREDICTIONS, "splits")             # train/val/test split indices + split maps
CHECKPOINTS = join(SUMATRA_DIR, "checkpoints")        # GP checkpoints
FIGS_DIR    = join(SUMATRA_DIR, "figs")               # figures (split maps, notebook figures)
DATA_DIR    = join(SUMATRA_DIR, "Data")               # input maps to correct (merged / CCI)
REF_RASTERS = join(DATA_ROOT, "Sumatra-AGB", "pred_rasters") if DATA_ROOT else None  # reference AGB


def _ensure(directory):
    """Create `directory` (and parents) if needed, and return it."""
    os.makedirs(directory, exist_ok=True)
    return directory


def _ood(ood):
    return "_ood" if ood else ""


def _read(bundle_path, legacy_path):
    """Resolve a file to READ: prefer the bundled copy; fall back to the full-tree copy."""
    if isfile(bundle_path):
        return bundle_path
    return legacy_path if legacy_path is not None else bundle_path


# --- Input maps + reference (read) ------------------------------------------------------
def merged_input(ens_models):
    """Our downsampled 100 m merged prediction (the raster we krige for the 'ours' result)."""
    return _read(join(BUNDLE_IN, "merged_downsampled-100m_composite.tif"),
                 join(DATA_DIR, "merged", "_".join(ens_models), "merged_downsampled-100m_composite.tif"))


def cci_input():
    """The ESA CCI AGB map over Sumatra (the raster we krige for the 'CCI' result)."""
    return _read(join(BUNDLE_IN, "CCI_N00E100.tif"),
                 join(DATA_DIR, "CCI", "CCI_N00E100.tif"))


def gedi_l4b():
    """GEDI L4B 1 km gridded AGB raster (used for the 1 km agreement check)."""
    return _read(join(BUNDLE_IN, "GEDI_L4B_AGBD_Sumatra.tif"),
                 join(DATA_DIR, "GEDI_L4B_AGBD_Sumatra.tif"))


def gedi_footprints():
    """GEDI L4A footprints (.gpkg): the reference used for kriging + footprint-level metrics."""
    return _read(join(BUNDLE_IN, "L4A_Sumatra.gpkg"),
                 join(DATA_ROOT, "GEDI", "Sumatra", "L4A_Sumatra.gpkg") if DATA_ROOT else None)


def reference_agb(resolution=100):
    """GEDI-derived Sumatra reference AGB raster at the given resolution (metres)."""
    return _read(join(BUNDLE_IN, f"agbd_{resolution}m.tif"),
                 join(REF_RASTERS, f"agbd_{resolution}m.tif") if REF_RASTERS else None)


# --- Per-run output directory (write) --------------------------------------------------
def run_dir(arch, year, ens_models, make=False):
    """Directory holding all outputs for one (arch, year, ensemble) run (full tree)."""
    d = join(PREDICTIONS, arch, str(year), "_".join(ens_models))
    return _ensure(d) if make else d


# --- Corrected raster + footprint results ----------------------------------------------
def corrected_tif(arch, year, ens_models, model_name, reference, stripe_size, ood=False, make=False):
    """The corrected AGB GeoTIFF written after kriging (bands: AGB, residuals, [STD])."""
    fname = f"kriging-{model_name}-{reference}_{stripe_size}{_ood(ood)}.tif"
    if make:
        return join(run_dir(arch, year, ens_models, True), fname)
    return _read(join(BUNDLE_OUT, fname), join(run_dir(arch, year, ens_models), fname))


def results_pkl(arch, year, ens_models, model_name, reference, stripe_size, ood=False, make=False):
    """Per-footprint {pre, ref, idx, post} values, for computing metrics in the notebook."""
    fname = f"results-{model_name}-{reference}_{stripe_size}{_ood(ood)}.pkl"
    if make:
        return join(run_dir(arch, year, ens_models, True), fname)
    return _read(join(BUNDLE_OUT, fname), join(run_dir(arch, year, ens_models), fname))


# --- Train/val/test split indices ------------------------------------------------------
def split_pkl(model_name, test_holdout, val_holdout, max_train_footprints,
              max_split_diff, stripe_size, reference, ood=False, make=False):
    """The {train, val, test} footprint-index split used for a run (reused if it exists)."""
    fname = (f"splits-{model_name}_{test_holdout:.1f}_{val_holdout:.1f}_"
             f"{max_train_footprints}_{max_split_diff:.1f}_{stripe_size}{_ood(ood)}-{reference}.pkl")
    if make:
        return join(_ensure(SPLITS_DIR), fname)
    return _read(join(BUNDLE_OUT, fname), join(SPLITS_DIR, fname))


def split_map_pkl(model_name, make=False):
    """2D array visualizing the geographic train/val/test split (read by the notebook)."""
    fname = f"split_map-{model_name}.pkl"
    if make:
        return join(_ensure(SPLITS_DIR), fname)
    return _read(join(BUNDLE_OUT, fname), join(SPLITS_DIR, fname))


# --- Checkpoints (write; not read by the notebook) -------------------------------------
def checkpoint_dir(model_name, make=False):
    d = join(CHECKPOINTS, model_name)
    return _ensure(d) if make else d


def checkpoint_data(model_name, make=False):
    """Training data + likelihood state saved alongside the model (`.pt`)."""
    return join(checkpoint_dir(model_name, make), f"{model_name}.pt")


def checkpoint_state(model_name, make=False):
    """Best GP model state dict (`.ckpt`)."""
    return join(checkpoint_dir(model_name, make), f"{model_name}.ckpt")


# --- Figures ---------------------------------------------------------------------------
def checkerboard_png(stripe_size, model_name, make=False):
    """Figure showing the geographic split (checkerboard/stripe pattern)."""
    if make:
        _ensure(FIGS_DIR)
    return join(FIGS_DIR, f"checkerboard_{stripe_size}_{model_name}.png")
