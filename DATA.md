# Data manifest

None of the large binary inputs are committed to this repository (see `.gitignore`).
Every script reads them from a single root, referred to in the code by the placeholder
path `/path/to/data` and documented in `config.py` as `DATA_ROOT`. Point `DATA_ROOT`
at a directory laid out as below, or edit the paths in `config.py`.

> All paths derive from a single `DATA_ROOT` (set it as an environment variable or edit
> `src/config.py`). Python code uses `config.DATA_ROOT`; shell scripts use `${DATA_ROOT}`.

## Required inputs

### Model training / inference (`src/model`, `src/inference`)
| What | Expected location (under DATA_ROOT) | Notes |
|------|-------------------------------------|-------|
| AGBD dataset patches (HDF5) + normalisation stats | `Data/patches/` | The AGBD dataset (Sialelli et al., 2025). |
| Biome `cat2vec` embeddings | `EcosystemAnalysis/Models/Baseline/cat2vec/AGBD/` | Precomputed (Appendix B.2). |
| Region / biome split + tile→region maps | `BiomassDatasetCreation/Data/download_Sentinel/biomes_split`, `.../s2_tile_to_region-*` | Used for FiLM conditioning. |
| BioFiLM checkpoints (ensemble of 3) | `EcosystemAnalysis/Models/Biomes/weights/nico_film/` | Run IDs `17997535-1/-2/-3` in the paper. |

### Dense map generation / data-prep (`src/data_prep`, `src/inference`)
| What | Expected location | Notes |
|------|-------------------|-------|
| Sentinel-2 L2A products | `S2_L2A/` (or `Data/S2_L2A/`) | Inputs to compositing / inference. |
| ALOS PALSAR-2 / DSM | `ALOS/` (or `Data/ALOS/`) | Input modality. |
| Copernicus land cover | `LC/` ; ESA WorldCover under `WorldCover/`, `data/ESA_WorldCover/` | Biome features. |
| Sentinel-2 MGRS tile index shapefile | `BiomassDatasetCreation/Data/download_Sentinel/sentinel_2_index_shapefile.shp` | Tile geometries. |

### Kriging / calibration (`src/kriging`)
| What | Expected location | Notes |
|------|-------------------|-------|
| Dense AGB prediction rasters (kriging input) | `EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/<ensemble>/` | Per-tile AGB + ensemble STD. |
| GEDI L4A footprints | `GEDI/` (per tile / AOI) | Filtered as in §4.2.2. |
| Tile lists | shipped in `src/kriging/txt_files/` | e.g. `valid_2020.txt`, `all_2020.txt`. |

### Sumatra use case (`src/sumatra`) — §4.3, Table 3, Fig 6
| What | Expected location | Notes |
|------|-------------------|-------|
| Sumatra reference AGB maps | `Sumatra-AGB/pred_rasters/agbd_{100,500,1000}m.tif` | Field/ALS reference (May et al., 2024). |
| GEDI L4A (Sumatra AOI) | `GEDI/Sumatra/L4A_Sumatra.gpkg` | For calibration. |
| ESA CCI biomass product (N00E100 v6, 2021–2022) | `EcosystemAnalysis/Models/Biomes/kriging/predictions/CCI/` | For the CCI calibration experiment. |
| GEDI L4B reference | `EcosystemAnalysis/.../Sumatra/Data/GEDI_L4B_AGBD_Sumatra.tif` | Used in `get_results.ipynb`. |

## External model dependency (not bundled)
- `src/data_prep/cloud_mask.py` requires the **cloudSEN12** package and weights
  (`cloudsen12_models`, ~61 MB). Install from https://github.com/IPL-UV/cloudsen12_models
  (kept out of this repo to avoid shipping third-party weights).

## Not required for this paper
`Data/patches/AEF`, `Data/patches/TESSERA`, and `CH/` paths appear in some scripts but
belong to other projects (AEF / TESSERA / canopy-height) and are not needed to reproduce
this paper.
