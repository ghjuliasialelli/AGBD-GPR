# `sumatra/data/` — data bundle for `get_results.ipynb`

Everything `get_results.ipynb` needs to run, so the analysis is reproducible from a fresh clone
with no other external data and no `AGBD_ENV`. The files are too large for GitHub: they are in the
`Sumatra/` folder of the ETH Research Collection record (flat, no subfolders). Download that folder, then

```bash
python src/sumatra/download_data.py --from <downloaded Sumatra folder or zip>  # sorts into inputs/ outputs/, checks MANIFEST.json
python src/sumatra/download_data.py --check                                    # only check the files already here
```

`MANIFEST.json` (in git) lists every file with its size and md5. All paths resolve through
`sumatra/paths.py`, which prefers these files and falls back to the full `DATA_ROOT` tree when a
file isn't here. After refreshing a file, rebuild the archive and the manifest with
`python src/sumatra/download_data.py --stage <dir>` (flat copy of every file) and upload the changed files to `Sumatra/`.

```
inputs/    the maps + reference the notebook opens directly
  merged_downsampled-{100,500,1000}m_composite.tif   our BioFiLM map (pre-kriging), 10 m tiles merged by
                                                     merge_tiles.py and averaged to each reference grid (AGB, STD)
  CCI_N00E100.tif                         ESA CCI AGB map over Sumatra (pre-kriging), 2021 and 2022 combined:
                                          band 1 median AGB, band 2 mean AGB_SD (the kriging's STD input),
                                          band 3 std of the two AGB maps
  GEDI_L4B_AGBD_Sumatra.tif               GEDI L4B 1 km gridded AGB (1 km agreement check)
  L4A_Sumatra.gpkg                        GEDI L4A footprints (kriging reference + metrics)
  agbd_100m.tif / agbd_500m.tif / agbd_1000m.tif   Sumatra reference AGB rasters (May et al. 2024)

outputs/   the kriging results the notebook reads back
  merged_downsampled-{100,500,1000}m_50_2500_dynamic.tif   our map kriged PER TILE at 10 m (2,500 fp/tile,
                                                     dynamic lengthscale), merged by merge_tiles.py
                                                     (bands AGB, STD, Residuals) = Table 4 "kriging at 10 m"
  kriging-sumatra_downsampled_gedi_composite-gedi_50.tif   our 100 m map kriged as one raster (AGB, Residuals, STD)
  kriging-sumatra_cci_gedi-gedi_50.tif                     ESA CCI map kriged as one raster, on the exact
                                                           reference grid (AGB, Residuals, STD)
  splits-<model>_..._-gedi.pkl            footprint-id splits (kept for the pipeline; ids are NOT unique,
                                          do not select footprints by id)
  split_map-<model>.pkl                   PIXEL split maps (1 train, 2 val, 3 test): use these for held-out metrics
```

**Revision 2 (2026-11):** CCI re-kriged after fixing the grid alignment (footprints were indexed on the
98.46 m CCI grid while the array was resampled to the 100 m reference); the 100 m kriging of our map
re-run with the current code (Table 4: 40.46 vs 40.49 published); the 10 m-kriged map now regenerated in the
repo by `merge_tiles.py` from the per-tile outputs (AGB identical to the published file; the legacy file's
"STD" band held the residuals, now fixed). `get_results.ipynb` ends with a cell printing Table 4 / G.5.
CCI input and kriging updated again on 2026-11-12 (R3.11): band 2 of `CCI_N00E100.tif` had held the std of the
two yearly AGB_SD maps (median 3.5 t/ha) instead of their mean (median 45.5 t/ha); rebuilt by
`revision-v2/scripts/r3_11_cci_meansd_input.py` (bands 1 and 3 identical to before) and re-kriged with the same
split (md5-identical). CCI post-kriging RMSE 50.08 -> 49.15 t/ha at 100 m. Previous files in
`revision-v2/archive/cci_std_of_sd/bundle/`.

**Regenerating the outputs.** Kriging: `kriging.sh` (env `PRED`, `DOWNSAMPLED`, `MAX_TRAIN_FOOTPRINTS`,
`NOISE_MODE`, `GAUSSIAN_NOISE`; `SUMATRA_OUT` redirects outputs). The per-tile 10 m route needs the 10 m
BioFiLM tiles (`predictions/nico_film/<ens>/<tile>_2021_composite.tif`, ~1 GB each, not bundled), then
`python sumatra/merge_tiles.py` (see its docstring). Copy refreshed files into `outputs/` under the same name.

## Original data sources

The files here are redistributed copies provided so the notebook is self-contained. The originals:

- **ESA CCI** biomass map (`CCI_N00E100.tif`) — ESA CCI Biomass product,
  <https://catalogue.ceda.ac.uk/uuid/6429d1aafe1e43b9b414e4a5a7f8b903/>
- **Sumatra reference** AGB rasters (`agbd_{100,500,1000}m.tif`) — from Paul B. May (2024),
  *Remote Sensing of Environment*,
  <https://www.sciencedirect.com/science/article/pii/S0034425724004103>; the reference prediction
  rasters were obtained by contacting the author.
- **GEDI** L4A footprints (`L4A_Sumatra.gpkg`) and L4B 1 km grid (`GEDI_L4B_AGBD_Sumatra.tif`) —
  NASA GEDI mission products.
- `merged_downsampled-*`, `kriging-*`, `splits-*` and `split_map-*` files are produced by this repository
  (BioFiLM inference + the Sumatra kriging pipeline + `merge_tiles.py`).
